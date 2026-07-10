from __future__ import annotations

import re

import pandas as pd

from src.database.duckdb_store import MARKET_REGIME_COLUMNS


SH_INDEX_CODE = "000001.SH"


def build_market_regime(index_daily: pd.DataFrame, limit_list_daily: pd.DataFrame) -> pd.DataFrame:
    if index_daily.empty or "trade_date" not in index_daily.columns:
        return pd.DataFrame(columns=MARKET_REGIME_COLUMNS)

    sh = index_daily.copy()
    if "index_code" in sh.columns:
        matched = sh[sh["index_code"].astype(str).str.upper() == SH_INDEX_CODE].copy()
        if not matched.empty:
            sh = matched
    if sh.empty or "close" not in sh.columns:
        return pd.DataFrame(columns=MARKET_REGIME_COLUMNS)

    sh["trade_date"] = _format_date_series(sh["trade_date"])
    sh["close"] = pd.to_numeric(sh["close"], errors="coerce")
    sh["pct_chg"] = pd.to_numeric(sh.get("pct_chg", pd.NA), errors="coerce")
    sh = sh.dropna(subset=["trade_date", "close"]).sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    if sh.empty:
        return pd.DataFrame(columns=MARKET_REGIME_COLUMNS)

    sh["ma5"] = sh["close"].rolling(5, min_periods=1).mean()
    sh["ma10"] = sh["close"].rolling(10, min_periods=1).mean()
    sh["ma20"] = sh["close"].rolling(20, min_periods=1).mean()
    sh["sh_above_ma5"] = sh["close"] > sh["ma5"]
    sh["sh_above_ma10"] = sh["close"] > sh["ma10"]
    sh["sh_above_ma20"] = sh["close"] > sh["ma20"]
    sh["index_trend_score"] = _index_trend_scores(sh)

    sentiment = _sentiment_by_date(limit_list_daily)
    rows = []
    for _, row in sh.iterrows():
        trade_date = str(row["trade_date"])
        stats = sentiment.get(trade_date, _empty_sentiment())
        sentiment_score = _sentiment_score(stats)
        total_score = float(row["index_trend_score"]) + sentiment_score
        market_regime = "strong" if total_score >= 50 else "neutral" if total_score >= 20 else "weak"
        risk_level = {"strong": "low", "neutral": "medium", "weak": "high"}[market_regime]
        rows.append(
            {
                "trade_date": trade_date,
                "sh_close": float(row["close"]),
                "sh_pct_chg": row["pct_chg"],
                "sh_above_ma5": bool(row["sh_above_ma5"]),
                "sh_above_ma10": bool(row["sh_above_ma10"]),
                "sh_above_ma20": bool(row["sh_above_ma20"]),
                "index_trend_score": float(row["index_trend_score"]),
                "limit_up_count": int(stats["limit_up_count"]),
                "limit_down_count": int(stats["limit_down_count"]),
                "break_board_count": int(stats["break_board_count"]),
                "limit_up_open_times_avg": stats["limit_up_open_times_avg"],
                "highest_streak": stats["highest_streak"],
                "sentiment_score": float(sentiment_score),
                "market_regime": market_regime,
                "risk_level": risk_level,
                "regime_reason": _reason(row, stats, sentiment_score, market_regime, risk_level),
            }
        )
    return pd.DataFrame(rows, columns=MARKET_REGIME_COLUMNS)


def _index_trend_score(row: pd.Series) -> float:
    score = 0
    if bool(row["sh_above_ma5"]):
        score += 10
    if bool(row["sh_above_ma10"]):
        score += 10
    if bool(row["sh_above_ma20"]):
        score += 20
    pct_chg = row.get("pct_chg")
    if pd.notna(pct_chg) and float(pct_chg) > 0:
        score += 10
    if pd.notna(pct_chg) and float(pct_chg) < -1.5:
        score -= 20
    return float(score)


def _index_trend_scores(frame: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    score += frame["sh_above_ma5"].astype(bool).astype(int) * 10
    score += frame["sh_above_ma10"].astype(bool).astype(int) * 10
    score += frame["sh_above_ma20"].astype(bool).astype(int) * 20
    pct_chg = pd.to_numeric(frame["pct_chg"], errors="coerce")
    score += (pct_chg > 0).astype(int) * 10
    score -= (pct_chg < -1.5).astype(int) * 20
    return score


def _sentiment_by_date(limit_list_daily: pd.DataFrame) -> dict[str, dict[str, object]]:
    if limit_list_daily.empty or "trade_date" not in limit_list_daily.columns:
        return {}
    df = limit_list_daily.copy()
    df["trade_date"] = _format_date_series(df["trade_date"])
    if "limit_type" not in df.columns:
        df["limit_type"] = ""
    if "status" not in df.columns:
        df["status"] = ""
    if "open_times" not in df.columns:
        df["open_times"] = 0
    df["limit_type_text"] = df["limit_type"].fillna("").astype(str)
    df["status_text"] = df["status"].fillna("").astype(str)
    df["open_times_num"] = pd.to_numeric(df["open_times"], errors="coerce").fillna(0)

    result: dict[str, dict[str, object]] = {}
    for trade_date, group in df.groupby("trade_date"):
        up_mask = _is_limit_up_series(group["limit_type_text"])
        down_mask = _is_limit_down_series(group["limit_type_text"])
        break_mask = (group["open_times_num"] > 0) | _is_break_board_status_series(group["status_text"])
        streak = _highest_streak(group)
        result[str(trade_date)] = {
            "limit_up_count": int(up_mask.sum()),
            "limit_down_count": int(down_mask.sum()),
            "break_board_count": int(break_mask.sum()),
            "limit_up_open_times_avg": float(group.loc[up_mask, "open_times_num"].mean()) if up_mask.any() else pd.NA,
            "highest_streak": streak,
        }
    return result


def _sentiment_score(stats: dict[str, object]) -> float:
    score = 0
    limit_up_count = int(stats["limit_up_count"])
    limit_down_count = int(stats["limit_down_count"])
    break_board_count = int(stats["break_board_count"])
    if limit_up_count >= 50:
        score += 20
    if limit_up_count >= 80:
        score += 20
    if limit_down_count >= 20:
        score -= 20
    break_ratio = break_board_count / max(limit_up_count, 1)
    if break_ratio > 0.3:
        score -= 20
    if break_ratio > 0.5:
        score -= 20
    return float(score)


def _highest_streak(group: pd.DataFrame) -> object:
    for column in ["strth", "streak", "lianban", "连板数", "连板高度"]:
        if column in group.columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if not values.empty:
                return float(values.max())
    return pd.NA


def _is_limit_up(value: object) -> bool:
    text = str(value or "").upper()
    return "涨停" in text or "U" in text or "LIMIT_UP" in text


def _is_limit_up_series(values: pd.Series) -> pd.Series:
    text = values.fillna("").astype(str).str.upper()
    return (
        text.str.contains("涨停", regex=False, na=False)
        | text.str.contains("U", regex=False, na=False)
        | text.str.contains("LIMIT_UP", regex=False, na=False)
    )


def _is_limit_down(value: object) -> bool:
    text = str(value or "").upper()
    return "跌停" in text or "D" in text or "LIMIT_DOWN" in text


def _is_limit_down_series(values: pd.Series) -> pd.Series:
    text = values.fillna("").astype(str).str.upper()
    return (
        text.str.contains("跌停", regex=False, na=False)
        | text.str.contains("D", regex=False, na=False)
        | text.str.contains("LIMIT_DOWN", regex=False, na=False)
    )


def _is_break_board_status(value: object) -> bool:
    text = str(value or "")
    return any(keyword in text for keyword in ["炸板", "打开", "开板", "broken", "break"])


def _is_break_board_status_series(values: pd.Series) -> pd.Series:
    text = values.fillna("").astype(str)
    mask = pd.Series(False, index=values.index)
    for keyword in ["炸板", "打开", "开板", "broken", "break"]:
        mask |= text.str.contains(keyword, regex=False, na=False)
    return mask


def _empty_sentiment() -> dict[str, object]:
    return {
        "limit_up_count": 0,
        "limit_down_count": 0,
        "break_board_count": 0,
        "limit_up_open_times_avg": pd.NA,
        "highest_streak": pd.NA,
    }


def _reason(
    row: pd.Series,
    stats: dict[str, object],
    sentiment_score: float,
    market_regime: str,
    risk_level: str,
) -> str:
    return (
        f"上证涨跌幅{_fmt(row.get('pct_chg'))}%，趋势分{_fmt(row.get('index_trend_score'))}；"
        f"涨停{stats['limit_up_count']}家，跌停{stats['limit_down_count']}家，"
        f"炸板{stats['break_board_count']}家，情绪分{_fmt(sentiment_score)}；"
        f"市场环境{market_regime}，风险等级{risk_level}。"
    )


def _fmt(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.2f}"


def _format_date(value: object) -> str:
    text = str(value).strip()
    if re.fullmatch(r"\d{8}", text):
        return pd.to_datetime(text, format="%Y%m%d").strftime("%Y-%m-%d")
    return pd.to_datetime(text).strftime("%Y-%m-%d")


def _format_date_series(values: pd.Series) -> pd.Series:
    text = values.fillna("").astype(str).str.strip()
    digits_8 = text.str.fullmatch(r"\d{8}", na=False)
    result = pd.Series("", index=values.index, dtype=object)
    if digits_8.any():
        result.loc[digits_8] = pd.to_datetime(
            text.loc[digits_8],
            format="%Y%m%d",
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")
    if (~digits_8).any():
        result.loc[~digits_8] = pd.to_datetime(text.loc[~digits_8], errors="coerce").dt.strftime("%Y-%m-%d")
    return result.fillna("")
