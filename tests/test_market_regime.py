import pandas as pd

from src.research.market_regime import build_market_regime


def _index_daily() -> pd.DataFrame:
    pct_chg = [0.2] * 25
    pct_chg[22] = -0.1
    pct_chg[24] = -2.0
    return pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=25).strftime("%Y-%m-%d"),
            "index_code": ["000001.SH"] * 25,
            "close": list(range(3000, 3024)) + [3000],
            "pct_chg": pct_chg,
        }
    )


def test_build_market_regime_generates_strong_neutral_weak_and_counts():
    index_daily = _index_daily()
    limit_rows = []
    for i in range(80):
        limit_rows.append({"trade_date": "2025-01-24", "code": f"{i:06d}", "limit_type": "U", "open_times": 0, "strth": i % 5})
    for i in range(25):
        limit_rows.append({"trade_date": "2025-01-25", "code": f"3{i:05d}", "limit_type": "D", "open_times": 0})
    for i in range(40):
        limit_rows.append({"trade_date": "2025-01-25", "code": f"6{i:05d}", "limit_type": "U", "open_times": 1})
    limit_list_daily = pd.DataFrame(limit_rows)

    result = build_market_regime(index_daily, limit_list_daily)
    by_date = result.set_index("trade_date")

    assert by_date.loc["2025-01-24", "market_regime"] == "strong"
    assert by_date.loc["2025-01-24", "limit_up_count"] == 80
    assert by_date.loc["2025-01-24", "highest_streak"] == 4
    assert by_date.loc["2025-01-23", "market_regime"] == "neutral"
    assert by_date.loc["2025-01-25", "market_regime"] == "weak"
    assert by_date.loc["2025-01-25", "risk_level"] == "high"
    assert by_date.loc["2025-01-25", "limit_down_count"] == 25
    assert by_date.loc["2025-01-25", "break_board_count"] == 40
