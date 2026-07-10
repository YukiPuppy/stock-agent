from pathlib import Path
import re

import duckdb
import pandas as pd

from src.backtest.trade_plan_backtester import BACKTEST_RESULT_COLUMNS, PERFORMANCE_COLUMNS as TRADE_PLAN_BACKTEST_PERFORMANCE_COLUMNS
from src.research.factor_diagnostics import FACTOR_DIAGNOSTIC_COLUMNS
from src.research.strategy_admission import STRATEGY_ADMISSION_COLUMNS
from src.trading.actual_trades import ACTUAL_TRADE_COLUMNS, normalize_actual_trades
from src.trading.daily_review import DAILY_REVIEW_COLUMNS
from src.trading.execution_review import EXECUTION_REVIEW_COLUMNS
from src.trading.period_review import PERIOD_REVIEW_COLUMNS
from src.trading.positions import POSITION_COLUMNS, POSITION_REVIEW_COLUMNS
from src.trading.trade_performance import ACTUAL_TRADE_PERFORMANCE_COLUMNS
from src.strategy.trade_plan_generator import TRADE_PLAN_COLUMNS


TRADE_CALENDAR_COLUMNS = ["trade_date", "exchange", "is_open", "pretrade_date"]
DAILY_BASIC_COLUMNS = [
    "trade_date",
    "code",
    "close",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "total_share",
    "float_share",
    "free_share",
    "total_mv",
    "circ_mv",
]
STOCK_LIMIT_COLUMNS = ["trade_date", "code", "pre_close", "up_limit", "down_limit"]
SUSPEND_DAILY_COLUMNS = ["trade_date", "code", "suspend_type", "suspend_timing"]
INDEX_DAILY_COLUMNS = [
    "trade_date",
    "index_code",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "pct_chg",
    "volume",
    "amount",
]
LIMIT_LIST_DAILY_COLUMNS = [
    "trade_date",
    "code",
    "name",
    "close",
    "pct_chg",
    "amp",
    "fc_ratio",
    "fl_ratio",
    "fd_amount",
    "first_time",
    "last_time",
    "open_times",
    "strth",
    "limit_type",
    "status",
]
MONEYFLOW_COLUMNS = [
    "trade_date",
    "code",
    "buy_sm_vol",
    "buy_sm_amount",
    "sell_sm_vol",
    "sell_sm_amount",
    "buy_md_vol",
    "buy_md_amount",
    "sell_md_vol",
    "sell_md_amount",
    "buy_lg_vol",
    "buy_lg_amount",
    "sell_lg_vol",
    "sell_lg_amount",
    "buy_elg_vol",
    "buy_elg_amount",
    "sell_elg_vol",
    "sell_elg_amount",
    "net_mf_vol",
    "net_mf_amount",
]
MONEYFLOW_FACTOR_COLUMNS = [
    "trade_date",
    "code",
    "net_mf_amount",
    "net_mf_vol",
    "main_net_amount",
    "main_net_vol",
    "big_net_amount",
    "big_net_vol",
    "small_net_amount",
    "small_net_vol",
    "main_net_amount_ratio",
    "big_net_amount_ratio",
    "small_net_amount_ratio",
    "moneyflow_score",
    "moneyflow_risk_flags",
]
SW_INDUSTRY_CLASSIFICATION_COLUMNS = [
    "industry_code",
    "industry_name",
    "level",
    "src",
    "parent_code",
    "index_code",
    "is_pub",
    "sort_code",
]
SW_DAILY_COLUMNS = [
    "trade_date",
    "industry_code",
    "industry_name",
    "open",
    "high",
    "low",
    "close",
    "change",
    "pct_change",
    "volume",
    "amount",
    "pe",
    "pb",
    "float_mv",
    "total_mv",
]
STOCK_INDUSTRY_MAP_COLUMNS = [
    "code",
    "name",
    "industry_name",
    "industry_code",
    "industry_level",
    "source",
]
INDUSTRY_STRENGTH_COLUMNS = [
    "trade_date",
    "industry_code",
    "industry_name",
    "close",
    "pct_change",
    "amount",
    "industry_return_3d",
    "industry_return_5d",
    "industry_amount_ratio_5",
    "industry_above_ma5",
    "industry_above_ma10",
    "industry_rank_pct_change",
    "industry_rank_return_5d",
    "industry_rank_amount",
    "industry_strength_score",
    "industry_strength_level",
    "industry_risk_flags",
]
DAILY_FACTOR_INDUSTRY_COLUMNS = [
    "industry_code",
    "industry_name",
    "industry_strength_score",
    "industry_strength_level",
    "industry_return_3d",
    "industry_return_5d",
    "industry_amount_ratio_5",
    "industry_risk_flags",
]
DAILY_FACTOR_MONEYFLOW_COLUMNS = [
    "net_mf_amount",
    "main_net_amount",
    "main_net_amount_ratio",
    "big_net_amount",
    "small_net_amount",
    "moneyflow_score",
    "moneyflow_risk_flags",
]
MARKET_REGIME_COLUMNS = [
    "trade_date",
    "sh_close",
    "sh_pct_chg",
    "sh_above_ma5",
    "sh_above_ma10",
    "sh_above_ma20",
    "index_trend_score",
    "limit_up_count",
    "limit_down_count",
    "break_board_count",
    "limit_up_open_times_avg",
    "highest_streak",
    "sentiment_score",
    "market_regime",
    "risk_level",
    "regime_reason",
]
DAILY_FACTOR_COLUMNS = [
    "trade_date",
    "code",
    "close",
    "pct_chg_1d",
    "pct_chg_3d",
    "pct_chg_5d",
    "pct_chg_10d",
    "ma5",
    "ma10",
    "ma20",
    "volume_ma5",
    "amount_ma5",
    "volume_ratio_5",
    "high_20",
    "low_20",
    "close_position_20",
    "above_ma5",
    "above_ma10",
    "above_ma20",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio_daily_basic",
    "pe_ttm",
    "pb",
    "total_mv",
    "circ_mv",
    "up_limit",
    "down_limit",
    "is_suspended",
    "is_limit_up_close",
    "is_limit_down_close",
    "limit_up_distance",
    "limit_down_distance",
    "net_mf_amount",
    "main_net_amount",
    "main_net_amount_ratio",
    "big_net_amount",
    "small_net_amount",
    "moneyflow_score",
    "moneyflow_risk_flags",
    "industry_code",
    "industry_name",
    "industry_strength_score",
    "industry_strength_level",
    "industry_return_3d",
    "industry_return_5d",
    "industry_amount_ratio_5",
    "industry_risk_flags",
]
DATA_QUALITY_REPORT_COLUMNS = ["check_name", "status", "issue_count", "message"]
PROVIDER_COMPARE_RESULT_COLUMNS = [
    "trade_date",
    "code",
    "field",
    "left_value",
    "right_value",
    "relative_diff",
    "status",
    "message",
]
DATA_UNIT_METADATA_COLUMNS = ["key", "value", "updated_at"]


def _normalize_daily_factor_keys(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "trade_date" in normalized.columns:
        normalized["trade_date"] = _normalize_trade_date_series(normalized["trade_date"])
    if "code" in normalized.columns:
        normalized["code"] = _normalize_stock_code_series(normalized["code"])
    return normalized


def _normalize_industry_strength_keys(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "trade_date" in normalized.columns:
        normalized["trade_date"] = _normalize_trade_date_series(normalized["trade_date"])
    if "industry_code" in normalized.columns:
        normalized["industry_code"] = normalized["industry_code"].fillna("").astype(str).str.strip()
    return normalized


def _normalize_stock_industry_map_keys(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "code" in normalized.columns:
        normalized["code"] = _normalize_stock_code_series(normalized["code"])
    if "industry_code" in normalized.columns:
        normalized["industry_code"] = normalized["industry_code"].fillna("").astype(str).str.strip()
    return normalized


def _normalize_trade_date_series(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.strip()
    digits = values.str.replace(r"\D", "", regex=True)
    normalized = digits.where(digits.str.len() == 8, "")
    needs_parse = normalized.eq("") & values.ne("")
    if needs_parse.any():
        parsed = pd.to_datetime(values[needs_parse], errors="coerce")
        normalized.loc[needs_parse] = parsed.dt.strftime("%Y%m%d").fillna("")
    return normalized


def _normalize_trade_date_value(value: object) -> str:
    return _normalize_trade_date_series(pd.Series([value])).iloc[0]


def _normalize_stock_code_series(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.strip()
    return values.map(_normalize_stock_code)


def _normalize_stock_code(value: str) -> str:
    match = re.search(r"\d{6}", value)
    if match:
        return match.group(0)
    digits = re.sub(r"\D", "", value)
    if digits:
        return digits.zfill(6)[-6:]
    return ""


def _extension_key_join_condition(table_name: str, incoming_name: str, column: str) -> str:
    if column == "trade_date":
        return f"replace({table_name}.{column}, '-', '') = replace({incoming_name}.{column}, '-', '')"
    return f"{table_name}.{column} = {incoming_name}.{column}"


class StockAgentStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def init_tables(self) -> None:
        self._ensure_parent_dir()
        with self._connect() as con:
            self._create_tables(con)

    def save_stock_basic(self, df: pd.DataFrame) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(
            df,
            ["code", "name", "market", "board", "industry", "list_status"],
        ).drop_duplicates(subset=["code"], keep="last")

        with self._connect() as con:
            self._create_tables(con)
            con.register("incoming_stock_basic", normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    """
                    DELETE FROM stock_basic
                    USING incoming_stock_basic
                    WHERE stock_basic.code = incoming_stock_basic.code
                    """
                )
                con.execute(
                    """
                    INSERT INTO stock_basic (code, name, market, board, industry, list_status)
                    SELECT code, name, market, board, industry, list_status
                    FROM incoming_stock_basic
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister("incoming_stock_basic")

    def load_stock_basic(self) -> pd.DataFrame:
        self._ensure_parent_dir()
        with self._connect() as con:
            self._create_tables(con)
            return con.execute(
                """
                SELECT code, name, market, board, industry, list_status
                FROM stock_basic
                ORDER BY code
                """
            ).fetchdf()

    def save_daily_bars(self, df: pd.DataFrame) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(
            df,
            ["trade_date", "code", "open", "high", "low", "close", "volume", "amount"],
        ).drop_duplicates(subset=["trade_date", "code"], keep="last")

        with self._connect() as con:
            self._create_tables(con)
            con.register("incoming_daily_bars", normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    """
                    DELETE FROM daily_bars
                    USING incoming_daily_bars
                    WHERE daily_bars.trade_date = incoming_daily_bars.trade_date
                      AND daily_bars.code = incoming_daily_bars.code
                    """
                )
                con.execute(
                    """
                    INSERT INTO daily_bars
                        (trade_date, code, open, high, low, close, volume, amount)
                    SELECT trade_date, code, open, high, low, close, volume, amount
                    FROM incoming_daily_bars
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister("incoming_daily_bars")

    def load_daily_bars(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        self._ensure_parent_dir()
        conditions = []
        params = []
        if start_date is not None:
            conditions.append("trade_date >= ?")
            params.append(start_date)
        if end_date is not None:
            conditions.append("trade_date <= ?")
            params.append(end_date)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT trade_date, code, open, high, low, close, volume, amount
            FROM daily_bars
            {where_clause}
            ORDER BY trade_date, code
        """

        with self._connect() as con:
            self._create_tables(con)
            return con.execute(query, params).fetchdf()

    def save_trade_calendar(self, df: pd.DataFrame) -> None:
        self._save_extension_table(
            df,
            "trade_calendar",
            TRADE_CALENDAR_COLUMNS,
            ["trade_date", "exchange"],
        )

    def load_trade_calendar(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        conditions = []
        params = []
        if start_date is not None:
            conditions.append("trade_date >= ?")
            params.append(start_date)
        if end_date is not None:
            conditions.append("trade_date <= ?")
            params.append(end_date)
        return self._load_extension_table("trade_calendar", TRADE_CALENDAR_COLUMNS, conditions, params)

    def save_daily_basic(self, df: pd.DataFrame) -> None:
        self._save_extension_table(df, "daily_basic", DAILY_BASIC_COLUMNS, ["trade_date", "code"])

    def load_daily_basic(self, trade_date: str | None = None) -> pd.DataFrame:
        conditions = []
        params = []
        if trade_date is not None:
            conditions.append("trade_date = ?")
            params.append(trade_date)
        return self._load_extension_table("daily_basic", DAILY_BASIC_COLUMNS, conditions, params)

    def save_stock_limits(self, df: pd.DataFrame) -> None:
        self._save_extension_table(df, "stock_limits", STOCK_LIMIT_COLUMNS, ["trade_date", "code"])

    def load_stock_limits(self, trade_date: str | None = None) -> pd.DataFrame:
        conditions = []
        params = []
        if trade_date is not None:
            conditions.append("trade_date = ?")
            params.append(trade_date)
        return self._load_extension_table("stock_limits", STOCK_LIMIT_COLUMNS, conditions, params)

    def save_suspend_daily(self, df: pd.DataFrame) -> None:
        self._save_extension_table(
            df,
            "suspend_daily",
            SUSPEND_DAILY_COLUMNS,
            ["trade_date", "code", "suspend_type"],
        )

    def load_suspend_daily(self, trade_date: str | None = None) -> pd.DataFrame:
        conditions = []
        params = []
        if trade_date is not None:
            conditions.append("trade_date = ?")
            params.append(trade_date)
        return self._load_extension_table("suspend_daily", SUSPEND_DAILY_COLUMNS, conditions, params)

    def save_index_daily(self, df: pd.DataFrame) -> None:
        self._save_extension_table(df, "index_daily", INDEX_DAILY_COLUMNS, ["trade_date", "index_code"])

    def load_index_daily(
        self,
        index_code: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        conditions = []
        params = []
        if index_code is not None:
            conditions.append("index_code = ?")
            params.append(index_code)
        if start_date is not None:
            conditions.append("trade_date >= ?")
            params.append(start_date)
        if end_date is not None:
            conditions.append("trade_date <= ?")
            params.append(end_date)
        return self._load_extension_table("index_daily", INDEX_DAILY_COLUMNS, conditions, params)

    def save_limit_list_daily(self, df: pd.DataFrame) -> None:
        self._save_extension_table(
            df,
            "limit_list_daily",
            LIMIT_LIST_DAILY_COLUMNS,
            ["trade_date", "code", "limit_type"],
        )

    def load_limit_list_daily(self, trade_date: str | None = None) -> pd.DataFrame:
        conditions = []
        params = []
        if trade_date is not None:
            conditions.append("trade_date = ?")
            params.append(trade_date)
        return self._load_extension_table("limit_list_daily", LIMIT_LIST_DAILY_COLUMNS, conditions, params)

    def save_moneyflow(self, df: pd.DataFrame) -> None:
        self._save_extension_table(df, "moneyflow", MONEYFLOW_COLUMNS, ["trade_date", "code"])

    def load_moneyflow(self, trade_date: str | None = None) -> pd.DataFrame:
        conditions = []
        params = []
        if trade_date is not None:
            conditions.append("trade_date = ?")
            params.append(trade_date)
        return self._load_extension_table("moneyflow", MONEYFLOW_COLUMNS, conditions, params)

    def save_moneyflow_factors(self, df: pd.DataFrame) -> None:
        self._save_extension_table(df, "moneyflow_factors", MONEYFLOW_FACTOR_COLUMNS, ["trade_date", "code"])

    def load_moneyflow_factors(
        self,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        conditions = []
        params = []
        if trade_date is not None:
            conditions.append("trade_date = ?")
            params.append(trade_date)
        if start_date is not None:
            conditions.append("trade_date >= ?")
            params.append(start_date)
        if end_date is not None:
            conditions.append("trade_date <= ?")
            params.append(end_date)
        return self._load_extension_table("moneyflow_factors", MONEYFLOW_FACTOR_COLUMNS, conditions, params)

    def save_market_regime(self, df: pd.DataFrame) -> None:
        self._save_extension_table(df, "market_regime", MARKET_REGIME_COLUMNS, ["trade_date"])

    def load_market_regime(self, trade_date: str | None = None) -> pd.DataFrame:
        conditions = []
        params = []
        if trade_date is not None:
            conditions.append("trade_date = ?")
            params.append(trade_date)
        return self._load_extension_table("market_regime", MARKET_REGIME_COLUMNS, conditions, params)

    def save_sw_industry_classification(self, df: pd.DataFrame) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(df, SW_INDUSTRY_CLASSIFICATION_COLUMNS).drop_duplicates(
            subset=["industry_code", "level", "src"],
            keep="last",
        )

        with self._connect() as con:
            self._create_tables(con)
            if normalized.empty:
                return
            con.register("incoming_sw_industry_classification", normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    """
                    DELETE FROM sw_industry_classification
                    USING (
                        SELECT DISTINCT level, src
                        FROM incoming_sw_industry_classification
                    ) incoming_scope
                    WHERE sw_industry_classification.level = incoming_scope.level
                      AND sw_industry_classification.src = incoming_scope.src
                    """
                )
                con.execute(
                    f"""
                    INSERT INTO sw_industry_classification ({", ".join(SW_INDUSTRY_CLASSIFICATION_COLUMNS)})
                    SELECT {", ".join(SW_INDUSTRY_CLASSIFICATION_COLUMNS)}
                    FROM incoming_sw_industry_classification
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister("incoming_sw_industry_classification")

    def load_sw_industry_classification(self, level: str | None = None) -> pd.DataFrame:
        conditions = []
        params = []
        if level is not None:
            conditions.append("level = ?")
            params.append(level)
        return self._load_extension_table("sw_industry_classification", SW_INDUSTRY_CLASSIFICATION_COLUMNS, conditions, params)

    def save_sw_daily(self, df: pd.DataFrame) -> None:
        self._save_extension_table(df, "sw_daily", SW_DAILY_COLUMNS, ["trade_date", "industry_code"])

    def load_sw_daily(self, trade_date: str | None = None) -> pd.DataFrame:
        conditions = []
        params = []
        if trade_date is not None:
            conditions.append("trade_date = ?")
            params.append(trade_date)
        return self._load_extension_table("sw_daily", SW_DAILY_COLUMNS, conditions, params)

    def save_stock_industry_map(self, df: pd.DataFrame) -> None:
        df = _normalize_stock_industry_map_keys(df)
        self._save_extension_table(df, "stock_industry_map", STOCK_INDUSTRY_MAP_COLUMNS, ["code"])

    def load_stock_industry_map(self) -> pd.DataFrame:
        return self._load_extension_table("stock_industry_map", STOCK_INDUSTRY_MAP_COLUMNS)

    def save_industry_strength(self, df: pd.DataFrame) -> None:
        df = _normalize_industry_strength_keys(df)
        self._save_extension_table(df, "industry_strength", INDUSTRY_STRENGTH_COLUMNS, ["trade_date", "industry_code"])

    def load_industry_strength(self, trade_date: str | None = None) -> pd.DataFrame:
        conditions = []
        params = []
        if trade_date is not None:
            conditions.append("trade_date = ?")
            params.append(_normalize_trade_date_value(trade_date))
        return self._load_extension_table("industry_strength", INDUSTRY_STRENGTH_COLUMNS, conditions, params)

    def save_data_quality_report(self, df: pd.DataFrame) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(df, DATA_QUALITY_REPORT_COLUMNS).drop_duplicates(
            subset=["check_name"],
            keep="last",
        )

        with self._connect() as con:
            self._create_tables(con)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute("DELETE FROM data_quality_report")
                if not normalized.empty:
                    con.register("incoming_data_quality_report", normalized)
                    try:
                        con.execute(
                            """
                            INSERT INTO data_quality_report (check_name, status, issue_count, message)
                            SELECT check_name, status, issue_count, message
                            FROM incoming_data_quality_report
                            """
                        )
                    finally:
                        con.unregister("incoming_data_quality_report")
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise

    def load_data_quality_report(self) -> pd.DataFrame:
        self._ensure_parent_dir()
        with self._connect() as con:
            self._create_tables(con)
            return con.execute(
                """
                SELECT check_name, status, issue_count, message
                FROM data_quality_report
                ORDER BY check_name
                """
            ).fetchdf()

    def save_provider_compare_result(self, df: pd.DataFrame) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(df, PROVIDER_COMPARE_RESULT_COLUMNS).drop_duplicates(
            subset=["trade_date", "code", "field"],
            keep="last",
        )

        with self._connect() as con:
            self._create_tables(con)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute("DELETE FROM provider_compare_result")
                if not normalized.empty:
                    con.register("incoming_provider_compare_result", normalized)
                    try:
                        con.execute(
                            """
                            INSERT INTO provider_compare_result (
                                trade_date, code, field, left_value, right_value,
                                relative_diff, status, message
                            )
                            SELECT
                                trade_date, code, field, left_value, right_value,
                                relative_diff, status, message
                            FROM incoming_provider_compare_result
                            """
                        )
                    finally:
                        con.unregister("incoming_provider_compare_result")
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise

    def load_provider_compare_result(self) -> pd.DataFrame:
        self._ensure_parent_dir()
        with self._connect() as con:
            self._create_tables(con)
            return con.execute(
                """
                SELECT
                    trade_date, code, field, left_value, right_value,
                    relative_diff, status, message
                FROM provider_compare_result
                ORDER BY trade_date, code, field
                """
            ).fetchdf()

    def save_data_unit_metadata(self, metadata: dict[str, str]) -> None:
        self._ensure_parent_dir()
        updated_at = pd.Timestamp.now("UTC").isoformat()
        normalized = pd.DataFrame(
            [{"key": str(key), "value": str(value), "updated_at": updated_at} for key, value in metadata.items()],
            columns=DATA_UNIT_METADATA_COLUMNS,
        )

        with self._connect() as con:
            self._create_tables(con)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute("DELETE FROM data_unit_metadata")
                if not normalized.empty:
                    con.register("incoming_data_unit_metadata", normalized)
                    try:
                        con.execute(
                            """
                            INSERT INTO data_unit_metadata (key, value, updated_at)
                            SELECT key, value, updated_at
                            FROM incoming_data_unit_metadata
                            """
                        )
                    finally:
                        con.unregister("incoming_data_unit_metadata")
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise

    def load_data_unit_metadata(self) -> pd.DataFrame:
        self._ensure_parent_dir()
        with self._connect() as con:
            self._create_tables(con)
            return con.execute(
                """
                SELECT key, value, updated_at
                FROM data_unit_metadata
                ORDER BY key
                """
            ).fetchdf()

    def save_daily_factors(self, df: pd.DataFrame) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(_normalize_daily_factor_keys(df), DAILY_FACTOR_COLUMNS).drop_duplicates(
            subset=["trade_date", "code"],
            keep="last",
        )

        with self._connect() as con:
            self._create_tables(con)
            con.register("incoming_daily_factors", normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    """
                    DELETE FROM daily_factors
                    USING incoming_daily_factors
                    WHERE replace(daily_factors.trade_date, '-', '') = incoming_daily_factors.trade_date
                      AND daily_factors.code = incoming_daily_factors.code
                    """
                )
                con.execute(
                    f"""
                    INSERT INTO daily_factors ({", ".join(DAILY_FACTOR_COLUMNS)})
                    SELECT {", ".join(DAILY_FACTOR_COLUMNS)}
                    FROM incoming_daily_factors
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister("incoming_daily_factors")

    def load_daily_factors(
        self,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        self._ensure_parent_dir()
        conditions = []
        params = []
        if trade_date is not None:
            conditions.append("trade_date = ?")
            params.append(_normalize_trade_date_value(trade_date))
        if start_date is not None:
            conditions.append("trade_date >= ?")
            params.append(_normalize_trade_date_value(start_date))
        if end_date is not None:
            conditions.append("trade_date <= ?")
            params.append(_normalize_trade_date_value(end_date))
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = f"""
            SELECT
                {", ".join(DAILY_FACTOR_COLUMNS)}
            FROM daily_factors
            {where_clause}
            ORDER BY trade_date, code
        """

        with self._connect() as con:
            self._create_tables(con)
            return con.execute(query, params).fetchdf()

    def save_factor_diagnostics(self, df: pd.DataFrame) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(df, FACTOR_DIAGNOSTIC_COLUMNS).drop_duplicates(
            subset=["factor_name"],
            keep="last",
        )

        with self._connect() as con:
            self._create_tables(con)
            if normalized.empty:
                return
            con.register("incoming_factor_diagnostics", normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    """
                    DELETE FROM factor_diagnostics
                    USING incoming_factor_diagnostics
                    WHERE factor_diagnostics.factor_name = incoming_factor_diagnostics.factor_name
                    """
                )
                con.execute(
                    f"""
                    INSERT INTO factor_diagnostics ({", ".join(FACTOR_DIAGNOSTIC_COLUMNS)})
                    SELECT {", ".join(FACTOR_DIAGNOSTIC_COLUMNS)}
                    FROM incoming_factor_diagnostics
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister("incoming_factor_diagnostics")

    def load_factor_diagnostics(self) -> pd.DataFrame:
        self._ensure_parent_dir()
        with self._connect() as con:
            self._create_tables(con)
            return con.execute(
                f"""
                SELECT {", ".join(FACTOR_DIAGNOSTIC_COLUMNS)}
                FROM factor_diagnostics
                ORDER BY factor_name
                """
            ).fetchdf()

    def save_candidate_pool(self, df: pd.DataFrame) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(
            df,
            [
                "trade_date",
                "code",
                "name",
                "market",
                "board",
                "close",
                "pct_chg_1d",
                "pct_chg_3d",
                "pct_chg_5d",
                "pct_chg_10d",
                "volume_ratio_5",
                "close_position_20",
                "above_ma5",
                "above_ma10",
                "above_ma20",
                "amount_ma5",
                "turnover_rate",
                "turnover_rate_f",
                "volume_ratio_daily_basic",
                "total_mv",
                "circ_mv",
                "up_limit",
                "down_limit",
                "is_suspended",
                "is_limit_up_close",
                "is_limit_down_close",
                "net_mf_amount",
                "main_net_amount",
                "main_net_amount_ratio",
                "big_net_amount",
                "small_net_amount",
                "moneyflow_score",
                "moneyflow_risk_flags",
                "industry_code",
                "industry_name",
                "industry_strength_score",
                "industry_strength_level",
                "industry_return_3d",
                "industry_return_5d",
                "industry_amount_ratio_5",
                "industry_risk_flags",
                "score",
                "rank",
                "reason",
                "strategy_names",
                "strategy_versions",
                "signal_count",
                "active_signal_count",
                "max_signal_strength",
                "total_signal_strength",
                "total_weighted_signal_strength",
                "avg_strategy_weight",
                "recommendations",
                "risk_flags",
            ],
        ).drop_duplicates(subset=["trade_date", "code"], keep="last")

        with self._connect() as con:
            self._create_tables(con)
            con.register("incoming_candidate_pool", normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    """
                    DELETE FROM candidate_pool
                    USING incoming_candidate_pool
                    WHERE candidate_pool.trade_date = incoming_candidate_pool.trade_date
                      AND candidate_pool.code = incoming_candidate_pool.code
                    """
                )
                con.execute(
                    """
                    INSERT INTO candidate_pool (
                        trade_date, code, name, market, board, close,
                        pct_chg_1d, pct_chg_3d, pct_chg_5d, pct_chg_10d,
                        volume_ratio_5, close_position_20,
                        above_ma5, above_ma10, above_ma20,
                        amount_ma5, turnover_rate, turnover_rate_f,
                        volume_ratio_daily_basic, total_mv, circ_mv,
                        up_limit, down_limit,
                        is_suspended, is_limit_up_close, is_limit_down_close,
                        net_mf_amount, main_net_amount, main_net_amount_ratio,
                        big_net_amount, small_net_amount, moneyflow_score,
                        moneyflow_risk_flags,
                        industry_code, industry_name, industry_strength_score,
                        industry_strength_level, industry_return_3d,
                        industry_return_5d, industry_amount_ratio_5,
                        industry_risk_flags,
                        score, rank, reason,
                        strategy_names, strategy_versions, signal_count,
                        active_signal_count, max_signal_strength,
                        total_signal_strength, total_weighted_signal_strength,
                        avg_strategy_weight, recommendations, risk_flags
                    )
                    SELECT
                        trade_date, code, name, market, board, close,
                        pct_chg_1d, pct_chg_3d, pct_chg_5d, pct_chg_10d,
                        volume_ratio_5, close_position_20,
                        above_ma5, above_ma10, above_ma20,
                        amount_ma5, turnover_rate, turnover_rate_f,
                        volume_ratio_daily_basic, total_mv, circ_mv,
                        up_limit, down_limit,
                        is_suspended, is_limit_up_close, is_limit_down_close,
                        net_mf_amount, main_net_amount, main_net_amount_ratio,
                        big_net_amount, small_net_amount, moneyflow_score,
                        moneyflow_risk_flags,
                        industry_code, industry_name, industry_strength_score,
                        industry_strength_level, industry_return_3d,
                        industry_return_5d, industry_amount_ratio_5,
                        industry_risk_flags,
                        score, rank, reason,
                        strategy_names, strategy_versions, signal_count,
                        active_signal_count, max_signal_strength,
                        total_signal_strength, total_weighted_signal_strength,
                        avg_strategy_weight, recommendations, risk_flags
                    FROM incoming_candidate_pool
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister("incoming_candidate_pool")

    def load_candidate_pool(self, trade_date: str | None = None) -> pd.DataFrame:
        self._ensure_parent_dir()
        params = []
        where_clause = ""
        if trade_date is not None:
            where_clause = "WHERE replace(trade_date, '-', '') = ?"
            params.append(_normalize_trade_date_value(trade_date))

        query = f"""
            SELECT
                trade_date, code, name, market, board, close,
                pct_chg_1d, pct_chg_3d, pct_chg_5d, pct_chg_10d,
                volume_ratio_5, close_position_20,
                above_ma5, above_ma10, above_ma20,
                amount_ma5, turnover_rate, turnover_rate_f,
                volume_ratio_daily_basic, total_mv, circ_mv,
                up_limit, down_limit,
                is_suspended, is_limit_up_close, is_limit_down_close,
                net_mf_amount, main_net_amount, main_net_amount_ratio,
                big_net_amount, small_net_amount, moneyflow_score,
                moneyflow_risk_flags,
                industry_code, industry_name, industry_strength_score,
                industry_strength_level, industry_return_3d,
                industry_return_5d, industry_amount_ratio_5,
                industry_risk_flags,
                score, rank, reason,
                strategy_names, strategy_versions, signal_count,
                active_signal_count, max_signal_strength,
                total_signal_strength, total_weighted_signal_strength,
                avg_strategy_weight, recommendations, risk_flags
            FROM candidate_pool
            {where_clause}
            ORDER BY trade_date, rank, code
        """

        with self._connect() as con:
            self._create_tables(con)
            return con.execute(query, params).fetchdf()

    def save_strategy_signals(self, df: pd.DataFrame) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(
            df,
            [
                "trade_date",
                "code",
                "strategy_name",
                "strategy_version",
                "signal_strength",
                "entry_reason",
                "risk_flags",
            ],
        )
        normalized = _normalize_daily_factor_keys(normalized)
        normalized["strategy_version"] = normalized["strategy_version"].fillna("v1")
        normalized = normalized.drop_duplicates(
            subset=["trade_date", "code", "strategy_name", "strategy_version"],
            keep="last",
        )

        with self._connect() as con:
            self._create_tables(con)
            con.register("incoming_strategy_signals", normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    """
                    DELETE FROM strategy_signals
                    USING incoming_strategy_signals
                    WHERE strategy_signals.trade_date = incoming_strategy_signals.trade_date
                      AND strategy_signals.code = incoming_strategy_signals.code
                      AND strategy_signals.strategy_name = incoming_strategy_signals.strategy_name
                      AND strategy_signals.strategy_version = incoming_strategy_signals.strategy_version
                    """
                )
                con.execute(
                    """
                    INSERT INTO strategy_signals (
                        trade_date, code, strategy_name, strategy_version, signal_strength,
                        entry_reason, risk_flags
                    )
                    SELECT
                        trade_date, code, strategy_name, strategy_version, signal_strength,
                        entry_reason, risk_flags
                    FROM incoming_strategy_signals
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister("incoming_strategy_signals")

    def load_strategy_signals(
        self,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        self._ensure_parent_dir()
        conditions = []
        params = []
        if trade_date is not None:
            conditions.append("replace(trade_date, '-', '') = ?")
            params.append(_normalize_trade_date_value(trade_date))
        if start_date is not None:
            conditions.append("replace(trade_date, '-', '') >= ?")
            params.append(_normalize_trade_date_value(start_date))
        if end_date is not None:
            conditions.append("replace(trade_date, '-', '') <= ?")
            params.append(_normalize_trade_date_value(end_date))
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = f"""
            SELECT
                trade_date, code, strategy_name, strategy_version, signal_strength,
                entry_reason, risk_flags
            FROM strategy_signals
            {where_clause}
            ORDER BY trade_date, signal_strength DESC, code, strategy_name, strategy_version
        """

        with self._connect() as con:
            self._create_tables(con)
            return con.execute(query, params).fetchdf()

    def save_backtest_results(self, df: pd.DataFrame) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(
            df,
            [
                "signal_date",
                "code",
                "strategy_name",
                "strategy_version",
                "signal_strength",
                "entry_date",
                "entry_open",
                "exit_date_1d",
                "exit_close_1d",
                "return_1d",
                "exit_date_3d",
                "exit_close_3d",
                "return_3d",
                "exit_date_5d",
                "exit_close_5d",
                "return_5d",
                "max_drawdown_1d",
                "max_drawdown_3d",
                "max_drawdown_5d",
                "is_valid",
                "invalid_reason",
            ],
        )
        normalized["strategy_version"] = normalized["strategy_version"].fillna("v1")
        normalized = normalized.drop_duplicates(
            subset=["signal_date", "code", "strategy_name", "strategy_version"],
            keep="last",
        )

        with self._connect() as con:
            self._create_tables(con)
            con.register("incoming_backtest_results", normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    """
                    DELETE FROM backtest_results
                    USING incoming_backtest_results
                    WHERE backtest_results.signal_date = incoming_backtest_results.signal_date
                      AND backtest_results.code = incoming_backtest_results.code
                      AND backtest_results.strategy_name = incoming_backtest_results.strategy_name
                      AND backtest_results.strategy_version = incoming_backtest_results.strategy_version
                    """
                )
                con.execute(
                    """
                    INSERT INTO backtest_results (
                        signal_date, code, strategy_name, strategy_version, signal_strength,
                        entry_date, entry_open,
                        exit_date_1d, exit_close_1d, return_1d,
                        exit_date_3d, exit_close_3d, return_3d,
                        exit_date_5d, exit_close_5d, return_5d,
                        max_drawdown_1d, max_drawdown_3d, max_drawdown_5d,
                        is_valid, invalid_reason
                    )
                    SELECT
                        signal_date, code, strategy_name, strategy_version, signal_strength,
                        entry_date, entry_open,
                        exit_date_1d, exit_close_1d, return_1d,
                        exit_date_3d, exit_close_3d, return_3d,
                        exit_date_5d, exit_close_5d, return_5d,
                        max_drawdown_1d, max_drawdown_3d, max_drawdown_5d,
                        is_valid, invalid_reason
                    FROM incoming_backtest_results
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister("incoming_backtest_results")

    def load_backtest_results(self, strategy_name: str | None = None) -> pd.DataFrame:
        self._ensure_parent_dir()
        params = []
        where_clause = ""
        if strategy_name is not None:
            where_clause = "WHERE strategy_name = ?"
            params.append(strategy_name)

        query = f"""
            SELECT
                signal_date, code, strategy_name, strategy_version, signal_strength,
                entry_date, entry_open,
                exit_date_1d, exit_close_1d, return_1d,
                exit_date_3d, exit_close_3d, return_3d,
                exit_date_5d, exit_close_5d, return_5d,
                max_drawdown_1d, max_drawdown_3d, max_drawdown_5d,
                is_valid, invalid_reason
            FROM backtest_results
            {where_clause}
            ORDER BY signal_date, code, strategy_name, strategy_version
        """

        with self._connect() as con:
            self._create_tables(con)
            return con.execute(query, params).fetchdf()

    def save_strategy_performance(self, df: pd.DataFrame) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(
            df,
            [
                "strategy_name",
                "strategy_version",
                "sample_count",
                "valid_count",
                "win_rate_1d",
                "win_rate_3d",
                "win_rate_5d",
                "avg_return_1d",
                "avg_return_3d",
                "avg_return_5d",
                "median_return_1d",
                "median_return_3d",
                "median_return_5d",
                "avg_max_drawdown_1d",
                "avg_max_drawdown_3d",
                "avg_max_drawdown_5d",
            ],
        )
        normalized["strategy_version"] = normalized["strategy_version"].fillna("v1")
        normalized = normalized.drop_duplicates(subset=["strategy_name", "strategy_version"], keep="last")

        with self._connect() as con:
            self._create_tables(con)
            con.register("incoming_strategy_performance", normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    """
                    DELETE FROM strategy_performance
                    USING incoming_strategy_performance
                    WHERE strategy_performance.strategy_name = incoming_strategy_performance.strategy_name
                      AND strategy_performance.strategy_version = incoming_strategy_performance.strategy_version
                    """
                )
                con.execute(
                    """
                    INSERT INTO strategy_performance (
                        strategy_name, strategy_version, sample_count, valid_count,
                        win_rate_1d, win_rate_3d, win_rate_5d,
                        avg_return_1d, avg_return_3d, avg_return_5d,
                        median_return_1d, median_return_3d, median_return_5d,
                        avg_max_drawdown_1d, avg_max_drawdown_3d, avg_max_drawdown_5d
                    )
                    SELECT
                        strategy_name, strategy_version, sample_count, valid_count,
                        win_rate_1d, win_rate_3d, win_rate_5d,
                        avg_return_1d, avg_return_3d, avg_return_5d,
                        median_return_1d, median_return_3d, median_return_5d,
                        avg_max_drawdown_1d, avg_max_drawdown_3d, avg_max_drawdown_5d
                    FROM incoming_strategy_performance
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister("incoming_strategy_performance")

    def load_strategy_performance(self) -> pd.DataFrame:
        self._ensure_parent_dir()
        with self._connect() as con:
            self._create_tables(con)
            return con.execute(
                """
                SELECT
                    strategy_name, strategy_version, sample_count, valid_count,
                    win_rate_1d, win_rate_3d, win_rate_5d,
                    avg_return_1d, avg_return_3d, avg_return_5d,
                    median_return_1d, median_return_3d, median_return_5d,
                    avg_max_drawdown_1d, avg_max_drawdown_3d, avg_max_drawdown_5d
                FROM strategy_performance
                ORDER BY strategy_name, strategy_version
                """
            ).fetchdf()

    def save_strategy_version_performance(self, df: pd.DataFrame) -> None:
        self._save_performance_table(df, "strategy_version_performance")

    def load_strategy_version_performance(self) -> pd.DataFrame:
        self._ensure_parent_dir()
        with self._connect() as con:
            self._create_tables(con)
            return con.execute(
                """
                SELECT
                    strategy_name, strategy_version, sample_count, valid_count,
                    win_rate_1d, win_rate_3d, win_rate_5d,
                    avg_return_1d, avg_return_3d, avg_return_5d,
                    median_return_1d, median_return_3d, median_return_5d,
                    avg_max_drawdown_1d, avg_max_drawdown_3d, avg_max_drawdown_5d
                FROM strategy_version_performance
                ORDER BY strategy_name, strategy_version
                """
            ).fetchdf()

    def save_parameter_search_performance(self, df: pd.DataFrame) -> None:
        self._save_performance_table(df, "parameter_search_performance")

    def load_parameter_search_performance(self) -> pd.DataFrame:
        return self._load_performance_table("parameter_search_performance")

    def save_strategy_version_evaluation(self, df: pd.DataFrame) -> None:
        self._save_evaluation_table(df, "strategy_version_evaluation")

    def load_strategy_version_evaluation(self) -> pd.DataFrame:
        return self._load_evaluation_table("strategy_version_evaluation")

    def save_parameter_search_results(self, df: pd.DataFrame) -> None:
        self._save_evaluation_table(df, "parameter_search_results")

    def load_parameter_search_results(self) -> pd.DataFrame:
        return self._load_evaluation_table("parameter_search_results")

    def save_parameter_search_backtest_results(self, df: pd.DataFrame) -> None:
        self._save_backtest_table(df, "parameter_search_backtest_results")

    def load_parameter_search_backtest_results(self) -> pd.DataFrame:
        return self._load_backtest_table("parameter_search_backtest_results")

    def save_walk_forward_validation(self, df: pd.DataFrame) -> None:
        self._ensure_parent_dir()
        columns = [
            "strategy_name",
            "strategy_version",
            "train_valid_count",
            "train_win_rate_3d",
            "train_avg_return_3d",
            "train_avg_drawdown_3d",
            "validation_valid_count",
            "validation_win_rate_3d",
            "validation_avg_return_3d",
            "validation_avg_drawdown_3d",
            "return_decay",
            "win_rate_decay",
            "drawdown_worsening",
            "stability_score",
            "overfit_risk",
            "validation_status",
            "validation_reason",
        ]
        normalized = self._normalize_dataframe(df, columns)
        normalized["strategy_version"] = normalized["strategy_version"].fillna("v1")
        normalized = normalized.drop_duplicates(subset=["strategy_name", "strategy_version"], keep="last")

        with self._connect() as con:
            self._create_tables(con)
            con.register("incoming_walk_forward_validation", normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    """
                    DELETE FROM walk_forward_validation
                    USING incoming_walk_forward_validation
                    WHERE walk_forward_validation.strategy_name = incoming_walk_forward_validation.strategy_name
                      AND walk_forward_validation.strategy_version = incoming_walk_forward_validation.strategy_version
                    """
                )
                con.execute(
                    """
                    INSERT INTO walk_forward_validation (
                        strategy_name, strategy_version,
                        train_valid_count, train_win_rate_3d, train_avg_return_3d, train_avg_drawdown_3d,
                        validation_valid_count, validation_win_rate_3d, validation_avg_return_3d,
                        validation_avg_drawdown_3d, return_decay, win_rate_decay, drawdown_worsening,
                        stability_score, overfit_risk, validation_status, validation_reason
                    )
                    SELECT
                        strategy_name, strategy_version,
                        train_valid_count, train_win_rate_3d, train_avg_return_3d, train_avg_drawdown_3d,
                        validation_valid_count, validation_win_rate_3d, validation_avg_return_3d,
                        validation_avg_drawdown_3d, return_decay, win_rate_decay, drawdown_worsening,
                        stability_score, overfit_risk, validation_status, validation_reason
                    FROM incoming_walk_forward_validation
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister("incoming_walk_forward_validation")

    def load_walk_forward_validation(self) -> pd.DataFrame:
        self._ensure_parent_dir()
        with self._connect() as con:
            self._create_tables(con)
            return con.execute(
                """
                SELECT
                    strategy_name, strategy_version,
                    train_valid_count, train_win_rate_3d, train_avg_return_3d, train_avg_drawdown_3d,
                    validation_valid_count, validation_win_rate_3d, validation_avg_return_3d,
                    validation_avg_drawdown_3d, return_decay, win_rate_decay, drawdown_worsening,
                    stability_score, overfit_risk, validation_status, validation_reason
                FROM walk_forward_validation
                ORDER BY stability_score DESC, strategy_name, strategy_version
                """
            ).fetchdf()

    def save_strategy_admission(self, df: pd.DataFrame) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(df, STRATEGY_ADMISSION_COLUMNS)
        normalized["strategy_version"] = normalized["strategy_version"].fillna("v1")
        normalized = normalized.drop_duplicates(subset=["strategy_name", "strategy_version"], keep="last")

        with self._connect() as con:
            self._create_tables(con)
            con.register("incoming_strategy_admission", normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    """
                    DELETE FROM strategy_admission
                    USING incoming_strategy_admission
                    WHERE strategy_admission.strategy_name = incoming_strategy_admission.strategy_name
                      AND strategy_admission.strategy_version = incoming_strategy_admission.strategy_version
                    """
                )
                con.execute(
                    f"""
                    INSERT INTO strategy_admission ({", ".join(STRATEGY_ADMISSION_COLUMNS)})
                    SELECT {", ".join(STRATEGY_ADMISSION_COLUMNS)}
                    FROM incoming_strategy_admission
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister("incoming_strategy_admission")

    def load_strategy_admission(self) -> pd.DataFrame:
        self._ensure_parent_dir()
        with self._connect() as con:
            self._create_tables(con)
            return con.execute(
                f"""
                SELECT {", ".join(STRATEGY_ADMISSION_COLUMNS)}
                FROM strategy_admission
                ORDER BY admission_score DESC, strategy_name, strategy_version
                """
            ).fetchdf()

    def _save_evaluation_table(self, df: pd.DataFrame, table_name: str) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(
            df,
            [
                "strategy_name",
                "strategy_version",
                "sample_count",
                "valid_count",
                "win_rate_3d",
                "avg_return_3d",
                "median_return_3d",
                "avg_max_drawdown_3d",
                "evaluation_score",
                "evaluation_status",
                "risk_level",
                "recommendation",
                "evaluation_reason",
            ],
        )
        normalized["strategy_version"] = normalized["strategy_version"].fillna("v1")
        normalized = normalized.drop_duplicates(subset=["strategy_name", "strategy_version"], keep="last")

        with self._connect() as con:
            self._create_tables(con)
            incoming_name = f"incoming_{table_name}"
            con.register(incoming_name, normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    f"""
                    DELETE FROM {table_name}
                    USING {incoming_name}
                    WHERE {table_name}.strategy_name = {incoming_name}.strategy_name
                      AND {table_name}.strategy_version = {incoming_name}.strategy_version
                    """
                )
                con.execute(
                    f"""
                    INSERT INTO {table_name} (
                        strategy_name, strategy_version, sample_count, valid_count,
                        win_rate_3d, avg_return_3d, median_return_3d, avg_max_drawdown_3d,
                        evaluation_score, evaluation_status, risk_level, recommendation,
                        evaluation_reason
                    )
                    SELECT
                        strategy_name, strategy_version, sample_count, valid_count,
                        win_rate_3d, avg_return_3d, median_return_3d, avg_max_drawdown_3d,
                        evaluation_score, evaluation_status, risk_level, recommendation,
                        evaluation_reason
                    FROM {incoming_name}
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister(incoming_name)

    def _load_evaluation_table(self, table_name: str) -> pd.DataFrame:
        self._ensure_parent_dir()
        with self._connect() as con:
            self._create_tables(con)
            return con.execute(
                f"""
                SELECT
                    strategy_name, strategy_version, sample_count, valid_count,
                    win_rate_3d, avg_return_3d, median_return_3d, avg_max_drawdown_3d,
                    evaluation_score, evaluation_status, risk_level, recommendation,
                    evaluation_reason
                FROM {table_name}
                ORDER BY evaluation_score DESC, strategy_name, strategy_version
                """
            ).fetchdf()

    def _save_backtest_table(self, df: pd.DataFrame, table_name: str) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(
            df,
            [
                "signal_date",
                "code",
                "strategy_name",
                "strategy_version",
                "signal_strength",
                "entry_date",
                "entry_open",
                "exit_date_1d",
                "exit_close_1d",
                "return_1d",
                "exit_date_3d",
                "exit_close_3d",
                "return_3d",
                "exit_date_5d",
                "exit_close_5d",
                "return_5d",
                "max_drawdown_1d",
                "max_drawdown_3d",
                "max_drawdown_5d",
                "is_valid",
                "invalid_reason",
            ],
        )
        normalized["strategy_version"] = normalized["strategy_version"].fillna("v1")
        normalized = normalized.drop_duplicates(
            subset=["signal_date", "code", "strategy_name", "strategy_version"],
            keep="last",
        )

        with self._connect() as con:
            self._create_tables(con)
            incoming_name = f"incoming_{table_name}"
            con.register(incoming_name, normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    f"""
                    DELETE FROM {table_name}
                    USING {incoming_name}
                    WHERE {table_name}.signal_date = {incoming_name}.signal_date
                      AND {table_name}.code = {incoming_name}.code
                      AND {table_name}.strategy_name = {incoming_name}.strategy_name
                      AND {table_name}.strategy_version = {incoming_name}.strategy_version
                    """
                )
                con.execute(
                    f"""
                    INSERT INTO {table_name} (
                        signal_date, code, strategy_name, strategy_version, signal_strength,
                        entry_date, entry_open,
                        exit_date_1d, exit_close_1d, return_1d,
                        exit_date_3d, exit_close_3d, return_3d,
                        exit_date_5d, exit_close_5d, return_5d,
                        max_drawdown_1d, max_drawdown_3d, max_drawdown_5d,
                        is_valid, invalid_reason
                    )
                    SELECT
                        signal_date, code, strategy_name, strategy_version, signal_strength,
                        entry_date, entry_open,
                        exit_date_1d, exit_close_1d, return_1d,
                        exit_date_3d, exit_close_3d, return_3d,
                        exit_date_5d, exit_close_5d, return_5d,
                        max_drawdown_1d, max_drawdown_3d, max_drawdown_5d,
                        is_valid, invalid_reason
                    FROM {incoming_name}
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister(incoming_name)

    def _load_backtest_table(self, table_name: str) -> pd.DataFrame:
        self._ensure_parent_dir()
        with self._connect() as con:
            self._create_tables(con)
            return con.execute(
                f"""
                SELECT
                    signal_date, code, strategy_name, strategy_version, signal_strength,
                    entry_date, entry_open,
                    exit_date_1d, exit_close_1d, return_1d,
                    exit_date_3d, exit_close_3d, return_3d,
                    exit_date_5d, exit_close_5d, return_5d,
                    max_drawdown_1d, max_drawdown_3d, max_drawdown_5d,
                    is_valid, invalid_reason
                FROM {table_name}
                ORDER BY signal_date, code, strategy_name, strategy_version
                """
            ).fetchdf()

    def _save_performance_table(self, df: pd.DataFrame, table_name: str) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(
            df,
            [
                "strategy_name",
                "strategy_version",
                "sample_count",
                "valid_count",
                "win_rate_1d",
                "win_rate_3d",
                "win_rate_5d",
                "avg_return_1d",
                "avg_return_3d",
                "avg_return_5d",
                "median_return_1d",
                "median_return_3d",
                "median_return_5d",
                "avg_max_drawdown_1d",
                "avg_max_drawdown_3d",
                "avg_max_drawdown_5d",
            ],
        )
        normalized["strategy_version"] = normalized["strategy_version"].fillna("v1")
        normalized = normalized.drop_duplicates(subset=["strategy_name", "strategy_version"], keep="last")

        with self._connect() as con:
            self._create_tables(con)
            incoming_name = f"incoming_{table_name}"
            con.register(incoming_name, normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    f"""
                    DELETE FROM {table_name}
                    USING {incoming_name}
                    WHERE {table_name}.strategy_name = {incoming_name}.strategy_name
                      AND {table_name}.strategy_version = {incoming_name}.strategy_version
                    """
                )
                con.execute(
                    f"""
                    INSERT INTO {table_name} (
                        strategy_name, strategy_version, sample_count, valid_count,
                        win_rate_1d, win_rate_3d, win_rate_5d,
                        avg_return_1d, avg_return_3d, avg_return_5d,
                        median_return_1d, median_return_3d, median_return_5d,
                        avg_max_drawdown_1d, avg_max_drawdown_3d, avg_max_drawdown_5d
                    )
                    SELECT
                        strategy_name, strategy_version, sample_count, valid_count,
                        win_rate_1d, win_rate_3d, win_rate_5d,
                        avg_return_1d, avg_return_3d, avg_return_5d,
                        median_return_1d, median_return_3d, median_return_5d,
                        avg_max_drawdown_1d, avg_max_drawdown_3d, avg_max_drawdown_5d
                    FROM {incoming_name}
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister(incoming_name)

    def _load_performance_table(self, table_name: str) -> pd.DataFrame:
        self._ensure_parent_dir()
        with self._connect() as con:
            self._create_tables(con)
            return con.execute(
                f"""
                SELECT
                    strategy_name, strategy_version, sample_count, valid_count,
                    win_rate_1d, win_rate_3d, win_rate_5d,
                    avg_return_1d, avg_return_3d, avg_return_5d,
                    median_return_1d, median_return_3d, median_return_5d,
                    avg_max_drawdown_1d, avg_max_drawdown_3d, avg_max_drawdown_5d
                FROM {table_name}
                ORDER BY strategy_name, strategy_version
                """
            ).fetchdf()

    def save_trade_plan(self, df: pd.DataFrame) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(
            df,
            [
                "trade_date",
                "code",
                "name",
                "rank",
                "strategy_names",
                "strategy_versions",
                "active_signal_count",
                "avg_strategy_weight",
                "recommendations",
                "risk_flags",
                "close",
                "strategy_type",
                "action",
                "entry_low",
                "entry_high",
                "position_low",
                "position_high",
                "stop_loss",
                "take_profit_1",
                "take_profit_2",
                "invalid_condition",
                "t_plus_1_risk",
                "plan_reason",
            ],
        ).drop_duplicates(subset=["trade_date", "code"], keep="last")

        with self._connect() as con:
            self._create_tables(con)
            con.register("incoming_trade_plan", normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    """
                    DELETE FROM trade_plan
                    USING incoming_trade_plan
                    WHERE trade_plan.trade_date = incoming_trade_plan.trade_date
                      AND trade_plan.code = incoming_trade_plan.code
                    """
                )
                con.execute(
                    """
                    INSERT INTO trade_plan (
                        trade_date, code, name, rank,
                        strategy_names, strategy_versions, active_signal_count,
                        avg_strategy_weight, recommendations, risk_flags,
                        close,
                        strategy_type, action,
                        entry_low, entry_high, position_low, position_high,
                        stop_loss, take_profit_1, take_profit_2,
                        invalid_condition, t_plus_1_risk, plan_reason
                    )
                    SELECT
                        trade_date, code, name, rank,
                        strategy_names, strategy_versions, active_signal_count,
                        avg_strategy_weight, recommendations, risk_flags,
                        close,
                        strategy_type, action,
                        entry_low, entry_high, position_low, position_high,
                        stop_loss, take_profit_1, take_profit_2,
                        invalid_condition, t_plus_1_risk, plan_reason
                    FROM incoming_trade_plan
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister("incoming_trade_plan")

    def load_trade_plan(self, trade_date: str | None = None) -> pd.DataFrame:
        self._ensure_parent_dir()
        params = []
        where_clause = ""
        if trade_date is not None:
            where_clause = "WHERE trade_date = ?"
            params.append(trade_date)

        query = f"""
            SELECT
                trade_date, code, name, rank,
                strategy_names, strategy_versions, active_signal_count,
                avg_strategy_weight, recommendations, risk_flags,
                close,
                strategy_type, action,
                entry_low, entry_high, position_low, position_high,
                stop_loss, take_profit_1, take_profit_2,
                invalid_condition, t_plus_1_risk, plan_reason
            FROM trade_plan
            {where_clause}
            ORDER BY trade_date, rank, code
        """

        with self._connect() as con:
            self._create_tables(con)
            return con.execute(query, params).fetchdf()

    def save_historical_trade_plans(self, df: pd.DataFrame) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(df, TRADE_PLAN_COLUMNS)
        for column in ["trade_date", "code", "strategy_names", "strategy_versions"]:
            normalized[column] = normalized[column].fillna("").astype(str)
        normalized = normalized.drop_duplicates(
            subset=["trade_date", "code", "strategy_names", "strategy_versions"],
            keep="last",
        )

        with self._connect() as con:
            self._create_tables(con)
            con.register("incoming_historical_trade_plans", normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    """
                    DELETE FROM historical_trade_plans
                    USING incoming_historical_trade_plans
                    WHERE historical_trade_plans.trade_date = incoming_historical_trade_plans.trade_date
                      AND historical_trade_plans.code = incoming_historical_trade_plans.code
                      AND historical_trade_plans.strategy_names = incoming_historical_trade_plans.strategy_names
                      AND historical_trade_plans.strategy_versions = incoming_historical_trade_plans.strategy_versions
                    """
                )
                con.execute(
                    f"""
                    INSERT INTO historical_trade_plans ({", ".join(TRADE_PLAN_COLUMNS)})
                    SELECT {", ".join(TRADE_PLAN_COLUMNS)}
                    FROM incoming_historical_trade_plans
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister("incoming_historical_trade_plans")

    def load_historical_trade_plans(self, trade_date: str | None = None) -> pd.DataFrame:
        self._ensure_parent_dir()
        params = []
        where_clause = ""
        if trade_date is not None:
            where_clause = "WHERE trade_date = ?"
            params.append(trade_date)
        with self._connect() as con:
            self._create_tables(con)
            return con.execute(
                f"""
                SELECT {", ".join(TRADE_PLAN_COLUMNS)}
                FROM historical_trade_plans
                {where_clause}
                ORDER BY trade_date, rank, code, strategy_names, strategy_versions
                """,
                params,
            ).fetchdf()

    def save_trade_plan_backtest_results(self, df: pd.DataFrame) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(df, BACKTEST_RESULT_COLUMNS)
        for column in ["plan_date", "code", "strategy_names", "strategy_versions"]:
            normalized[column] = normalized[column].fillna("").astype(str)
        normalized = normalized.drop_duplicates(
            subset=["plan_date", "code", "strategy_names", "strategy_versions"],
            keep="last",
        )

        with self._connect() as con:
            self._create_tables(con)
            con.register("incoming_trade_plan_backtest_results", normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    """
                    DELETE FROM trade_plan_backtest_results
                    USING incoming_trade_plan_backtest_results
                    WHERE trade_plan_backtest_results.plan_date = incoming_trade_plan_backtest_results.plan_date
                      AND trade_plan_backtest_results.code = incoming_trade_plan_backtest_results.code
                      AND trade_plan_backtest_results.strategy_names = incoming_trade_plan_backtest_results.strategy_names
                      AND trade_plan_backtest_results.strategy_versions = incoming_trade_plan_backtest_results.strategy_versions
                    """
                )
                con.execute(
                    f"""
                    INSERT INTO trade_plan_backtest_results ({", ".join(BACKTEST_RESULT_COLUMNS)})
                    SELECT {", ".join(BACKTEST_RESULT_COLUMNS)}
                    FROM incoming_trade_plan_backtest_results
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister("incoming_trade_plan_backtest_results")

    def load_trade_plan_backtest_results(self) -> pd.DataFrame:
        self._ensure_parent_dir()
        with self._connect() as con:
            self._create_tables(con)
            return con.execute(
                f"""
                SELECT {", ".join(BACKTEST_RESULT_COLUMNS)}
                FROM trade_plan_backtest_results
                ORDER BY plan_date, code, strategy_names, strategy_versions
                """
            ).fetchdf()

    def save_trade_plan_backtest_performance(self, df: pd.DataFrame) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(df, TRADE_PLAN_BACKTEST_PERFORMANCE_COLUMNS)
        for column in ["strategy_names", "strategy_versions", "action"]:
            normalized[column] = normalized[column].fillna("").astype(str)
        normalized = normalized.drop_duplicates(
            subset=["strategy_names", "strategy_versions", "action"],
            keep="last",
        )

        with self._connect() as con:
            self._create_tables(con)
            con.register("incoming_trade_plan_backtest_performance", normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    """
                    DELETE FROM trade_plan_backtest_performance
                    USING incoming_trade_plan_backtest_performance
                    WHERE trade_plan_backtest_performance.strategy_names = incoming_trade_plan_backtest_performance.strategy_names
                      AND trade_plan_backtest_performance.strategy_versions = incoming_trade_plan_backtest_performance.strategy_versions
                      AND trade_plan_backtest_performance.action = incoming_trade_plan_backtest_performance.action
                    """
                )
                con.execute(
                    f"""
                    INSERT INTO trade_plan_backtest_performance ({", ".join(TRADE_PLAN_BACKTEST_PERFORMANCE_COLUMNS)})
                    SELECT {", ".join(TRADE_PLAN_BACKTEST_PERFORMANCE_COLUMNS)}
                    FROM incoming_trade_plan_backtest_performance
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister("incoming_trade_plan_backtest_performance")

    def load_trade_plan_backtest_performance(self) -> pd.DataFrame:
        self._ensure_parent_dir()
        with self._connect() as con:
            self._create_tables(con)
            return con.execute(
                f"""
                SELECT {", ".join(TRADE_PLAN_BACKTEST_PERFORMANCE_COLUMNS)}
                FROM trade_plan_backtest_performance
                ORDER BY strategy_names, strategy_versions, action
                """
            ).fetchdf()

    def save_actual_trades(self, df: pd.DataFrame) -> None:
        self._ensure_parent_dir()
        normalized = normalize_actual_trades(df).drop_duplicates(
            subset=["trade_date", "trade_time", "code", "side", "price", "volume"],
            keep="last",
        )

        with self._connect() as con:
            self._create_tables(con)
            con.register("incoming_actual_trades", normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    """
                    DELETE FROM actual_trades
                    USING incoming_actual_trades
                    WHERE actual_trades.trade_date IS NOT DISTINCT FROM incoming_actual_trades.trade_date
                      AND actual_trades.trade_time IS NOT DISTINCT FROM incoming_actual_trades.trade_time
                      AND actual_trades.code IS NOT DISTINCT FROM incoming_actual_trades.code
                      AND actual_trades.side IS NOT DISTINCT FROM incoming_actual_trades.side
                      AND actual_trades.price IS NOT DISTINCT FROM incoming_actual_trades.price
                      AND actual_trades.volume IS NOT DISTINCT FROM incoming_actual_trades.volume
                    """
                )
                con.execute(
                    """
                    INSERT INTO actual_trades (
                        trade_date, trade_time, code, name, side,
                        price, volume, amount, position_ratio,
                        strategy_name, plan_rank, is_follow_plan, reason, note
                    )
                    SELECT
                        trade_date, trade_time, code, name, side,
                        price, volume, amount, position_ratio,
                        strategy_name, plan_rank, is_follow_plan, reason, note
                    FROM incoming_actual_trades
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister("incoming_actual_trades")

    def load_actual_trades(self, trade_date: str | None = None) -> pd.DataFrame:
        self._ensure_parent_dir()
        params = []
        where_clause = ""
        if trade_date is not None:
            where_clause = "WHERE trade_date = ?"
            params.append(trade_date)

        query = f"""
            SELECT
                trade_date, trade_time, code, name, side,
                price, volume, amount, position_ratio,
                strategy_name, plan_rank, is_follow_plan, reason, note
            FROM actual_trades
            {where_clause}
            ORDER BY trade_date, trade_time, code, side
        """

        with self._connect() as con:
            self._create_tables(con)
            return con.execute(query, params).fetchdf()

    def save_execution_review(self, df: pd.DataFrame) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(df, EXECUTION_REVIEW_COLUMNS).drop_duplicates(
            subset=["trade_date", "code", "side", "trade_time"],
            keep="last",
        )

        with self._connect() as con:
            self._create_tables(con)
            con.register("incoming_execution_review", normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    """
                    DELETE FROM execution_review
                    USING incoming_execution_review
                    WHERE execution_review.trade_date IS NOT DISTINCT FROM incoming_execution_review.trade_date
                      AND execution_review.code IS NOT DISTINCT FROM incoming_execution_review.code
                      AND execution_review.side IS NOT DISTINCT FROM incoming_execution_review.side
                      AND execution_review.trade_time IS NOT DISTINCT FROM incoming_execution_review.trade_time
                    """
                )
                con.execute(
                    """
                    INSERT INTO execution_review (
                        trade_date, trade_time, code, name, side,
                        actual_price, actual_volume, actual_amount, position_ratio,
                        plan_rank, planned_action, entry_low, entry_high,
                        position_low, position_high, stop_loss,
                        take_profit_1, take_profit_2,
                        plan_match_status, execution_status,
                        execution_flags, execution_comment
                    )
                    SELECT
                        trade_date, trade_time, code, name, side,
                        actual_price, actual_volume, actual_amount, position_ratio,
                        plan_rank, planned_action, entry_low, entry_high,
                        position_low, position_high, stop_loss,
                        take_profit_1, take_profit_2,
                        plan_match_status, execution_status,
                        execution_flags, execution_comment
                    FROM incoming_execution_review
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister("incoming_execution_review")

    def load_execution_review(self, trade_date: str | None = None) -> pd.DataFrame:
        self._ensure_parent_dir()
        params = []
        where_clause = ""
        if trade_date is not None:
            where_clause = "WHERE trade_date = ?"
            params.append(trade_date)

        query = f"""
            SELECT
                trade_date, trade_time, code, name, side,
                actual_price, actual_volume, actual_amount, position_ratio,
                plan_rank, planned_action, entry_low, entry_high,
                position_low, position_high, stop_loss,
                take_profit_1, take_profit_2,
                plan_match_status, execution_status,
                execution_flags, execution_comment
            FROM execution_review
            {where_clause}
            ORDER BY trade_date, trade_time, code, side
        """

        with self._connect() as con:
            self._create_tables(con)
            return con.execute(query, params).fetchdf()

    def save_actual_trade_performance(self, df: pd.DataFrame) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(df, ACTUAL_TRADE_PERFORMANCE_COLUMNS).drop_duplicates(
            subset=["trade_date", "trade_time", "code", "side", "entry_price", "entry_volume"],
            keep="last",
        )

        with self._connect() as con:
            self._create_tables(con)
            con.register("incoming_actual_trade_performance", normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    """
                    DELETE FROM actual_trade_performance
                    USING incoming_actual_trade_performance
                    WHERE actual_trade_performance.trade_date IS NOT DISTINCT FROM incoming_actual_trade_performance.trade_date
                      AND actual_trade_performance.trade_time IS NOT DISTINCT FROM incoming_actual_trade_performance.trade_time
                      AND actual_trade_performance.code IS NOT DISTINCT FROM incoming_actual_trade_performance.code
                      AND actual_trade_performance.side IS NOT DISTINCT FROM incoming_actual_trade_performance.side
                      AND actual_trade_performance.entry_price IS NOT DISTINCT FROM incoming_actual_trade_performance.entry_price
                      AND actual_trade_performance.entry_volume IS NOT DISTINCT FROM incoming_actual_trade_performance.entry_volume
                    """
                )
                con.execute(
                    """
                    INSERT INTO actual_trade_performance (
                        trade_date, trade_time, code, name, side,
                        entry_price, entry_volume, entry_amount, position_ratio,
                        strategy_name, plan_rank, plan_match_status,
                        execution_status, execution_flags,
                        return_1d, return_3d, return_5d,
                        max_drawdown_1d, max_drawdown_3d, max_drawdown_5d,
                        max_favorable_1d, max_favorable_3d, max_favorable_5d,
                        is_valid, invalid_reason, performance_comment
                    )
                    SELECT
                        trade_date, trade_time, code, name, side,
                        entry_price, entry_volume, entry_amount, position_ratio,
                        strategy_name, plan_rank, plan_match_status,
                        execution_status, execution_flags,
                        return_1d, return_3d, return_5d,
                        max_drawdown_1d, max_drawdown_3d, max_drawdown_5d,
                        max_favorable_1d, max_favorable_3d, max_favorable_5d,
                        is_valid, invalid_reason, performance_comment
                    FROM incoming_actual_trade_performance
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister("incoming_actual_trade_performance")

    def load_actual_trade_performance(self, trade_date: str | None = None) -> pd.DataFrame:
        self._ensure_parent_dir()
        params = []
        where_clause = ""
        if trade_date is not None:
            where_clause = "WHERE trade_date = ?"
            params.append(trade_date)

        query = f"""
            SELECT
                trade_date, trade_time, code, name, side,
                entry_price, entry_volume, entry_amount, position_ratio,
                strategy_name, plan_rank, plan_match_status,
                execution_status, execution_flags,
                return_1d, return_3d, return_5d,
                max_drawdown_1d, max_drawdown_3d, max_drawdown_5d,
                max_favorable_1d, max_favorable_3d, max_favorable_5d,
                is_valid, invalid_reason, performance_comment
            FROM actual_trade_performance
            {where_clause}
            ORDER BY trade_date, trade_time, code, side
        """

        with self._connect() as con:
            self._create_tables(con)
            return con.execute(query, params).fetchdf()

    def save_positions(self, df: pd.DataFrame) -> None:
        self._save_position_table(df, "positions", POSITION_COLUMNS)

    def load_positions(self, as_of_date: str | None = None) -> pd.DataFrame:
        return self._load_position_table("positions", POSITION_COLUMNS, as_of_date)

    def save_position_review(self, df: pd.DataFrame) -> None:
        self._save_position_table(df, "position_review", POSITION_REVIEW_COLUMNS)

    def load_position_review(self, as_of_date: str | None = None) -> pd.DataFrame:
        return self._load_position_table("position_review", POSITION_REVIEW_COLUMNS, as_of_date)

    def _save_position_table(self, df: pd.DataFrame, table_name: str, columns: list[str]) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(df, columns).drop_duplicates(
            subset=["as_of_date", "code"],
            keep="last",
        )

        with self._connect() as con:
            self._create_tables(con)
            incoming_name = f"incoming_{table_name}"
            con.register(incoming_name, normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    f"""
                    DELETE FROM {table_name}
                    USING {incoming_name}
                    WHERE {table_name}.as_of_date IS NOT DISTINCT FROM {incoming_name}.as_of_date
                      AND {table_name}.code IS NOT DISTINCT FROM {incoming_name}.code
                    """
                )
                con.execute(
                    f"""
                    INSERT INTO {table_name} ({", ".join(columns)})
                    SELECT {", ".join(columns)}
                    FROM {incoming_name}
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister(incoming_name)

    def _load_position_table(
        self,
        table_name: str,
        columns: list[str],
        as_of_date: str | None = None,
    ) -> pd.DataFrame:
        self._ensure_parent_dir()
        params = []
        where_clause = ""
        if as_of_date is not None:
            where_clause = "WHERE as_of_date = ?"
            params.append(as_of_date)

        query = f"""
            SELECT {", ".join(columns)}
            FROM {table_name}
            {where_clause}
            ORDER BY as_of_date, code
        """

        with self._connect() as con:
            self._create_tables(con)
            return con.execute(query, params).fetchdf()

    def save_daily_review(self, df: pd.DataFrame) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(df, DAILY_REVIEW_COLUMNS).drop_duplicates(
            subset=["trade_date"],
            keep="last",
        )

        with self._connect() as con:
            self._create_tables(con)
            con.register("incoming_daily_review", normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    """
                    DELETE FROM daily_review
                    USING incoming_daily_review
                    WHERE daily_review.trade_date IS NOT DISTINCT FROM incoming_daily_review.trade_date
                    """
                )
                con.execute(
                    """
                    INSERT INTO daily_review (
                        trade_date, actual_trade_count, buy_count, sell_count,
                        planned_trade_count, matched_plan_count, off_plan_count,
                        follow_plan_count, deviation_count, chase_count,
                        over_position_count, bought_watch_only_count,
                        execution_score, main_issues, review_summary,
                        next_action_suggestion, valid_performance_count,
                        avg_return_1d, avg_return_3d, avg_return_5d,
                        plan_trade_avg_return_3d, off_plan_avg_return_3d,
                        chase_trade_count, chase_avg_return_3d
                    )
                    SELECT
                        trade_date, actual_trade_count, buy_count, sell_count,
                        planned_trade_count, matched_plan_count, off_plan_count,
                        follow_plan_count, deviation_count, chase_count,
                        over_position_count, bought_watch_only_count,
                        execution_score, main_issues, review_summary,
                        next_action_suggestion, valid_performance_count,
                        avg_return_1d, avg_return_3d, avg_return_5d,
                        plan_trade_avg_return_3d, off_plan_avg_return_3d,
                        chase_trade_count, chase_avg_return_3d
                    FROM incoming_daily_review
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister("incoming_daily_review")

    def load_daily_review(self, trade_date: str | None = None) -> pd.DataFrame:
        self._ensure_parent_dir()
        params = []
        where_clause = ""
        if trade_date is not None:
            where_clause = "WHERE trade_date = ?"
            params.append(trade_date)

        query = f"""
            SELECT
                trade_date, actual_trade_count, buy_count, sell_count,
                planned_trade_count, matched_plan_count, off_plan_count,
                follow_plan_count, deviation_count, chase_count,
                over_position_count, bought_watch_only_count,
                execution_score, main_issues, review_summary,
                next_action_suggestion, valid_performance_count,
                avg_return_1d, avg_return_3d, avg_return_5d,
                plan_trade_avg_return_3d, off_plan_avg_return_3d,
                chase_trade_count, chase_avg_return_3d
            FROM daily_review
            {where_clause}
            ORDER BY trade_date
        """

        with self._connect() as con:
            self._create_tables(con)
            return con.execute(query, params).fetchdf()

    def save_period_review(self, df: pd.DataFrame) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(df, PERIOD_REVIEW_COLUMNS).drop_duplicates(
            subset=["start_date", "end_date"],
            keep="last",
        )

        with self._connect() as con:
            self._create_tables(con)
            con.register("incoming_period_review", normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    """
                    DELETE FROM period_review
                    USING incoming_period_review
                    WHERE period_review.start_date IS NOT DISTINCT FROM incoming_period_review.start_date
                      AND period_review.end_date IS NOT DISTINCT FROM incoming_period_review.end_date
                    """
                )
                con.execute(
                    f"""
                    INSERT INTO period_review ({", ".join(PERIOD_REVIEW_COLUMNS)})
                    SELECT {", ".join(PERIOD_REVIEW_COLUMNS)}
                    FROM incoming_period_review
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister("incoming_period_review")

    def load_period_review(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        self._ensure_parent_dir()
        conditions = []
        params = []
        if start_date is not None:
            conditions.append("start_date >= ?")
            params.append(start_date)
        if end_date is not None:
            conditions.append("end_date <= ?")
            params.append(end_date)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT {", ".join(PERIOD_REVIEW_COLUMNS)}
            FROM period_review
            {where_clause}
            ORDER BY start_date, end_date
        """

        with self._connect() as con:
            self._create_tables(con)
            return con.execute(query, params).fetchdf()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self.db_path)

    def _ensure_parent_dir(self) -> None:
        parent = Path(self.db_path).expanduser().parent
        if str(parent) != ".":
            parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _create_tables(con: duckdb.DuckDBPyConnection) -> None:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_basic (
                code VARCHAR PRIMARY KEY,
                name VARCHAR,
                market VARCHAR,
                board VARCHAR,
                industry VARCHAR,
                list_status VARCHAR
            )
            """
        )
        StockAgentStore._ensure_columns(
            con,
            "stock_basic",
            {"industry": "VARCHAR"},
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_bars (
                trade_date VARCHAR,
                code VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                amount DOUBLE,
                PRIMARY KEY (trade_date, code)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_calendar (
                trade_date VARCHAR,
                exchange VARCHAR,
                is_open BIGINT,
                pretrade_date VARCHAR,
                PRIMARY KEY (trade_date, exchange)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_basic (
                trade_date VARCHAR,
                code VARCHAR,
                close DOUBLE,
                turnover_rate DOUBLE,
                turnover_rate_f DOUBLE,
                volume_ratio DOUBLE,
                pe DOUBLE,
                pe_ttm DOUBLE,
                pb DOUBLE,
                ps DOUBLE,
                ps_ttm DOUBLE,
                dv_ratio DOUBLE,
                dv_ttm DOUBLE,
                total_share DOUBLE,
                float_share DOUBLE,
                free_share DOUBLE,
                total_mv DOUBLE,
                circ_mv DOUBLE,
                PRIMARY KEY (trade_date, code)
            )
            """
        )
        StockAgentStore._ensure_columns(
            con,
            "daily_basic",
            {
                "close": "DOUBLE",
                "turnover_rate": "DOUBLE",
                "turnover_rate_f": "DOUBLE",
                "volume_ratio": "DOUBLE",
                "pe": "DOUBLE",
                "pe_ttm": "DOUBLE",
                "pb": "DOUBLE",
                "ps": "DOUBLE",
                "ps_ttm": "DOUBLE",
                "dv_ratio": "DOUBLE",
                "dv_ttm": "DOUBLE",
                "total_share": "DOUBLE",
                "float_share": "DOUBLE",
                "free_share": "DOUBLE",
                "total_mv": "DOUBLE",
                "circ_mv": "DOUBLE",
            },
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_limits (
                trade_date VARCHAR,
                code VARCHAR,
                pre_close DOUBLE,
                up_limit DOUBLE,
                down_limit DOUBLE,
                PRIMARY KEY (trade_date, code)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS suspend_daily (
                trade_date VARCHAR,
                code VARCHAR,
                suspend_type VARCHAR,
                suspend_timing VARCHAR,
                PRIMARY KEY (trade_date, code, suspend_type)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS index_daily (
                trade_date VARCHAR,
                index_code VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                pre_close DOUBLE,
                pct_chg DOUBLE,
                volume DOUBLE,
                amount DOUBLE,
                PRIMARY KEY (trade_date, index_code)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS limit_list_daily (
                trade_date VARCHAR,
                code VARCHAR,
                name VARCHAR,
                close DOUBLE,
                pct_chg DOUBLE,
                amp DOUBLE,
                fc_ratio DOUBLE,
                fl_ratio DOUBLE,
                fd_amount DOUBLE,
                first_time VARCHAR,
                last_time VARCHAR,
                open_times DOUBLE,
                strth DOUBLE,
                limit_type VARCHAR,
                status VARCHAR,
                PRIMARY KEY (trade_date, code, limit_type)
            )
            """
        )
        moneyflow_numeric_columns = {
            column: "DOUBLE" for column in MONEYFLOW_COLUMNS if column not in {"trade_date", "code"}
        }
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS moneyflow (
                trade_date VARCHAR,
                code VARCHAR,
                buy_sm_vol DOUBLE,
                buy_sm_amount DOUBLE,
                sell_sm_vol DOUBLE,
                sell_sm_amount DOUBLE,
                buy_md_vol DOUBLE,
                buy_md_amount DOUBLE,
                sell_md_vol DOUBLE,
                sell_md_amount DOUBLE,
                buy_lg_vol DOUBLE,
                buy_lg_amount DOUBLE,
                sell_lg_vol DOUBLE,
                sell_lg_amount DOUBLE,
                buy_elg_vol DOUBLE,
                buy_elg_amount DOUBLE,
                sell_elg_vol DOUBLE,
                sell_elg_amount DOUBLE,
                net_mf_vol DOUBLE,
                net_mf_amount DOUBLE,
                PRIMARY KEY (trade_date, code)
            )
            """
        )
        StockAgentStore._ensure_columns(con, "moneyflow", moneyflow_numeric_columns)
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS moneyflow_factors (
                trade_date VARCHAR,
                code VARCHAR,
                net_mf_amount DOUBLE,
                net_mf_vol DOUBLE,
                main_net_amount DOUBLE,
                main_net_vol DOUBLE,
                big_net_amount DOUBLE,
                big_net_vol DOUBLE,
                small_net_amount DOUBLE,
                small_net_vol DOUBLE,
                main_net_amount_ratio DOUBLE,
                big_net_amount_ratio DOUBLE,
                small_net_amount_ratio DOUBLE,
                moneyflow_score DOUBLE,
                moneyflow_risk_flags VARCHAR,
                PRIMARY KEY (trade_date, code)
            )
            """
        )
        StockAgentStore._ensure_columns(
            con,
            "moneyflow_factors",
            {column: "DOUBLE" for column in MONEYFLOW_FACTOR_COLUMNS if column not in {"trade_date", "code", "moneyflow_risk_flags"}}
            | {"moneyflow_risk_flags": "VARCHAR"},
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS market_regime (
                trade_date VARCHAR PRIMARY KEY,
                sh_close DOUBLE,
                sh_pct_chg DOUBLE,
                sh_above_ma5 BOOLEAN,
                sh_above_ma10 BOOLEAN,
                sh_above_ma20 BOOLEAN,
                index_trend_score DOUBLE,
                limit_up_count BIGINT,
                limit_down_count BIGINT,
                break_board_count BIGINT,
                limit_up_open_times_avg DOUBLE,
                highest_streak DOUBLE,
                sentiment_score DOUBLE,
                market_regime VARCHAR,
                risk_level VARCHAR,
                regime_reason VARCHAR
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS sw_industry_classification (
                industry_code VARCHAR,
                industry_name VARCHAR,
                level VARCHAR,
                src VARCHAR,
                parent_code VARCHAR,
                index_code VARCHAR,
                is_pub VARCHAR,
                sort_code VARCHAR,
                PRIMARY KEY (industry_code, level, src)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS sw_daily (
                trade_date VARCHAR,
                industry_code VARCHAR,
                industry_name VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                change DOUBLE,
                pct_change DOUBLE,
                volume DOUBLE,
                amount DOUBLE,
                pe DOUBLE,
                pb DOUBLE,
                float_mv DOUBLE,
                total_mv DOUBLE,
                PRIMARY KEY (trade_date, industry_code)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_industry_map (
                code VARCHAR PRIMARY KEY,
                name VARCHAR,
                industry_name VARCHAR,
                industry_code VARCHAR,
                industry_level VARCHAR,
                source VARCHAR
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS industry_strength (
                trade_date VARCHAR,
                industry_code VARCHAR,
                industry_name VARCHAR,
                close DOUBLE,
                pct_change DOUBLE,
                amount DOUBLE,
                industry_return_3d DOUBLE,
                industry_return_5d DOUBLE,
                industry_amount_ratio_5 DOUBLE,
                industry_above_ma5 BOOLEAN,
                industry_above_ma10 BOOLEAN,
                industry_rank_pct_change DOUBLE,
                industry_rank_return_5d DOUBLE,
                industry_rank_amount DOUBLE,
                industry_strength_score DOUBLE,
                industry_strength_level VARCHAR,
                industry_risk_flags VARCHAR,
                PRIMARY KEY (trade_date, industry_code)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS data_quality_report (
                check_name VARCHAR PRIMARY KEY,
                status VARCHAR,
                issue_count BIGINT,
                message VARCHAR
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS provider_compare_result (
                trade_date VARCHAR,
                code VARCHAR,
                field VARCHAR,
                left_value DOUBLE,
                right_value DOUBLE,
                relative_diff DOUBLE,
                status VARCHAR,
                message VARCHAR,
                PRIMARY KEY (trade_date, code, field)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS data_unit_metadata (
                key VARCHAR PRIMARY KEY,
                value VARCHAR,
                updated_at VARCHAR
            )
            """
        )
        StockAgentStore._ensure_columns(
            con,
            "data_unit_metadata",
            {"updated_at": "VARCHAR"},
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_factors (
                trade_date VARCHAR,
                code VARCHAR,
                close DOUBLE,
                pct_chg_1d DOUBLE,
                pct_chg_3d DOUBLE,
                pct_chg_5d DOUBLE,
                pct_chg_10d DOUBLE,
                ma5 DOUBLE,
                ma10 DOUBLE,
                ma20 DOUBLE,
                volume_ma5 DOUBLE,
                amount_ma5 DOUBLE,
                volume_ratio_5 DOUBLE,
                high_20 DOUBLE,
                low_20 DOUBLE,
                close_position_20 DOUBLE,
                above_ma5 BOOLEAN,
                above_ma10 BOOLEAN,
                above_ma20 BOOLEAN,
                turnover_rate DOUBLE,
                turnover_rate_f DOUBLE,
                volume_ratio_daily_basic DOUBLE,
                pe_ttm DOUBLE,
                pb DOUBLE,
                total_mv DOUBLE,
                circ_mv DOUBLE,
                up_limit DOUBLE,
                down_limit DOUBLE,
                is_suspended BOOLEAN,
                is_limit_up_close BOOLEAN,
                is_limit_down_close BOOLEAN,
                limit_up_distance DOUBLE,
                limit_down_distance DOUBLE,
                net_mf_amount DOUBLE,
                main_net_amount DOUBLE,
                main_net_amount_ratio DOUBLE,
                big_net_amount DOUBLE,
                small_net_amount DOUBLE,
                moneyflow_score DOUBLE,
                moneyflow_risk_flags VARCHAR,
                industry_code VARCHAR,
                industry_name VARCHAR,
                industry_strength_score DOUBLE,
                industry_strength_level VARCHAR,
                industry_return_3d DOUBLE,
                industry_return_5d DOUBLE,
                industry_amount_ratio_5 DOUBLE,
                industry_risk_flags VARCHAR,
                PRIMARY KEY (trade_date, code)
            )
            """
        )
        StockAgentStore._ensure_columns(
            con,
            "daily_factors",
            {
                "turnover_rate": "DOUBLE",
                "turnover_rate_f": "DOUBLE",
                "volume_ratio_daily_basic": "DOUBLE",
                "pe_ttm": "DOUBLE",
                "pb": "DOUBLE",
                "total_mv": "DOUBLE",
                "circ_mv": "DOUBLE",
                "up_limit": "DOUBLE",
                "down_limit": "DOUBLE",
                "is_suspended": "BOOLEAN",
                "is_limit_up_close": "BOOLEAN",
                "is_limit_down_close": "BOOLEAN",
                "limit_up_distance": "DOUBLE",
                "limit_down_distance": "DOUBLE",
                "net_mf_amount": "DOUBLE",
                "main_net_amount": "DOUBLE",
                "main_net_amount_ratio": "DOUBLE",
                "big_net_amount": "DOUBLE",
                "small_net_amount": "DOUBLE",
                "moneyflow_score": "DOUBLE",
                "moneyflow_risk_flags": "VARCHAR",
                "industry_code": "VARCHAR",
                "industry_name": "VARCHAR",
                "industry_strength_score": "DOUBLE",
                "industry_strength_level": "VARCHAR",
                "industry_return_3d": "DOUBLE",
                "industry_return_5d": "DOUBLE",
                "industry_amount_ratio_5": "DOUBLE",
                "industry_risk_flags": "VARCHAR",
            },
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS factor_diagnostics (
                factor_name VARCHAR PRIMARY KEY,
                total_count BIGINT,
                non_null_count BIGINT,
                missing_count BIGINT,
                missing_rate DOUBLE,
                mean DOUBLE,
                std DOUBLE,
                min DOUBLE,
                p25 DOUBLE,
                median DOUBLE,
                p75 DOUBLE,
                max DOUBLE,
                candidate_non_null_count BIGINT,
                candidate_mean DOUBLE,
                trade_plan_non_null_count BIGINT,
                trade_plan_mean DOUBLE,
                diagnostic_status VARCHAR,
                diagnostic_message VARCHAR
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS candidate_pool (
                trade_date VARCHAR,
                code VARCHAR,
                name VARCHAR,
                market VARCHAR,
                board VARCHAR,
                close DOUBLE,
                pct_chg_1d DOUBLE,
                pct_chg_3d DOUBLE,
                pct_chg_5d DOUBLE,
                pct_chg_10d DOUBLE,
                volume_ratio_5 DOUBLE,
                close_position_20 DOUBLE,
                above_ma5 BOOLEAN,
                above_ma10 BOOLEAN,
                above_ma20 BOOLEAN,
                amount_ma5 DOUBLE,
                turnover_rate DOUBLE,
                turnover_rate_f DOUBLE,
                volume_ratio_daily_basic DOUBLE,
                total_mv DOUBLE,
                circ_mv DOUBLE,
                up_limit DOUBLE,
                down_limit DOUBLE,
                is_suspended BOOLEAN,
                is_limit_up_close BOOLEAN,
                is_limit_down_close BOOLEAN,
                net_mf_amount DOUBLE,
                main_net_amount DOUBLE,
                main_net_amount_ratio DOUBLE,
                big_net_amount DOUBLE,
                small_net_amount DOUBLE,
                moneyflow_score DOUBLE,
                moneyflow_risk_flags VARCHAR,
                industry_code VARCHAR,
                industry_name VARCHAR,
                industry_strength_score DOUBLE,
                industry_strength_level VARCHAR,
                industry_return_3d DOUBLE,
                industry_return_5d DOUBLE,
                industry_amount_ratio_5 DOUBLE,
                industry_risk_flags VARCHAR,
                score DOUBLE,
                rank BIGINT,
                reason VARCHAR,
                strategy_names VARCHAR,
                strategy_versions VARCHAR,
                signal_count BIGINT,
                active_signal_count BIGINT,
                max_signal_strength DOUBLE,
                total_signal_strength DOUBLE,
                total_weighted_signal_strength DOUBLE,
                avg_strategy_weight DOUBLE,
                recommendations VARCHAR,
                risk_flags VARCHAR,
                PRIMARY KEY (trade_date, code)
            )
            """
        )
        StockAgentStore._ensure_columns(
            con,
            "candidate_pool",
            {
                "market": "VARCHAR",
                "board": "VARCHAR",
                "strategy_names": "VARCHAR",
                "strategy_versions": "VARCHAR",
                "signal_count": "BIGINT",
                "active_signal_count": "BIGINT",
                "max_signal_strength": "DOUBLE",
                "total_signal_strength": "DOUBLE",
                "total_weighted_signal_strength": "DOUBLE",
                "avg_strategy_weight": "DOUBLE",
                "recommendations": "VARCHAR",
                "risk_flags": "VARCHAR",
                "turnover_rate": "DOUBLE",
                "turnover_rate_f": "DOUBLE",
                "volume_ratio_daily_basic": "DOUBLE",
                "total_mv": "DOUBLE",
                "circ_mv": "DOUBLE",
                "up_limit": "DOUBLE",
                "down_limit": "DOUBLE",
                "is_suspended": "BOOLEAN",
                "is_limit_up_close": "BOOLEAN",
                "is_limit_down_close": "BOOLEAN",
                "net_mf_amount": "DOUBLE",
                "main_net_amount": "DOUBLE",
                "main_net_amount_ratio": "DOUBLE",
                "big_net_amount": "DOUBLE",
                "small_net_amount": "DOUBLE",
                "moneyflow_score": "DOUBLE",
                "moneyflow_risk_flags": "VARCHAR",
                "industry_code": "VARCHAR",
                "industry_name": "VARCHAR",
                "industry_strength_score": "DOUBLE",
                "industry_strength_level": "VARCHAR",
                "industry_return_3d": "DOUBLE",
                "industry_return_5d": "DOUBLE",
                "industry_amount_ratio_5": "DOUBLE",
                "industry_risk_flags": "VARCHAR",
            },
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_signals (
                trade_date VARCHAR,
                code VARCHAR,
                strategy_name VARCHAR,
                strategy_version VARCHAR,
                signal_strength DOUBLE,
                entry_reason VARCHAR,
                risk_flags VARCHAR,
                PRIMARY KEY (trade_date, code, strategy_name, strategy_version)
            )
            """
        )
        StockAgentStore._ensure_strategy_signals_schema(con)
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS backtest_results (
                signal_date VARCHAR,
                code VARCHAR,
                strategy_name VARCHAR,
                strategy_version VARCHAR,
                signal_strength DOUBLE,
                entry_date VARCHAR,
                entry_open DOUBLE,
                exit_date_1d VARCHAR,
                exit_close_1d DOUBLE,
                return_1d DOUBLE,
                exit_date_3d VARCHAR,
                exit_close_3d DOUBLE,
                return_3d DOUBLE,
                exit_date_5d VARCHAR,
                exit_close_5d DOUBLE,
                return_5d DOUBLE,
                max_drawdown_1d DOUBLE,
                max_drawdown_3d DOUBLE,
                max_drawdown_5d DOUBLE,
                is_valid BOOLEAN,
                invalid_reason VARCHAR,
                PRIMARY KEY (signal_date, code, strategy_name, strategy_version)
            )
            """
        )
        StockAgentStore._ensure_backtest_results_schema(con)
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_performance (
                strategy_name VARCHAR,
                strategy_version VARCHAR,
                sample_count BIGINT,
                valid_count BIGINT,
                win_rate_1d DOUBLE,
                win_rate_3d DOUBLE,
                win_rate_5d DOUBLE,
                avg_return_1d DOUBLE,
                avg_return_3d DOUBLE,
                avg_return_5d DOUBLE,
                median_return_1d DOUBLE,
                median_return_3d DOUBLE,
                median_return_5d DOUBLE,
                avg_max_drawdown_1d DOUBLE,
                avg_max_drawdown_3d DOUBLE,
                avg_max_drawdown_5d DOUBLE,
                PRIMARY KEY (strategy_name, strategy_version)
            )
            """
        )
        StockAgentStore._ensure_strategy_performance_schema(con)
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_version_performance (
                strategy_name VARCHAR,
                strategy_version VARCHAR,
                sample_count BIGINT,
                valid_count BIGINT,
                win_rate_1d DOUBLE,
                win_rate_3d DOUBLE,
                win_rate_5d DOUBLE,
                avg_return_1d DOUBLE,
                avg_return_3d DOUBLE,
                avg_return_5d DOUBLE,
                median_return_1d DOUBLE,
                median_return_3d DOUBLE,
                median_return_5d DOUBLE,
                avg_max_drawdown_1d DOUBLE,
                avg_max_drawdown_3d DOUBLE,
                avg_max_drawdown_5d DOUBLE,
                PRIMARY KEY (strategy_name, strategy_version)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_version_evaluation (
                strategy_name VARCHAR,
                strategy_version VARCHAR,
                sample_count BIGINT,
                valid_count BIGINT,
                win_rate_3d DOUBLE,
                avg_return_3d DOUBLE,
                median_return_3d DOUBLE,
                avg_max_drawdown_3d DOUBLE,
                evaluation_score DOUBLE,
                evaluation_status VARCHAR,
                risk_level VARCHAR,
                recommendation VARCHAR,
                evaluation_reason VARCHAR,
                PRIMARY KEY (strategy_name, strategy_version)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS parameter_search_backtest_results (
                signal_date VARCHAR,
                code VARCHAR,
                strategy_name VARCHAR,
                strategy_version VARCHAR,
                signal_strength DOUBLE,
                entry_date VARCHAR,
                entry_open DOUBLE,
                exit_date_1d VARCHAR,
                exit_close_1d DOUBLE,
                return_1d DOUBLE,
                exit_date_3d VARCHAR,
                exit_close_3d DOUBLE,
                return_3d DOUBLE,
                exit_date_5d VARCHAR,
                exit_close_5d DOUBLE,
                return_5d DOUBLE,
                max_drawdown_1d DOUBLE,
                max_drawdown_3d DOUBLE,
                max_drawdown_5d DOUBLE,
                is_valid BOOLEAN,
                invalid_reason VARCHAR,
                PRIMARY KEY (signal_date, code, strategy_name, strategy_version)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS parameter_search_performance (
                strategy_name VARCHAR,
                strategy_version VARCHAR,
                sample_count BIGINT,
                valid_count BIGINT,
                win_rate_1d DOUBLE,
                win_rate_3d DOUBLE,
                win_rate_5d DOUBLE,
                avg_return_1d DOUBLE,
                avg_return_3d DOUBLE,
                avg_return_5d DOUBLE,
                median_return_1d DOUBLE,
                median_return_3d DOUBLE,
                median_return_5d DOUBLE,
                avg_max_drawdown_1d DOUBLE,
                avg_max_drawdown_3d DOUBLE,
                avg_max_drawdown_5d DOUBLE,
                PRIMARY KEY (strategy_name, strategy_version)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS parameter_search_results (
                strategy_name VARCHAR,
                strategy_version VARCHAR,
                sample_count BIGINT,
                valid_count BIGINT,
                win_rate_3d DOUBLE,
                avg_return_3d DOUBLE,
                median_return_3d DOUBLE,
                avg_max_drawdown_3d DOUBLE,
                evaluation_score DOUBLE,
                evaluation_status VARCHAR,
                risk_level VARCHAR,
                recommendation VARCHAR,
                evaluation_reason VARCHAR,
                PRIMARY KEY (strategy_name, strategy_version)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS walk_forward_validation (
                strategy_name VARCHAR,
                strategy_version VARCHAR,
                train_valid_count BIGINT,
                train_win_rate_3d DOUBLE,
                train_avg_return_3d DOUBLE,
                train_avg_drawdown_3d DOUBLE,
                validation_valid_count BIGINT,
                validation_win_rate_3d DOUBLE,
                validation_avg_return_3d DOUBLE,
                validation_avg_drawdown_3d DOUBLE,
                return_decay DOUBLE,
                win_rate_decay DOUBLE,
                drawdown_worsening DOUBLE,
                stability_score DOUBLE,
                overfit_risk VARCHAR,
                validation_status VARCHAR,
                validation_reason VARCHAR,
                PRIMARY KEY (strategy_name, strategy_version)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_admission (
                strategy_name VARCHAR,
                strategy_version VARCHAR,
                source VARCHAR,
                valid_count BIGINT,
                evaluation_recommendation VARCHAR,
                evaluation_score DOUBLE,
                oos_status VARCHAR,
                oos_risk VARCHAR,
                oos_stability_score DOUBLE,
                trade_plan_valid_count BIGINT,
                trade_plan_trigger_rate DOUBLE,
                trade_plan_win_rate DOUBLE,
                trade_plan_avg_return DOUBLE,
                trade_plan_avg_drawdown DOUBLE,
                admission_score DOUBLE,
                admission_status VARCHAR,
                admission_recommendation VARCHAR,
                admission_reason VARCHAR,
                PRIMARY KEY (strategy_name, strategy_version)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_plan (
                trade_date VARCHAR,
                code VARCHAR,
                name VARCHAR,
                rank BIGINT,
                close DOUBLE,
                strategy_names VARCHAR,
                strategy_versions VARCHAR,
                active_signal_count BIGINT,
                avg_strategy_weight DOUBLE,
                recommendations VARCHAR,
                risk_flags VARCHAR,
                strategy_type VARCHAR,
                action VARCHAR,
                entry_low DOUBLE,
                entry_high DOUBLE,
                position_low DOUBLE,
                position_high DOUBLE,
                stop_loss DOUBLE,
                take_profit_1 DOUBLE,
                take_profit_2 DOUBLE,
                invalid_condition VARCHAR,
                t_plus_1_risk VARCHAR,
                plan_reason VARCHAR,
                PRIMARY KEY (trade_date, code)
            )
            """
        )
        StockAgentStore._ensure_columns(
            con,
            "trade_plan",
            {
                "strategy_names": "VARCHAR",
                "strategy_versions": "VARCHAR",
                "active_signal_count": "BIGINT",
                "avg_strategy_weight": "DOUBLE",
                "recommendations": "VARCHAR",
                "risk_flags": "VARCHAR",
            },
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS historical_trade_plans (
                trade_date VARCHAR,
                code VARCHAR,
                name VARCHAR,
                rank BIGINT,
                strategy_names VARCHAR,
                strategy_versions VARCHAR,
                active_signal_count BIGINT,
                avg_strategy_weight DOUBLE,
                recommendations VARCHAR,
                risk_flags VARCHAR,
                close DOUBLE,
                strategy_type VARCHAR,
                action VARCHAR,
                entry_low DOUBLE,
                entry_high DOUBLE,
                position_low DOUBLE,
                position_high DOUBLE,
                stop_loss DOUBLE,
                take_profit_1 DOUBLE,
                take_profit_2 DOUBLE,
                invalid_condition VARCHAR,
                t_plus_1_risk VARCHAR,
                plan_reason VARCHAR,
                PRIMARY KEY (trade_date, code, strategy_names, strategy_versions)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_plan_backtest_results (
                plan_date VARCHAR,
                code VARCHAR,
                name VARCHAR,
                action VARCHAR,
                strategy_names VARCHAR,
                strategy_versions VARCHAR,
                recommendations VARCHAR,
                avg_strategy_weight DOUBLE,
                entry_low DOUBLE,
                entry_high DOUBLE,
                stop_loss DOUBLE,
                take_profit_1 DOUBLE,
                take_profit_2 DOUBLE,
                entry_date VARCHAR,
                entry_price DOUBLE,
                exit_date VARCHAR,
                exit_price DOUBLE,
                exit_reason VARCHAR,
                holding_days BIGINT,
                return_pct DOUBLE,
                max_drawdown DOUBLE,
                max_favorable DOUBLE,
                is_triggered BOOLEAN,
                is_valid BOOLEAN,
                invalid_reason VARCHAR,
                backtest_comment VARCHAR,
                PRIMARY KEY (plan_date, code, strategy_names, strategy_versions)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_plan_backtest_performance (
                strategy_names VARCHAR,
                strategy_versions VARCHAR,
                action VARCHAR,
                plan_count BIGINT,
                triggered_count BIGINT,
                valid_count BIGINT,
                trigger_rate DOUBLE,
                win_rate DOUBLE,
                avg_return DOUBLE,
                median_return DOUBLE,
                avg_max_drawdown DOUBLE,
                avg_max_favorable DOUBLE,
                stop_loss_rate DOUBLE,
                take_profit_rate DOUBLE,
                time_exit_rate DOUBLE,
                PRIMARY KEY (strategy_names, strategy_versions, action)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS actual_trades (
                trade_date VARCHAR,
                trade_time VARCHAR,
                code VARCHAR,
                name VARCHAR,
                side VARCHAR,
                price DOUBLE,
                volume DOUBLE,
                amount DOUBLE,
                position_ratio DOUBLE,
                strategy_name VARCHAR,
                plan_rank DOUBLE,
                is_follow_plan VARCHAR,
                reason VARCHAR,
                note VARCHAR
            )
            """
        )
        StockAgentStore._ensure_columns(
            con,
            "actual_trades",
            {
                "trade_date": "VARCHAR",
                "trade_time": "VARCHAR",
                "code": "VARCHAR",
                "name": "VARCHAR",
                "side": "VARCHAR",
                "price": "DOUBLE",
                "volume": "DOUBLE",
                "amount": "DOUBLE",
                "position_ratio": "DOUBLE",
                "strategy_name": "VARCHAR",
                "plan_rank": "DOUBLE",
                "is_follow_plan": "VARCHAR",
                "reason": "VARCHAR",
                "note": "VARCHAR",
            },
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS execution_review (
                trade_date VARCHAR,
                trade_time VARCHAR,
                code VARCHAR,
                name VARCHAR,
                side VARCHAR,
                actual_price DOUBLE,
                actual_volume DOUBLE,
                actual_amount DOUBLE,
                position_ratio DOUBLE,
                plan_rank DOUBLE,
                planned_action VARCHAR,
                entry_low DOUBLE,
                entry_high DOUBLE,
                position_low DOUBLE,
                position_high DOUBLE,
                stop_loss DOUBLE,
                take_profit_1 DOUBLE,
                take_profit_2 DOUBLE,
                plan_match_status VARCHAR,
                execution_status VARCHAR,
                execution_flags VARCHAR,
                execution_comment VARCHAR
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_review (
                trade_date VARCHAR PRIMARY KEY,
                actual_trade_count BIGINT,
                buy_count BIGINT,
                sell_count BIGINT,
                planned_trade_count BIGINT,
                matched_plan_count BIGINT,
                off_plan_count BIGINT,
                follow_plan_count BIGINT,
                deviation_count BIGINT,
                chase_count BIGINT,
                over_position_count BIGINT,
                bought_watch_only_count BIGINT,
                execution_score BIGINT,
                main_issues VARCHAR,
                review_summary VARCHAR,
                next_action_suggestion VARCHAR
            )
            """
        )
        StockAgentStore._ensure_columns(
            con,
            "daily_review",
            {
                "valid_performance_count": "BIGINT",
                "avg_return_1d": "DOUBLE",
                "avg_return_3d": "DOUBLE",
                "avg_return_5d": "DOUBLE",
                "plan_trade_avg_return_3d": "DOUBLE",
                "off_plan_avg_return_3d": "DOUBLE",
                "chase_trade_count": "BIGINT",
                "chase_avg_return_3d": "DOUBLE",
            },
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS actual_trade_performance (
                trade_date VARCHAR,
                trade_time VARCHAR,
                code VARCHAR,
                name VARCHAR,
                side VARCHAR,
                entry_price DOUBLE,
                entry_volume DOUBLE,
                entry_amount DOUBLE,
                position_ratio DOUBLE,
                strategy_name VARCHAR,
                plan_rank DOUBLE,
                plan_match_status VARCHAR,
                execution_status VARCHAR,
                execution_flags VARCHAR,
                return_1d DOUBLE,
                return_3d DOUBLE,
                return_5d DOUBLE,
                max_drawdown_1d DOUBLE,
                max_drawdown_3d DOUBLE,
                max_drawdown_5d DOUBLE,
                max_favorable_1d DOUBLE,
                max_favorable_3d DOUBLE,
                max_favorable_5d DOUBLE,
                is_valid BOOLEAN,
                invalid_reason VARCHAR,
                performance_comment VARCHAR
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                as_of_date VARCHAR,
                code VARCHAR,
                name VARCHAR,
                holding_volume DOUBLE,
                available_volume DOUBLE,
                frozen_volume DOUBLE,
                cost_amount DOUBLE,
                cost_price DOUBLE,
                latest_price DOUBLE,
                market_value DOUBLE,
                floating_pnl DOUBLE,
                floating_pnl_pct DOUBLE,
                position_ratio DOUBLE,
                first_buy_date VARCHAR,
                latest_trade_date VARCHAR,
                strategy_name VARCHAR,
                plan_rank DOUBLE,
                t_plus_1_status VARCHAR,
                position_status VARCHAR,
                PRIMARY KEY (as_of_date, code)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS position_review (
                as_of_date VARCHAR,
                code VARCHAR,
                name VARCHAR,
                holding_volume DOUBLE,
                available_volume DOUBLE,
                frozen_volume DOUBLE,
                cost_amount DOUBLE,
                cost_price DOUBLE,
                latest_price DOUBLE,
                market_value DOUBLE,
                floating_pnl DOUBLE,
                floating_pnl_pct DOUBLE,
                position_ratio DOUBLE,
                first_buy_date VARCHAR,
                latest_trade_date VARCHAR,
                strategy_name VARCHAR,
                plan_rank DOUBLE,
                t_plus_1_status VARCHAR,
                position_status VARCHAR,
                planned_stop_loss DOUBLE,
                planned_take_profit_1 DOUBLE,
                planned_take_profit_2 DOUBLE,
                position_risk_level VARCHAR,
                position_flags VARCHAR,
                position_comment VARCHAR,
                next_action_hint VARCHAR,
                PRIMARY KEY (as_of_date, code)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS period_review (
                start_date VARCHAR,
                end_date VARCHAR,
                trading_days BIGINT,
                actual_trade_count BIGINT,
                buy_count BIGINT,
                sell_count BIGINT,
                follow_plan_count BIGINT,
                off_plan_count BIGINT,
                deviation_count BIGINT,
                chase_count BIGINT,
                over_position_count BIGINT,
                bought_watch_only_count BIGINT,
                avg_execution_score DOUBLE,
                valid_performance_count BIGINT,
                avg_return_1d DOUBLE,
                avg_return_3d DOUBLE,
                avg_return_5d DOUBLE,
                plan_trade_avg_return_3d DOUBLE,
                off_plan_avg_return_3d DOUBLE,
                chase_avg_return_3d DOUBLE,
                over_position_avg_return_3d DOUBLE,
                best_trade_code VARCHAR,
                worst_trade_code VARCHAR,
                main_issues VARCHAR,
                period_summary VARCHAR,
                next_period_suggestion VARCHAR,
                PRIMARY KEY (start_date, end_date)
            )
            """
        )

    @staticmethod
    def _normalize_dataframe(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        normalized = df.copy()
        for column in columns:
            if column not in normalized.columns:
                normalized[column] = None
        return normalized.loc[:, columns]

    def _save_extension_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        columns: list[str],
        key_columns: list[str],
    ) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(df, columns).drop_duplicates(subset=key_columns, keep="last")

        with self._connect() as con:
            self._create_tables(con)
            if normalized.empty:
                return
            incoming_name = f"incoming_{table_name}"
            con.register(incoming_name, normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                join_conditions = " AND ".join(
                    _extension_key_join_condition(table_name, incoming_name, column) for column in key_columns
                )
                con.execute(
                    f"""
                    DELETE FROM {table_name}
                    USING {incoming_name}
                    WHERE {join_conditions}
                    """
                )
                con.execute(
                    f"""
                    INSERT INTO {table_name} ({", ".join(columns)})
                    SELECT {", ".join(columns)}
                    FROM {incoming_name}
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister(incoming_name)

    def _load_extension_table(
        self,
        table_name: str,
        columns: list[str],
        conditions: list[str] | None = None,
        params: list[str] | None = None,
    ) -> pd.DataFrame:
        self._ensure_parent_dir()
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        order_columns = []
        if "trade_date" in columns:
            order_columns.append("trade_date")
        if "code" in columns:
            order_columns.append("code")
        if "industry_code" in columns:
            order_columns.append("industry_code")
        if "level" in columns:
            order_columns.append("level")
        if "src" in columns:
            order_columns.append("src")
        if "index_code" in columns:
            order_columns.append("index_code")
        if "exchange" in columns:
            order_columns.append("exchange")
        if "limit_type" in columns:
            order_columns.append("limit_type")
        if "suspend_type" in columns:
            order_columns.append("suspend_type")
        query = f"""
            SELECT {", ".join(columns)}
            FROM {table_name}
            {where_clause}
            {"ORDER BY " + ", ".join(order_columns) if order_columns else ""}
        """
        with self._connect() as con:
            self._create_tables(con)
            return con.execute(query, params or []).fetchdf()

    @staticmethod
    def _ensure_columns(
        con: duckdb.DuckDBPyConnection,
        table_name: str,
        columns: dict[str, str],
    ) -> None:
        existing = {row[1] for row in con.execute(f"PRAGMA table_info('{table_name}')").fetchall()}
        for column_name, column_type in columns.items():
            if column_name not in existing:
                con.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

    @staticmethod
    def _ensure_strategy_signals_schema(con: duckdb.DuckDBPyConnection) -> None:
        table_info = con.execute("PRAGMA table_info('strategy_signals')").fetchall()
        columns = {row[1] for row in table_info}
        pk_columns = [row[1] for row in sorted(table_info, key=lambda row: row[5]) if row[5] > 0]
        expected_pk = ["trade_date", "code", "strategy_name", "strategy_version"]
        if "strategy_version" in columns and pk_columns == expected_pk:
            return

        con.execute(
            """
            CREATE TABLE strategy_signals_new (
                trade_date VARCHAR,
                code VARCHAR,
                strategy_name VARCHAR,
                strategy_version VARCHAR,
                signal_strength DOUBLE,
                entry_reason VARCHAR,
                risk_flags VARCHAR,
                PRIMARY KEY (trade_date, code, strategy_name, strategy_version)
            )
            """
        )
        version_expr = "COALESCE(strategy_version, 'v1')" if "strategy_version" in columns else "'v1'"
        con.execute(
            f"""
            INSERT INTO strategy_signals_new (
                trade_date, code, strategy_name, strategy_version,
                signal_strength, entry_reason, risk_flags
            )
            SELECT
                trade_date, code, strategy_name, {version_expr} AS strategy_version,
                signal_strength, entry_reason, risk_flags
            FROM strategy_signals
            """
        )
        con.execute("DROP TABLE strategy_signals")
        con.execute("ALTER TABLE strategy_signals_new RENAME TO strategy_signals")

    @staticmethod
    def _ensure_backtest_results_schema(con: duckdb.DuckDBPyConnection) -> None:
        table_info = con.execute("PRAGMA table_info('backtest_results')").fetchall()
        columns = {row[1] for row in table_info}
        pk_columns = [row[1] for row in sorted(table_info, key=lambda row: row[5]) if row[5] > 0]
        expected_pk = ["signal_date", "code", "strategy_name", "strategy_version"]
        if "strategy_version" in columns and pk_columns == expected_pk:
            return

        con.execute(
            """
            CREATE TABLE backtest_results_new (
                signal_date VARCHAR,
                code VARCHAR,
                strategy_name VARCHAR,
                strategy_version VARCHAR,
                signal_strength DOUBLE,
                entry_date VARCHAR,
                entry_open DOUBLE,
                exit_date_1d VARCHAR,
                exit_close_1d DOUBLE,
                return_1d DOUBLE,
                exit_date_3d VARCHAR,
                exit_close_3d DOUBLE,
                return_3d DOUBLE,
                exit_date_5d VARCHAR,
                exit_close_5d DOUBLE,
                return_5d DOUBLE,
                max_drawdown_1d DOUBLE,
                max_drawdown_3d DOUBLE,
                max_drawdown_5d DOUBLE,
                is_valid BOOLEAN,
                invalid_reason VARCHAR,
                PRIMARY KEY (signal_date, code, strategy_name, strategy_version)
            )
            """
        )
        version_expr = "COALESCE(strategy_version, 'v1')" if "strategy_version" in columns else "'v1'"
        con.execute(
            f"""
            INSERT INTO backtest_results_new (
                signal_date, code, strategy_name, strategy_version, signal_strength,
                entry_date, entry_open,
                exit_date_1d, exit_close_1d, return_1d,
                exit_date_3d, exit_close_3d, return_3d,
                exit_date_5d, exit_close_5d, return_5d,
                max_drawdown_1d, max_drawdown_3d, max_drawdown_5d,
                is_valid, invalid_reason
            )
            SELECT
                signal_date, code, strategy_name, {version_expr} AS strategy_version, signal_strength,
                entry_date, entry_open,
                exit_date_1d, exit_close_1d, return_1d,
                exit_date_3d, exit_close_3d, return_3d,
                exit_date_5d, exit_close_5d, return_5d,
                max_drawdown_1d, max_drawdown_3d, max_drawdown_5d,
                is_valid, invalid_reason
            FROM backtest_results
            """
        )
        con.execute("DROP TABLE backtest_results")
        con.execute("ALTER TABLE backtest_results_new RENAME TO backtest_results")

    @staticmethod
    def _ensure_strategy_performance_schema(con: duckdb.DuckDBPyConnection) -> None:
        table_info = con.execute("PRAGMA table_info('strategy_performance')").fetchall()
        columns = {row[1] for row in table_info}
        pk_columns = [row[1] for row in sorted(table_info, key=lambda row: row[5]) if row[5] > 0]
        expected_pk = ["strategy_name", "strategy_version"]
        if "strategy_version" in columns and pk_columns == expected_pk:
            return

        con.execute(
            """
            CREATE TABLE strategy_performance_new (
                strategy_name VARCHAR,
                strategy_version VARCHAR,
                sample_count BIGINT,
                valid_count BIGINT,
                win_rate_1d DOUBLE,
                win_rate_3d DOUBLE,
                win_rate_5d DOUBLE,
                avg_return_1d DOUBLE,
                avg_return_3d DOUBLE,
                avg_return_5d DOUBLE,
                median_return_1d DOUBLE,
                median_return_3d DOUBLE,
                median_return_5d DOUBLE,
                avg_max_drawdown_1d DOUBLE,
                avg_max_drawdown_3d DOUBLE,
                avg_max_drawdown_5d DOUBLE,
                PRIMARY KEY (strategy_name, strategy_version)
            )
            """
        )
        version_expr = "COALESCE(strategy_version, 'v1')" if "strategy_version" in columns else "'v1'"
        con.execute(
            f"""
            INSERT INTO strategy_performance_new (
                strategy_name, strategy_version, sample_count, valid_count,
                win_rate_1d, win_rate_3d, win_rate_5d,
                avg_return_1d, avg_return_3d, avg_return_5d,
                median_return_1d, median_return_3d, median_return_5d,
                avg_max_drawdown_1d, avg_max_drawdown_3d, avg_max_drawdown_5d
            )
            SELECT
                strategy_name, {version_expr} AS strategy_version, sample_count, valid_count,
                win_rate_1d, win_rate_3d, win_rate_5d,
                avg_return_1d, avg_return_3d, avg_return_5d,
                median_return_1d, median_return_3d, median_return_5d,
                avg_max_drawdown_1d, avg_max_drawdown_3d, avg_max_drawdown_5d
            FROM strategy_performance
            """
        )
        con.execute("DROP TABLE strategy_performance")
        con.execute("ALTER TABLE strategy_performance_new RENAME TO strategy_performance")
