from pathlib import Path

import duckdb
import pandas as pd


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
            ["code", "name", "market", "board", "list_status"],
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
                    INSERT INTO stock_basic (code, name, market, board, list_status)
                    SELECT code, name, market, board, list_status
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
                SELECT code, name, market, board, list_status
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

    def save_daily_factors(self, df: pd.DataFrame) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(
            df,
            [
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
            ],
        ).drop_duplicates(subset=["trade_date", "code"], keep="last")

        with self._connect() as con:
            self._create_tables(con)
            con.register("incoming_daily_factors", normalized)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    """
                    DELETE FROM daily_factors
                    USING incoming_daily_factors
                    WHERE daily_factors.trade_date = incoming_daily_factors.trade_date
                      AND daily_factors.code = incoming_daily_factors.code
                    """
                )
                con.execute(
                    """
                    INSERT INTO daily_factors (
                        trade_date, code, close,
                        pct_chg_1d, pct_chg_3d, pct_chg_5d, pct_chg_10d,
                        ma5, ma10, ma20,
                        volume_ma5, amount_ma5, volume_ratio_5,
                        high_20, low_20, close_position_20,
                        above_ma5, above_ma10, above_ma20
                    )
                    SELECT
                        trade_date, code, close,
                        pct_chg_1d, pct_chg_3d, pct_chg_5d, pct_chg_10d,
                        ma5, ma10, ma20,
                        volume_ma5, amount_ma5, volume_ratio_5,
                        high_20, low_20, close_position_20,
                        above_ma5, above_ma10, above_ma20
                    FROM incoming_daily_factors
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister("incoming_daily_factors")

    def load_daily_factors(self, trade_date: str | None = None) -> pd.DataFrame:
        self._ensure_parent_dir()
        params = []
        where_clause = ""
        if trade_date is not None:
            where_clause = "WHERE trade_date = ?"
            params.append(trade_date)

        query = f"""
            SELECT
                trade_date, code, close,
                pct_chg_1d, pct_chg_3d, pct_chg_5d, pct_chg_10d,
                ma5, ma10, ma20,
                volume_ma5, amount_ma5, volume_ratio_5,
                high_20, low_20, close_position_20,
                above_ma5, above_ma10, above_ma20
            FROM daily_factors
            {where_clause}
            ORDER BY trade_date, code
        """

        with self._connect() as con:
            self._create_tables(con)
            return con.execute(query, params).fetchdf()

    def save_candidate_pool(self, df: pd.DataFrame) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(
            df,
            [
                "trade_date",
                "code",
                "name",
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
                "score",
                "rank",
                "reason",
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
                        trade_date, code, name, close,
                        pct_chg_1d, pct_chg_3d, pct_chg_5d, pct_chg_10d,
                        volume_ratio_5, close_position_20,
                        above_ma5, above_ma10, above_ma20,
                        amount_ma5, score, rank, reason
                    )
                    SELECT
                        trade_date, code, name, close,
                        pct_chg_1d, pct_chg_3d, pct_chg_5d, pct_chg_10d,
                        volume_ratio_5, close_position_20,
                        above_ma5, above_ma10, above_ma20,
                        amount_ma5, score, rank, reason
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
            where_clause = "WHERE trade_date = ?"
            params.append(trade_date)

        query = f"""
            SELECT
                trade_date, code, name, close,
                pct_chg_1d, pct_chg_3d, pct_chg_5d, pct_chg_10d,
                volume_ratio_5, close_position_20,
                above_ma5, above_ma10, above_ma20,
                amount_ma5, score, rank, reason
            FROM candidate_pool
            {where_clause}
            ORDER BY trade_date, rank, code
        """

        with self._connect() as con:
            self._create_tables(con)
            return con.execute(query, params).fetchdf()

    def save_trade_plan(self, df: pd.DataFrame) -> None:
        self._ensure_parent_dir()
        normalized = self._normalize_dataframe(
            df,
            [
                "trade_date",
                "code",
                "name",
                "rank",
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
                        trade_date, code, name, rank, close,
                        strategy_type, action,
                        entry_low, entry_high, position_low, position_high,
                        stop_loss, take_profit_1, take_profit_2,
                        invalid_condition, t_plus_1_risk, plan_reason
                    )
                    SELECT
                        trade_date, code, name, rank, close,
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
                trade_date, code, name, rank, close,
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
                list_status VARCHAR
            )
            """
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
                PRIMARY KEY (trade_date, code)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS candidate_pool (
                trade_date VARCHAR,
                code VARCHAR,
                name VARCHAR,
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
                score DOUBLE,
                rank BIGINT,
                reason VARCHAR,
                PRIMARY KEY (trade_date, code)
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

    @staticmethod
    def _normalize_dataframe(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        normalized = df.copy()
        for column in columns:
            if column not in normalized.columns:
                normalized[column] = None
        return normalized.loc[:, columns]
