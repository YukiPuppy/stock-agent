import duckdb
import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.reset_market_data_cache import main, reset_market_data_cache


def _seed_reset_db(db_path):
    store = StockAgentStore(str(db_path))
    store.init_tables()
    store.save_daily_bars(
        pd.DataFrame(
            {
                "trade_date": ["2026-01-02"],
                "code": ["600000"],
                "open": [10.0],
                "high": [10.5],
                "low": [9.8],
                "close": [10.2],
                "volume": [1000.0],
                "amount": [10200.0],
            }
        )
    )
    store.save_actual_trades(
        pd.DataFrame(
            {
                "trade_date": ["2026-01-02"],
                "trade_time": ["10:00:00"],
                "code": ["600000"],
                "name": ["浦发银行"],
                "side": ["buy"],
                "price": [10.0],
                "volume": [100.0],
                "amount": [1000.0],
            }
        )
    )
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            INSERT INTO daily_factors (trade_date, code, close)
            VALUES ('2026-01-02', '600000', 10.2)
            """
        )
        con.execute(
            """
            INSERT INTO strategy_signals (
                trade_date, code, strategy_name, strategy_version, signal_strength
            )
            VALUES ('2026-01-02', '600000', 'demo', 'v1', 1.0)
            """
        )
        con.execute(
            """
            INSERT INTO candidate_pool (trade_date, code, name)
            VALUES ('2026-01-02', '600000', '浦发银行')
            """
        )
        con.execute(
            """
            INSERT INTO trade_plan (trade_date, code, name)
            VALUES ('2026-01-02', '600000', '浦发银行')
            """
        )
    return store


def _row_count(db_path, table_name):
    with duckdb.connect(str(db_path)) as con:
        return con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]


def test_reset_market_data_cache_defaults_backup_and_preserve_actual_trades(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    _seed_reset_db(db_path)

    summary = reset_market_data_cache(db_path=str(db_path))

    assert summary["backup_path"]
    assert (tmp_path / "backups").is_dir()
    assert _row_count(db_path, "daily_bars") == 0
    assert _row_count(db_path, "daily_factors") == 0
    assert _row_count(db_path, "strategy_signals") == 0
    assert _row_count(db_path, "candidate_pool") == 0
    assert _row_count(db_path, "trade_plan") == 0
    assert _row_count(db_path, "actual_trades") == 1
    assert "actual_trades" in summary["preserved_tables"]


def test_reset_market_data_cache_clear_actual_trades_when_requested(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    _seed_reset_db(db_path)

    summary = reset_market_data_cache(
        db_path=str(db_path),
        backup=False,
        preserve_actual_trades=False,
    )

    assert _row_count(db_path, "actual_trades") == 0
    assert "actual_trades" in summary["cleared_tables"]
    assert "actual_trades" not in summary["preserved_tables"]


def test_reset_market_data_cache_missing_tables_do_not_crash(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    with duckdb.connect(str(db_path)) as con:
        con.execute("CREATE TABLE daily_bars (trade_date VARCHAR, code VARCHAR)")
        con.execute("INSERT INTO daily_bars VALUES ('2026-01-02', '600000')")

    summary = reset_market_data_cache(db_path=str(db_path), backup=False)

    assert _row_count(db_path, "daily_bars") == 0
    assert "daily_bars" in summary["cleared_tables"]
    assert summary["metadata"]["OFFICIAL_DATA_PROVIDER"] == "tushare"


def test_reset_market_data_cache_clear_reports_only_target_markdown(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    _seed_reset_db(db_path)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    target = reports_dir / "daily_report_2026-01-02.md"
    target.write_text("old", encoding="utf-8")
    non_target_md = reports_dir / "notes.md"
    non_target_md.write_text("keep", encoding="utf-8")
    non_md = reports_dir / "daily_report_2026-01-02.txt"
    non_md.write_text("keep", encoding="utf-8")
    system_health = reports_dir / "system_health_2026-01-02.md"
    system_health.write_text("old", encoding="utf-8")

    summary = reset_market_data_cache(
        db_path=str(db_path),
        backup=False,
        clear_reports=True,
        reports_dir=str(reports_dir),
    )

    assert summary["cleared_reports_count"] == 2
    assert not target.exists()
    assert not system_health.exists()
    assert non_target_md.exists()
    assert non_md.exists()
    assert reports_dir.exists()


def test_reset_market_data_cache_writes_data_unit_metadata(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    _seed_reset_db(db_path)

    reset_market_data_cache(db_path=str(db_path), backup=False)

    metadata = StockAgentStore(str(db_path)).load_data_unit_metadata().set_index("key")["value"].to_dict()
    assert metadata["DATA_UNIT_VERSION"] == "daily_bars_v2_tushare_units"
    assert metadata["OFFICIAL_DATA_PROVIDER"] == "tushare"
    assert metadata["DAILY_BARS_VOLUME_UNIT"] == "手"
    assert metadata["DAILY_BARS_AMOUNT_UNIT"] == "千元"
    assert metadata["DAILY_FACTORS_AMOUNT_MA5_UNIT"] == "千元"
    assert metadata["ACTUAL_TRADES_AMOUNT_UNIT"] == "元"
    assert metadata["POSITIONS_AMOUNT_UNIT"] == "元"


def test_reset_market_data_cache_cli_supports_clear_reports(tmp_path, capsys):
    db_path = tmp_path / "stock_agent.duckdb"
    _seed_reset_db(db_path)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "daily_report_2026-01-02.md").write_text("old", encoding="utf-8")

    main(
        [
            "--db-path",
            str(db_path),
            "--no-backup",
            "--clear-reports",
            "--reports-dir",
            str(reports_dir),
        ]
    )

    output = capsys.readouterr().out
    assert "Market data cache reset finished." in output
    assert "cleared_reports_count: 1" in output
    assert "official_data_provider: tushare" in output
