"""Reset local market-data cache tables and record canonical data units."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
import shutil

from src.config import settings, units
from src.database.duckdb_store import StockAgentStore


MARKET_DATA_CACHE_TABLES = [
    "daily_bars",
    "daily_factors",
    "strategy_signals",
    "candidate_pool",
    "trade_plan",
    "historical_trade_plans",
    "backtest_results",
    "strategy_performance",
    "strategy_version_performance",
    "strategy_version_evaluation",
    "parameter_search_results",
    "parameter_search_performance",
    "parameter_search_backtest_results",
    "walk_forward_validation",
    "trade_plan_backtest_results",
    "trade_plan_backtest_performance",
    "strategy_admission",
    "data_quality_report",
    "provider_compare_result",
    "execution_review",
    "daily_review",
    "actual_trade_performance",
    "positions",
    "position_review",
    "period_review",
]

PRESERVED_TABLES = ["stock_basic", "actual_trades"]

REPORT_PATTERNS = [
    "daily_report_*.md",
    "strategy_evaluation_*.md",
    "parameter_search_*.md",
    "walk_forward_validation_*.md",
    "trade_plan_backtest_*.md",
    "strategy_admission_*.md",
    "data_quality_*.md",
    "daily_review_*.md",
    "trade_performance_*.md",
    "position_review_*.md",
    "period_review_*.md",
    "system_health_*.md",
]


def reset_market_data_cache(
    db_path: str | None = None,
    backup: bool = True,
    preserve_actual_trades: bool = True,
    clear_reports: bool = False,
    reports_dir: str = "reports",
) -> dict:
    """Clear market-data-derived tables and persist Tushare unit metadata."""
    resolved_db_path = db_path if db_path is not None else settings.DB_PATH
    db_file = Path(resolved_db_path).expanduser()
    db_exists = db_file.exists()
    backup_path = _backup_database(db_file) if backup and db_exists else None

    store = StockAgentStore(resolved_db_path)
    tables_to_clear = list(MARKET_DATA_CACHE_TABLES)
    preserved_tables = ["stock_basic"]
    if preserve_actual_trades:
        preserved_tables.append("actual_trades")
    else:
        tables_to_clear.append("actual_trades")

    cleared_tables: list[str] = []
    with store._connect() as con:
        con.execute("BEGIN TRANSACTION")
        try:
            existing_tables = {
                str(row[0])
                for row in con.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'main'
                    """
                ).fetchall()
            }
            for table_name in tables_to_clear:
                if table_name in existing_tables:
                    con.execute(f"DELETE FROM {table_name}")
                    cleared_tables.append(table_name)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

    metadata = {
        "DATA_UNIT_VERSION": units.DATA_UNIT_VERSION,
        "OFFICIAL_DATA_PROVIDER": units.OFFICIAL_DATA_PROVIDER,
        "DAILY_BARS_VOLUME_UNIT": units.DAILY_BARS_VOLUME_UNIT,
        "DAILY_BARS_AMOUNT_UNIT": units.DAILY_BARS_AMOUNT_UNIT,
        "DAILY_FACTORS_AMOUNT_MA5_UNIT": units.DAILY_FACTORS_AMOUNT_MA5_UNIT,
        "ACTUAL_TRADES_AMOUNT_UNIT": units.ACTUAL_TRADES_AMOUNT_UNIT,
        "POSITIONS_AMOUNT_UNIT": units.POSITIONS_AMOUNT_UNIT,
    }
    store.save_data_unit_metadata(metadata)
    cleared_reports_count = _clear_reports(reports_dir) if clear_reports else 0

    return {
        "db_path": resolved_db_path,
        "db_exists_before_reset": db_exists,
        "message": "" if db_exists else f"Database did not exist before reset; initialized metadata at {resolved_db_path}.",
        "backup_path": str(backup_path) if backup_path is not None else "",
        "cleared_tables": cleared_tables,
        "preserved_tables": preserved_tables,
        "cleared_reports_count": cleared_reports_count,
        "metadata": metadata,
        "data_unit_version": metadata["DATA_UNIT_VERSION"],
        "official_data_provider": metadata["OFFICIAL_DATA_PROVIDER"],
    }


def _backup_database(db_file: Path) -> Path:
    backup_dir = db_file.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"stock_agent_before_market_reset_{timestamp}.duckdb"
    shutil.copy2(db_file, backup_path)
    return backup_path


def _clear_reports(reports_dir: str) -> int:
    base = Path(reports_dir)
    if not base.exists():
        return 0

    deleted_paths: set[Path] = set()
    for pattern in REPORT_PATTERNS:
        for path in base.glob(pattern):
            if path.is_file() and path.suffix == ".md" and path not in deleted_paths:
                path.unlink()
                deleted_paths.add(path)
    return len(deleted_paths)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset local market-data cache tables.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--clear-reports", action="store_true", default=False)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--no-backup", action="store_true", default=False)
    parser.add_argument("--clear-actual-trades", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = reset_market_data_cache(
        db_path=args.db_path,
        backup=not args.no_backup,
        preserve_actual_trades=not args.clear_actual_trades,
        clear_reports=args.clear_reports,
        reports_dir=args.reports_dir,
    )
    print("Market data cache reset finished.")
    if summary["message"]:
        print(summary["message"])
    print(f"db_path: {summary['db_path']}")
    print(f"backup_path: {summary['backup_path']}")
    print(f"cleared_tables: {', '.join(summary['cleared_tables'])}")
    print(f"preserved_tables: {', '.join(summary['preserved_tables'])}")
    print(f"cleared_reports_count: {summary['cleared_reports_count']}")
    print(f"data_unit_version: {summary['data_unit_version']}")
    print(f"official_data_provider: {summary['official_data_provider']}")


if __name__ == "__main__":
    main()
