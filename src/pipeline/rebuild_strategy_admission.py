"""Rebuild strategy admission from persisted research outputs only."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from datetime import date

import pandas as pd

from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.pipeline.export_strategy_admission_report import export_strategy_admission_report
from src.pipeline.memory import log_memory
from src.research.strategy_admission import build_strategy_admission


LOGGER = logging.getLogger(__name__)


def rebuild_strategy_admission(
    *,
    run_id: str,
    db_path: str | None = None,
    output_dir: str = "reports",
    dry_run: bool = False,
    replace_current_run: bool = False,
) -> dict:
    resolved_db_path = db_path or DB_PATH
    store = StockAgentStore(resolved_db_path)
    log_memory("rebuild_strategy_admission", "before_load")
    evaluation = store.load_strategy_version_evaluation(run_id=run_id)
    parameter_results = store.load_parameter_search_results(run_id=run_id)
    walk_forward = store.load_walk_forward_validation(run_id=run_id)
    trade_plan_performance = store.load_trade_plan_backtest_performance(run_id=run_id)
    log_memory("rebuild_strategy_admission", "inputs_loaded")

    admission = build_strategy_admission(
        strategy_evaluation=evaluation,
        parameter_search_results=parameter_results,
        walk_forward_validation=walk_forward,
        trade_plan_backtest_performance=trade_plan_performance,
    ).assign(run_id=run_id)
    nonnull_trade_plan = int(admission.get("trade_plan_win_rate", pd.Series(dtype=float)).notna().sum())
    if not trade_plan_performance.empty and nonnull_trade_plan == 0:
        LOGGER.warning(
            "run_id=%s has %s trade-plan performance rows but none map to admission candidates; "
            "check buy-like action and composite strategy dimensions",
            run_id,
            len(trade_plan_performance),
        )
    elif trade_plan_performance.empty:
        LOGGER.warning(
            "run_id=%s has no trade_plan_backtest_performance rows; admission trade_plan_* metrics remain empty",
            run_id,
        )

    existing = store.load_strategy_admission(run_id=run_id)
    write_status = "dry_run"
    if not dry_run:
        if not existing.empty and not replace_current_run:
            write_status = "preserved_existing_run_use_replace_current_run_to_overwrite"
            LOGGER.warning(
                "run_id=%s already has %s admission rows; preserving them because --replace-current-run was not set",
                run_id,
                len(existing),
            )
        else:
            if replace_current_run:
                store.delete_run_rows(run_id, ["strategy_admission"])
            store.save_strategy_admission(admission)
            write_status = "written"
    report_path = export_strategy_admission_report(
        db_path=resolved_db_path,
        output_dir=output_dir,
        report_date=date.today().isoformat(),
        admission=admission,
        run_id=run_id,
    )
    status_counts = (
        admission["admission_status"].fillna("missing").value_counts(dropna=False).to_dict()
        if "admission_status" in admission.columns
        else {}
    )
    summary = {
        "run_id": run_id,
        "db_path": resolved_db_path,
        "dry_run": dry_run,
        "replace_current_run": replace_current_run,
        "write_status": write_status,
        "strategy_version_evaluation_rows": len(evaluation),
        "parameter_search_results_rows": len(parameter_results),
        "walk_forward_validation_rows": len(walk_forward),
        "trade_plan_backtest_performance_rows": len(trade_plan_performance),
        "strategy_admission_rows": len(admission),
        "trade_plan_win_rate_nonnull_rows": nonnull_trade_plan,
        "admission_status_distribution": status_counts,
        "risk_rejected_count": int(status_counts.get("risk_rejected", 0)),
        "oos_failed_count": int(status_counts.get("oos_failed", 0)),
        "strategy_admission_report_path": report_path,
    }
    log_memory("rebuild_strategy_admission", "finished")
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild strategy_admission for one run without rerunning search, OOS, or trade-plan backtests."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replace-current-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = _parse_args(argv)
    summary = rebuild_strategy_admission(
        run_id=args.run_id,
        db_path=args.db_path,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        replace_current_run=args.replace_current_run,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
