import pandas as pd

from src.database.duckdb_store import PROVIDER_COMPARE_RESULT_COLUMNS, StockAgentStore
from src.pipeline.export_data_quality_report import export_data_quality_report


def test_export_data_quality_report_writes_file(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_data_quality_report(
        pd.DataFrame([{"check_name": "empty_data", "status": "ok", "issue_count": 0, "message": "ok"}])
    )

    output = export_data_quality_report(
        db_path=store.db_path,
        output_dir=str(tmp_path / "reports"),
        report_date="2026-01-02",
    )

    assert output.endswith("data_quality_2026-01-02.md")
    assert "数据质量与数据源对齐检查报告" in (tmp_path / "reports" / "data_quality_2026-01-02.md").read_text(encoding="utf-8")


def test_export_data_quality_report_omits_cleared_provider_compare_rows(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_data_quality_report(
        pd.DataFrame([{"check_name": "empty_data", "status": "ok", "issue_count": 0, "message": "ok"}])
    )
    store.save_provider_compare_result(
        pd.DataFrame(
            [
                {
                    "trade_date": "20260102",
                    "code": "600000",
                    "field": "volume",
                    "left_value": 100.0,
                    "right_value": 10000.0,
                    "relative_diff": 0.99,
                    "status": "warning",
                    "message": "volume relative_diff old warning",
                },
                {
                    "trade_date": "20260102",
                    "code": "600000",
                    "field": "amount",
                    "left_value": 100.0,
                    "right_value": 100000.0,
                    "relative_diff": 0.999,
                    "status": "warning",
                    "message": "amount relative_diff old warning",
                },
            ]
        )
    )
    store.save_provider_compare_result(pd.DataFrame(columns=PROVIDER_COMPARE_RESULT_COLUMNS))

    output = export_data_quality_report(
        db_path=store.db_path,
        output_dir=str(tmp_path / "reports"),
        report_date="2026-01-02",
    )

    content = (tmp_path / "reports" / "data_quality_2026-01-02.md").read_text(encoding="utf-8")
    assert output.endswith("data_quality_2026-01-02.md")
    assert "当前未执行或暂无数据源对齐异常。" in content
    assert "volume relative_diff old warning" not in content
    assert "amount relative_diff old warning" not in content
    assert "0.99" not in content
    assert "0.999" not in content
