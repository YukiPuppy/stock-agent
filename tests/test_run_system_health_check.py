from pathlib import Path

from src.pipeline.run_system_health_check import export_system_health_report, main


def test_run_system_health_check_export_report_writes_file(tmp_path, capsys):
    output_dir = tmp_path / "reports"
    db_path = tmp_path / "stock_agent.duckdb"

    main(
        [
            "--db-path",
            str(db_path),
            "--reports-dir",
            str(output_dir),
            "--configs-dir",
            str(tmp_path / "configs"),
            "--export-report",
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert "System health check finished." in captured.out
    assert "overall_status:" in captured.out
    reports = list(Path(output_dir).glob("system_health_*.md"))
    assert len(reports) == 1


def test_export_system_health_report_uses_requested_date(tmp_path):
    summary = {
        "overall_status": "partial",
        "blocking_issues": [],
        "warnings": [],
        "next_suggestions": [],
    }

    path = export_system_health_report(summary, output_dir=str(tmp_path), report_date="2026-01-02")

    assert path.endswith("system_health_2026-01-02.md")
    assert Path(path).is_file()

