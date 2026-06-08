import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline import update_trade_calendar as module


def test_update_trade_calendar_uses_mock_provider(tmp_path, monkeypatch):
    class FakeProvider:
        def get_trade_calendar(self, start_date, end_date, exchange="SSE"):
            return pd.DataFrame(
                {"trade_date": ["2025-01-02"], "exchange": [exchange], "is_open": [1], "pretrade_date": ["2024-12-31"]}
            )

    monkeypatch.setattr(module, "get_data_provider", lambda name: FakeProvider())
    db_path = tmp_path / "stock_agent.duckdb"

    result = module.update_trade_calendar("20250102", "20250102", db_path=str(db_path))

    saved = StockAgentStore(str(db_path)).load_trade_calendar()
    assert len(result) == 1
    assert saved.loc[0, "trade_date"] == "2025-01-02"
