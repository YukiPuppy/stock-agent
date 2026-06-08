import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline import update_suspend_daily as module


def test_update_suspend_daily_uses_mock_provider(tmp_path, monkeypatch):
    class FakeProvider:
        def get_suspend_daily(self, trade_date):
            return pd.DataFrame(
                {"trade_date": ["2025-01-02"], "code": ["000001"], "suspend_type": ["S"], "suspend_timing": ["09:30"]}
            )

    monkeypatch.setattr(module, "get_data_provider", lambda name: FakeProvider())
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_trade_calendar(pd.DataFrame({"trade_date": ["2025-01-02"], "exchange": ["SSE"], "is_open": [1]}))

    result = module.update_suspend_daily("20250102", "20250102", db_path=str(db_path), sleep_seconds=0)

    assert len(result) == 1
    assert store.load_suspend_daily("2025-01-02").loc[0, "suspend_type"] == "S"
