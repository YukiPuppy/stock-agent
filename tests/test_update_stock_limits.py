import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline import update_stock_limits as module


def test_update_stock_limits_uses_mock_provider(tmp_path, monkeypatch):
    class FakeProvider:
        def get_stock_limits(self, trade_date):
            return pd.DataFrame(
                {"trade_date": ["2025-01-02"], "code": ["000001"], "pre_close": [10.0], "up_limit": [11.0], "down_limit": [9.0]}
            )

    monkeypatch.setattr(module, "get_data_provider", lambda name: FakeProvider())
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_trade_calendar(pd.DataFrame({"trade_date": ["2025-01-02"], "exchange": ["SSE"], "is_open": [1]}))

    result = module.update_stock_limits("20250102", "20250102", db_path=str(db_path), sleep_seconds=0)

    assert len(result) == 1
    assert store.load_stock_limits("2025-01-02").loc[0, "up_limit"] == 11.0
