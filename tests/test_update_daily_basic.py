import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline import update_daily_basic as module


def test_update_daily_basic_uses_mock_provider_and_trade_calendar(tmp_path, monkeypatch):
    class FakeProvider:
        def get_daily_basic(self, trade_date):
            return pd.DataFrame(
                {
                    "trade_date": ["2025-01-02"],
                    "code": ["000001"],
                    "turnover_rate": [1.2],
                    "volume_ratio": [1.5],
                    "total_mv": [200.0],
                    "circ_mv": [100.0],
                }
            )

    monkeypatch.setattr(module, "get_data_provider", lambda name: FakeProvider())
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_trade_calendar(
        pd.DataFrame(
            [
                {"trade_date": "2025-01-01", "exchange": "SSE", "is_open": 0},
                {"trade_date": "2025-01-02", "exchange": "SSE", "is_open": 1},
            ]
        )
    )

    result = module.update_daily_basic("20250101", "20250102", db_path=str(db_path), sleep_seconds=0)

    assert result["code"].tolist() == ["000001"]
    daily_basic = store.load_daily_basic("2025-01-02")
    assert daily_basic.loc[0, "turnover_rate"] == 1.2
    assert daily_basic.loc[0, "volume_ratio"] == 1.5
    assert daily_basic.loc[0, "total_mv"] == 200.0
    assert daily_basic.loc[0, "circ_mv"] == 100.0
