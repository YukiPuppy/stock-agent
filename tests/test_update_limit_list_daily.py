import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline import update_limit_list_daily as module


def test_update_limit_list_daily_uses_mock_provider_and_trade_calendar(tmp_path, monkeypatch):
    calls = []

    class FakeProvider:
        def get_limit_list_daily(self, trade_date):
            calls.append(trade_date)
            return pd.DataFrame(
                {
                    "trade_date": ["2025-01-02"],
                    "code": ["000001"],
                    "name": ["平安银行"],
                    "close": [11.0],
                    "pct_chg": [10.0],
                    "limit_type": ["U"],
                    "status": [""],
                    "open_times": [0],
                }
            )

    monkeypatch.setattr(module, "get_data_provider", lambda name: FakeProvider())
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_trade_calendar(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-02", "2025-01-03"],
                "exchange": ["SSE", "SSE"],
                "is_open": [1, 0],
                "pretrade_date": ["", ""],
            }
        )
    )

    result = module.update_limit_list_daily("20250102", "20250103", db_path=str(db_path), sleep_seconds=0)

    assert calls == ["20250102"]
    assert len(result) == 1
    assert store.load_limit_list_daily("2025-01-02").loc[0, "limit_type"] == "U"
