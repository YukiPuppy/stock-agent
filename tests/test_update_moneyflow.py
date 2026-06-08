import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.update_moneyflow import update_moneyflow


class MockProvider:
    def __init__(self):
        self.calls = []

    def get_moneyflow(self, trade_date=None, start_date=None, end_date=None, code=None):
        self.calls.append(trade_date)
        return pd.DataFrame(
            {
                "trade_date": [pd.to_datetime(trade_date).strftime("%Y-%m-%d")],
                "code": ["000001"],
                "buy_lg_amount": [100.0],
                "sell_lg_amount": [20.0],
            }
        )


def test_update_moneyflow_uses_mock_provider_and_trade_calendar(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_trade_calendar(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-02", "2025-01-03"],
                "exchange": ["SSE", "SSE"],
                "is_open": [1, 0],
            }
        )
    )
    provider = MockProvider()

    result, _ = update_moneyflow(
        start_date="20250102",
        end_date="20250103",
        db_path=str(db_path),
        sleep_seconds=0,
        provider=provider,
    )

    assert provider.calls == ["20250102"]
    assert len(result) == 1
    assert len(store.load_moneyflow()) == 1
