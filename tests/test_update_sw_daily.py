import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline import update_sw_daily as pipeline


def test_update_sw_daily_uses_trade_calendar_and_mock_provider(tmp_path, monkeypatch):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_trade_calendar(
        pd.DataFrame(
            [
                {"trade_date": "2025-01-02", "exchange": "SSE", "is_open": 1},
                {"trade_date": "2025-01-03", "exchange": "SSE", "is_open": 0},
            ]
        )
    )

    class Provider:
        def __init__(self):
            self.trade_dates = []

        def get_sw_daily(self, trade_date=None, **kwargs):
            self.trade_dates.append(trade_date)
            return pd.DataFrame(
                [{"trade_date": "2025-01-02", "industry_code": "801780.SI", "industry_name": "银行", "close": 1000}]
            )

    provider = Provider()
    monkeypatch.setattr(pipeline, "get_data_provider", lambda name: provider)

    result, _ = pipeline.update_sw_daily(
        start_date="20250102",
        end_date="20250103",
        db_path=str(db_path),
        sleep_seconds=0,
    )

    assert provider.trade_dates == ["20250102"]
    assert len(result) == 1
    assert store.load_sw_daily().loc[0, "industry_code"] == "801780.SI"
