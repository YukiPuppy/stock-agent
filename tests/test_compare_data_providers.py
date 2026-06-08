import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.compare_data_providers import run_provider_compare


class FakeProvider:
    def __init__(self, close):
        self.close = close

    def get_daily_bars(self, code, start_date, end_date):
        return pd.DataFrame(
            {
                "trade_date": [start_date],
                "code": [code],
                "open": [10.0],
                "high": [10.5],
                "low": [9.8],
                "close": [self.close],
                "volume": [1000],
                "amount": [10000],
            }
        )


def test_run_provider_compare_uses_mock_providers(monkeypatch, tmp_path):
    def fake_get_data_provider(name):
        return FakeProvider(10.0 if name == "akshare" else 10.5)

    monkeypatch.setattr("src.pipeline.compare_data_providers.get_data_provider", fake_get_data_provider)
    db_path = str(tmp_path / "stock_agent.duckdb")

    compare_result, summary = run_provider_compare("600000", "20260102", "20260103", db_path=db_path)

    assert not compare_result.empty
    assert not summary.empty
    assert not StockAgentStore(db_path).load_provider_compare_result().empty

