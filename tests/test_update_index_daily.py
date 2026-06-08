import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline import update_index_daily as module


def test_update_index_daily_uses_mock_provider(tmp_path, monkeypatch):
    class FakeProvider:
        def get_index_daily(self, index_code, start_date, end_date):
            return pd.DataFrame(
                {
                    "trade_date": ["2025-01-02"],
                    "index_code": [index_code],
                    "open": [3100.0],
                    "high": [3120.0],
                    "low": [3090.0],
                    "close": [3110.0],
                    "pre_close": [3100.0],
                    "pct_chg": [0.32],
                    "volume": [100.0],
                    "amount": [200.0],
                }
            )

    monkeypatch.setattr(module, "get_data_provider", lambda name: FakeProvider())
    db_path = tmp_path / "stock_agent.duckdb"

    result = module.update_index_daily(
        "20250102",
        "20250102",
        index_codes=["000001.SH", "399001.SZ"],
        db_path=str(db_path),
        sleep_seconds=0,
    )

    saved = StockAgentStore(str(db_path)).load_index_daily()
    assert len(result) == 2
    assert saved["index_code"].tolist() == ["000001.SH", "399001.SZ"]
