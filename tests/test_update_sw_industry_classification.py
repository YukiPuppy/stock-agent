import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline import update_sw_industry_classification as pipeline


def test_update_sw_industry_classification_uses_mock_provider_and_saves(tmp_path, monkeypatch):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_stock_basic(pd.DataFrame({"code": ["000001"], "name": ["A"], "market": ["主板"], "board": ["银行"]}))

    class Provider:
        def get_sw_industry_classification(self, level="L1", src="SW2021"):
            return pd.DataFrame(
                [{"industry_code": "801780.SI", "industry_name": "银行", "level": level, "src": src}]
            )

    monkeypatch.setattr(pipeline, "get_data_provider", lambda name: Provider())

    classification, stock_map, _ = pipeline.update_sw_industry_classification(db_path=str(db_path))

    assert len(classification) == 1
    assert len(stock_map) == 1
    assert store.load_sw_industry_classification().loc[0, "industry_code"] == "801780.SI"
    assert store.load_stock_industry_map().loc[0, "industry_code"] == "801780.SI"
