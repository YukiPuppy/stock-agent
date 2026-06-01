import pandas as pd
import pytest

from src.database.duckdb_store import StockAgentStore
from src.pipeline.update_stock_basic import update_stock_basic


def test_update_stock_basic_filters_and_writes_to_duckdb(tmp_path, monkeypatch):
    raw = pd.DataFrame(
        {
            "code": ["600000", "000001", "300750", "688981", "830001", "600001"],
            "name": ["浦发银行", "平安银行", "宁德时代", "中芯国际", "北交样本", "ST样本"],
            "market": ["SH", "SZ", "SZ", "SH", "BJ", "SH"],
            "board": ["主板", "主板", "创业板", "科创板", "北交所", "主板"],
            "list_status": ["L", "L", "L", "L", "L", "L"],
        }
    )

    class FakeProvider:
        def get_stock_basic(self):
            return raw

    provider_names = []

    def fake_get_data_provider(name="akshare"):
        provider_names.append(name)
        return FakeProvider()

    monkeypatch.setattr(
        "src.pipeline.update_stock_basic.get_data_provider",
        fake_get_data_provider,
    )

    db_path = tmp_path / "stock_agent.duckdb"
    result = update_stock_basic(str(db_path))
    saved = StockAgentStore(str(db_path)).load_stock_basic()

    expected_records = [
        {
            "code": "000001",
            "name": "平安银行",
            "market": "SZ",
            "board": "主板",
            "list_status": "L",
        },
        {
            "code": "600000",
            "name": "浦发银行",
            "market": "SH",
            "board": "主板",
            "list_status": "L",
        },
    ]

    assert result["code"].tolist() == ["600000", "000001"]
    assert result["name"].tolist() == ["浦发银行", "平安银行"]
    assert saved.to_dict("records") == expected_records
    assert provider_names == ["akshare"]


def test_update_stock_basic_passes_provider_to_factory(tmp_path, monkeypatch):
    raw = pd.DataFrame(
        {
            "code": ["600000"],
            "name": ["浦发银行"],
            "market": ["SH"],
            "board": ["主板"],
            "list_status": ["L"],
        }
    )

    class FakeProvider:
        def get_stock_basic(self):
            return raw

    provider_names = []

    def fake_get_data_provider(name="akshare"):
        provider_names.append(name)
        return FakeProvider()

    monkeypatch.setattr(
        "src.pipeline.update_stock_basic.get_data_provider",
        fake_get_data_provider,
    )

    update_stock_basic(str(tmp_path / "stock_agent.duckdb"), provider="AKSHARE")

    assert provider_names == ["AKSHARE"]


def test_update_stock_basic_tushare_raises_not_implemented(monkeypatch):
    def fake_get_data_provider(name="akshare"):
        raise NotImplementedError("TushareProvider will be added after AKShare MVP is verified.")

    monkeypatch.setattr(
        "src.pipeline.update_stock_basic.get_data_provider",
        fake_get_data_provider,
    )

    with pytest.raises(NotImplementedError, match="TushareProvider will be added"):
        update_stock_basic(provider="tushare")
