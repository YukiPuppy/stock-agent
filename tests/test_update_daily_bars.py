import pandas as pd
from src.database.duckdb_store import StockAgentStore
from src.pipeline.update_daily_bars import update_daily_bars


DAILY_BAR_COLUMNS = ["trade_date", "code", "open", "high", "low", "close", "volume", "amount"]


def _stock_basic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["600000", "000001", "600519"],
            "name": ["浦发银行", "平安银行", "贵州茅台"],
            "market": ["SH", "SZ", "SH"],
            "board": ["主板", "主板", "主板"],
            "list_status": ["L", "L", "L"],
        }
    )


def _daily_bars(code: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2025-01-02"],
            "code": [code],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [1000.0],
            "amount": [10000.0],
        }
    )


def test_update_daily_bars_fetches_codes_and_saves(monkeypatch):
    requested_codes = []
    saved = []
    provider_names = []

    def fake_load_stock_basic(self):
        return _stock_basic()

    class FakeProvider:
        def get_daily_bars(self, code, start_date, end_date):
            requested_codes.append((code, start_date, end_date))
            return _daily_bars(code)

    def fake_get_data_provider(name="akshare"):
        provider_names.append(name)
        return FakeProvider()

    def fake_save_daily_bars(self, df):
        saved.append(df.copy())

    monkeypatch.setattr(StockAgentStore, "load_stock_basic", fake_load_stock_basic)
    monkeypatch.setattr("src.pipeline.update_daily_bars.get_data_provider", fake_get_data_provider)
    monkeypatch.setattr(StockAgentStore, "save_daily_bars", fake_save_daily_bars)

    monkeypatch.setattr("src.pipeline.update_daily_bars.settings.DEFAULT_DATA_PROVIDER", "tushare")

    result = update_daily_bars("20250101", "20250110", db_path="test.duckdb")

    assert requested_codes == [
        ("600000", "20250101", "20250110"),
        ("000001", "20250101", "20250110"),
        ("600519", "20250101", "20250110"),
    ]
    assert result.columns.tolist() == DAILY_BAR_COLUMNS
    assert result["code"].tolist() == ["600000", "000001", "600519"]
    assert provider_names == ["tushare"]
    assert len(saved) == 1
    pd.testing.assert_frame_equal(saved[0], result)


def test_update_daily_bars_respects_limit(monkeypatch):
    requested_codes = []
    saved = []

    def fake_load_stock_basic(self):
        return _stock_basic()

    class FakeProvider:
        def get_daily_bars(self, code, start_date, end_date):
            requested_codes.append(code)
            return _daily_bars(code)

    def fake_get_data_provider(name="akshare"):
        return FakeProvider()

    def fake_save_daily_bars(self, df):
        saved.append(df.copy())

    monkeypatch.setattr(StockAgentStore, "load_stock_basic", fake_load_stock_basic)
    monkeypatch.setattr("src.pipeline.update_daily_bars.get_data_provider", fake_get_data_provider)
    monkeypatch.setattr(StockAgentStore, "save_daily_bars", fake_save_daily_bars)

    result = update_daily_bars("20250101", "20250110", db_path="test.duckdb", limit=2)

    assert requested_codes == ["600000", "000001"]
    assert result["code"].tolist() == ["600000", "000001"]
    assert len(saved) == 1


def test_update_daily_bars_continues_when_one_stock_fails(monkeypatch, capsys):
    requested_codes = []
    saved = []

    def fake_load_stock_basic(self):
        return _stock_basic()

    class FakeProvider:
        def get_daily_bars(self, code, start_date, end_date):
            requested_codes.append(code)
            if code == "000001":
                raise RuntimeError("provider unavailable")
            return _daily_bars(code)

    def fake_get_data_provider(name="akshare"):
        return FakeProvider()

    def fake_save_daily_bars(self, df):
        saved.append(df.copy())

    monkeypatch.setattr(StockAgentStore, "load_stock_basic", fake_load_stock_basic)
    monkeypatch.setattr("src.pipeline.update_daily_bars.get_data_provider", fake_get_data_provider)
    monkeypatch.setattr(StockAgentStore, "save_daily_bars", fake_save_daily_bars)

    result = update_daily_bars("20250101", "20250110", db_path="test.duckdb")
    output = capsys.readouterr().out

    assert requested_codes == ["600000", "000001", "600519"]
    assert result.columns.tolist() == DAILY_BAR_COLUMNS
    assert result["code"].tolist() == ["600000", "600519"]
    assert "[1/3] fetching 600000" in output
    assert "[1/3] success rows=1" in output
    assert "[2/3] failed code=000001 error=provider unavailable" in output
    assert "成功数量: 2" in output
    assert "失败数量: 1" in output
    assert len(saved) == 1
    pd.testing.assert_frame_equal(saved[0], result)


def test_update_daily_bars_all_failures_do_not_save(monkeypatch, capsys):
    requested_codes = []
    saved = []

    def fake_load_stock_basic(self):
        return _stock_basic()

    class FakeProvider:
        def get_daily_bars(self, code, start_date, end_date):
            requested_codes.append(code)
            raise RuntimeError("provider unavailable")

    def fake_get_data_provider(name="akshare"):
        return FakeProvider()

    def fake_save_daily_bars(self, df):
        saved.append(df.copy())

    monkeypatch.setattr(StockAgentStore, "load_stock_basic", fake_load_stock_basic)
    monkeypatch.setattr("src.pipeline.update_daily_bars.get_data_provider", fake_get_data_provider)
    monkeypatch.setattr(StockAgentStore, "save_daily_bars", fake_save_daily_bars)

    result = update_daily_bars("20250101", "20250110", db_path="test.duckdb")
    output = capsys.readouterr().out

    assert requested_codes == ["600000", "000001", "600519"]
    assert result.empty
    assert result.columns.tolist() == DAILY_BAR_COLUMNS
    assert saved == []
    assert "no daily bars fetched, nothing saved" in output
    assert "成功数量: 0" in output
    assert "失败数量: 3" in output


def test_update_daily_bars_sleep_seconds_between_stocks(monkeypatch):
    sleeps = []

    def fake_load_stock_basic(self):
        return _stock_basic()

    class FakeProvider:
        def get_daily_bars(self, code, start_date, end_date):
            return _daily_bars(code)

    def fake_get_data_provider(name="akshare"):
        return FakeProvider()

    def fake_save_daily_bars(self, df):
        pass

    monkeypatch.setattr(StockAgentStore, "load_stock_basic", fake_load_stock_basic)
    monkeypatch.setattr("src.pipeline.update_daily_bars.get_data_provider", fake_get_data_provider)
    monkeypatch.setattr(StockAgentStore, "save_daily_bars", fake_save_daily_bars)
    monkeypatch.setattr("src.pipeline.update_daily_bars.time.sleep", lambda seconds: sleeps.append(seconds))

    update_daily_bars(
        "20250101",
        "20250110",
        db_path="test.duckdb",
        sleep_seconds=0.25,
    )

    assert sleeps == [0.25, 0.25]


def test_update_daily_bars_passes_provider_to_factory(monkeypatch):
    provider_names = []

    def fake_load_stock_basic(self):
        return _stock_basic().head(1)

    class FakeProvider:
        def get_daily_bars(self, code, start_date, end_date):
            return _daily_bars(code)

    def fake_get_data_provider(name="akshare"):
        provider_names.append(name)
        return FakeProvider()

    monkeypatch.setattr(StockAgentStore, "load_stock_basic", fake_load_stock_basic)
    monkeypatch.setattr("src.pipeline.update_daily_bars.get_data_provider", fake_get_data_provider)
    monkeypatch.setattr(StockAgentStore, "save_daily_bars", lambda self, df: None)

    update_daily_bars("20250101", "20250110", db_path="test.duckdb", provider="AKSHARE")

    assert provider_names == ["akshare"]


def test_update_daily_bars_default_provider_uses_settings(monkeypatch):
    provider_names = []

    def fake_load_stock_basic(self):
        return _stock_basic().head(1)

    class FakeProvider:
        def get_daily_bars(self, code, start_date, end_date):
            return _daily_bars(code)

    def fake_get_data_provider(name=None):
        provider_names.append(name)
        return FakeProvider()

    monkeypatch.setattr("src.pipeline.update_daily_bars.settings.DEFAULT_DATA_PROVIDER", "tushare")
    monkeypatch.setattr(StockAgentStore, "load_stock_basic", fake_load_stock_basic)
    monkeypatch.setattr("src.pipeline.update_daily_bars.get_data_provider", fake_get_data_provider)
    monkeypatch.setattr(StockAgentStore, "save_daily_bars", lambda self, df: None)

    update_daily_bars("20250101", "20250110", db_path="test.duckdb")

    assert provider_names == ["tushare"]


def test_update_daily_bars_tushare_uses_mock_provider(monkeypatch):
    provider_names = []
    requested_codes = []

    def fake_load_stock_basic(self):
        return _stock_basic().head(1)

    class FakeProvider:
        def get_daily_bars(self, code, start_date, end_date):
            requested_codes.append(code)
            return _daily_bars(code)

    def fake_get_data_provider(name="akshare"):
        provider_names.append(name)
        return FakeProvider()

    monkeypatch.setattr(StockAgentStore, "load_stock_basic", fake_load_stock_basic)
    monkeypatch.setattr("src.pipeline.update_daily_bars.get_data_provider", fake_get_data_provider)
    monkeypatch.setattr(StockAgentStore, "save_daily_bars", lambda self, df: None)

    result = update_daily_bars("20250101", "20250110", db_path="test.duckdb", provider="tushare")

    assert provider_names == ["tushare"]
    assert requested_codes == ["600000"]
    assert result["code"].tolist() == ["600000"]
