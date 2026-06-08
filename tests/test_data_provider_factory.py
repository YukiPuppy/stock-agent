import pytest

from src.data_providers.akshare_provider import AkshareProvider
from src.data_providers.factory import get_data_provider
from src.data_providers.tushare_provider import TushareProvider


def test_get_data_provider_returns_akshare_provider():
    assert isinstance(get_data_provider("akshare"), AkshareProvider)


def test_get_data_provider_defaults_none_to_settings_provider(monkeypatch):
    monkeypatch.setattr("src.data_providers.factory.settings.DEFAULT_DATA_PROVIDER", "tushare")
    monkeypatch.setattr(TushareProvider, "__init__", lambda self: None)

    assert isinstance(get_data_provider(None), TushareProvider)


def test_get_data_provider_is_case_insensitive():
    assert isinstance(get_data_provider("AKSHARE"), AkshareProvider)


def test_get_data_provider_returns_tushare_provider(monkeypatch):
    monkeypatch.setattr(TushareProvider, "__init__", lambda self: None)

    assert isinstance(get_data_provider("tushare"), TushareProvider)


def test_get_data_provider_unknown_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported data provider"):
        get_data_provider("unknown")
