import pytest

from src.data_providers.akshare_provider import AkshareProvider
from src.data_providers.factory import get_data_provider


def test_get_data_provider_returns_akshare_provider():
    assert isinstance(get_data_provider("akshare"), AkshareProvider)


def test_get_data_provider_defaults_none_to_akshare_provider():
    assert isinstance(get_data_provider(None), AkshareProvider)


def test_get_data_provider_is_case_insensitive():
    assert isinstance(get_data_provider("AKSHARE"), AkshareProvider)


def test_get_data_provider_tushare_not_implemented():
    with pytest.raises(NotImplementedError, match="TushareProvider will be added"):
        get_data_provider("tushare")


def test_get_data_provider_unknown_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported data provider"):
        get_data_provider("unknown")
