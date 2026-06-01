from src.config import settings


def test_settings_importable():
    assert settings is not None


def test_db_path_has_default_value():
    assert settings.DB_PATH == "data/stock_agent.duckdb"
