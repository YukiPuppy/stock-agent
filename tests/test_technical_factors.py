import pandas as pd
import pytest

from src.factors.technical_factors import DAILY_FACTOR_COLUMNS, compute_daily_factors


def _bars(code: str, closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    if volumes is None:
        volumes = [100.0 + i * 100.0 for i in range(len(closes))]

    return pd.DataFrame(
        {
            "trade_date": [f"202601{i + 1:02d}" for i in range(len(closes))],
            "code": [code] * len(closes),
            "open": closes,
            "high": [close + 1.0 for close in closes],
            "low": [close - 1.0 for close in closes],
            "close": closes,
            "volume": volumes,
            "amount": [volume * close for volume, close in zip(volumes, closes, strict=True)],
        }
    )


def test_single_stock_computes_pct_chg_ma_and_volume_ratio():
    result = compute_daily_factors(_bars("600000", [10.0, 11.0, 12.0, 13.0, 14.0]))

    assert result.columns.tolist() == DAILY_FACTOR_COLUMNS
    assert pd.isna(result.loc[0, "pct_chg_1d"])
    assert result.loc[1, "pct_chg_1d"] == pytest.approx(0.1)
    assert result.loc[4, "ma5"] == pytest.approx(12.0)
    assert result.loc[4, "volume_ma5"] == pytest.approx(300.0)
    assert result.loc[4, "volume_ratio_5"] == pytest.approx(500.0 / 300.0)


def test_multiple_stocks_do_not_leak_group_state():
    daily_bars = pd.concat(
        [
            _bars("600000", [10.0, 11.0]),
            _bars("000001", [20.0, 22.0]),
        ],
        ignore_index=True,
    )

    result = compute_daily_factors(daily_bars)

    first_rows = result.groupby("code", sort=False).head(1)
    assert first_rows["pct_chg_1d"].isna().all()

    second_rows = result.groupby("code", sort=False).tail(1).sort_values("code").reset_index(drop=True)
    assert second_rows.loc[0, "pct_chg_1d"] == pytest.approx(0.1)
    assert second_rows.loc[1, "pct_chg_1d"] == pytest.approx(0.1)


def test_empty_daily_bars_returns_standard_empty_dataframe():
    result = compute_daily_factors(pd.DataFrame())

    assert result.empty
    assert result.columns.tolist() == DAILY_FACTOR_COLUMNS
