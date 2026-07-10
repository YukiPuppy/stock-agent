import pandas as pd
import pytest

from src.strategy.base_strategy import SIGNAL_COLUMNS
from src.strategy.industry_rotation_strategy import IndustryRotationStrategy
from src.strategy.low_vol_trend_strategy import LowVolTrendStrategy
from src.strategy.moneyflow_accumulation_strategy import MoneyflowAccumulationStrategy
from src.strategy.oversold_rebound_strategy import OversoldReboundStrategy
from src.strategy.relative_strength_pullback_strategy import RelativeStrengthPullbackStrategy
from src.strategy.strategy_runner import run_strategies
from src.strategy.volume_dryup_breakout_strategy import VolumeDryupBreakoutStrategy


def _base_row(**overrides):
    row = {
        "trade_date": "2026-01-02",
        "code": "600001",
        "name": "样本股份",
        "market": "SH",
        "board": "主板",
        "close": 10.0,
        "pct_chg_1d": 0.02,
        "pct_chg_3d": 0.04,
        "pct_chg_5d": 0.05,
        "pct_chg_10d": 0.08,
        "above_ma5": True,
        "above_ma10": True,
        "above_ma20": True,
        "close_position_20": 0.74,
        "volume_ratio_5": 1.25,
        "volume_ratio_daily_basic": 1.35,
        "amount_ma5": 20000.0,
        "turnover_rate": 1.2,
        "is_suspended": False,
        "is_limit_up_close": False,
        "is_limit_down_close": False,
        "moneyflow_score": 18.0,
        "main_net_amount": 1000.0,
        "main_net_amount_ratio": 3.0,
        "big_net_amount": 800.0,
        "net_mf_amount": 1200.0,
        "industry_strength_score": 72.0,
        "industry_strength_level": "strong",
        "industry_return_3d": 0.02,
        "industry_return_5d": 0.018,
        "industry_amount_ratio_5": 1.1,
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    ("strategy", "row", "expected_version"),
    [
        (
            IndustryRotationStrategy({"enabled": True, "version": "v1_strength_follow"}),
            _base_row(),
            "v1_strength_follow",
        ),
        (
            MoneyflowAccumulationStrategy({"enabled": True, "version": "v1_main_inflow"}),
            _base_row(),
            "v1_main_inflow",
        ),
        (
            LowVolTrendStrategy({"enabled": True, "version": "v1_ma_alignment"}),
            _base_row(pct_chg_1d=0.01),
            "v1_ma_alignment",
        ),
        (
            OversoldReboundStrategy({"enabled": True, "version": "v1_mild_oversold"}),
            _base_row(
                pct_chg_1d=-0.03,
                pct_chg_3d=-0.04,
                pct_chg_5d=-0.08,
                pct_chg_10d=-0.12,
                close_position_20=0.35,
                moneyflow_score=6.0,
                industry_strength_score=55.0,
            ),
            "v1_mild_oversold",
        ),
        (
            VolumeDryupBreakoutStrategy({"enabled": True, "version": "v1_dryup_recover"}),
            _base_row(),
            "v1_dryup_recover",
        ),
        (
            RelativeStrengthPullbackStrategy({"enabled": True, "version": "v1_rs_pullback"}),
            _base_row(pct_chg_5d=0.03, pct_chg_10d=0.06, industry_return_5d=0.005),
            "v1_rs_pullback",
        ),
    ],
)
def test_new_strategies_generate_standard_signals(strategy, row, expected_version):
    result = strategy.generate_signals(pd.DataFrame([row]), trade_date="2026-01-02")

    assert not result.empty
    assert result.columns.tolist() == SIGNAL_COLUMNS
    assert result.loc[0, "strategy_version"] == expected_version
    assert result.loc[0, "signal_strength"] > 0


def test_new_strategies_apply_tradable_universe_and_risk_filters():
    rows = [
        _base_row(code="600001"),
        _base_row(code="300001"),
        _base_row(code="688001"),
        _base_row(code="920001"),
        _base_row(code="600002", name="*ST样本"),
        _base_row(code="600003", is_suspended=True),
        _base_row(code="600004", amount_ma5=1.0),
        _base_row(code="600005", is_limit_up_close=True),
        _base_row(code="600006", is_limit_down_close=True),
    ]

    result = MoneyflowAccumulationStrategy({"enabled": True, "version": "v1_main_inflow"}).generate_signals(
        pd.DataFrame(rows),
        trade_date="2026-01-02",
    )

    assert result["code"].tolist() == ["600001"]


def test_oversold_rebound_excludes_continuous_risk_downtrend():
    rows = [
        _base_row(
            code="600001",
            pct_chg_1d=-0.03,
            pct_chg_3d=-0.04,
            pct_chg_5d=-0.08,
            pct_chg_10d=-0.12,
            close_position_20=0.35,
            moneyflow_score=8.0,
            industry_strength_score=55.0,
        ),
        _base_row(
            code="600002",
            pct_chg_1d=-0.04,
            pct_chg_3d=-0.11,
            pct_chg_5d=-0.18,
            pct_chg_10d=-0.24,
            close_position_20=0.25,
            moneyflow_score=8.0,
            industry_strength_score=55.0,
        ),
    ]

    result = OversoldReboundStrategy({"enabled": True, "version": "v1_mild_oversold"}).generate_signals(
        pd.DataFrame(rows)
    )

    assert result["code"].tolist() == ["600001"]


def test_market_regime_gating_reduces_aggressive_signal_and_adds_flag():
    row = _base_row(market_regime="weak", risk_level="high")
    weak = VolumeDryupBreakoutStrategy({"enabled": True, "version": "v1_dryup_recover"}).generate_signals(
        pd.DataFrame([row])
    )
    neutral = VolumeDryupBreakoutStrategy({"enabled": True, "version": "v1_dryup_recover"}).generate_signals(
        pd.DataFrame([_base_row()])
    )

    assert weak.loc[0, "signal_strength"] < neutral.loc[0, "signal_strength"]
    assert "weak_market_regime" in weak.loc[0, "risk_flags"]


def test_strategy_runner_can_enable_new_strategy_from_config(tmp_path):
    config_path = tmp_path / "strategies.yaml"
    config_path.write_text(
        """
trend_pullback:
  enabled: false
breakout_volume:
  enabled: false
support_rebound:
  enabled: false
industry_rotation:
  enabled: true
  version: v1_strength_follow
moneyflow_accumulation:
  enabled: false
low_vol_trend:
  enabled: false
oversold_rebound:
  enabled: false
volume_dryup_breakout:
  enabled: false
relative_strength_pullback:
  enabled: false
""",
        encoding="utf-8",
    )

    result = run_strategies(pd.DataFrame([_base_row()]), trade_date="2026-01-02", config_path=str(config_path))

    assert result["strategy_name"].tolist() == ["industry_rotation"]
