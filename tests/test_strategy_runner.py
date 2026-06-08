import pandas as pd

from src.strategy.base_strategy import BaseStrategy, SIGNAL_COLUMNS
from src.strategy.strategy_runner import run_strategies


class DuplicateStrategy(BaseStrategy):
    name = "duplicate"

    def __init__(self, strength: float):
        self.strength = strength

    def generate_signals(self, daily_factors: pd.DataFrame, trade_date: str | None = None) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_date": ["2026-01-02"],
                "code": ["600000"],
                "strategy_name": [self.name],
                "signal_strength": [self.strength],
                "entry_reason": ["reason"],
                "risk_flags": [""],
            }
        )


def test_run_strategies_merges_and_keeps_strongest_duplicate():
    result = run_strategies(
        pd.DataFrame(),
        strategies=[DuplicateStrategy(1.0), DuplicateStrategy(3.0)],
    )

    assert len(result) == 1
    assert result.loc[0, "signal_strength"] == 3.0
    assert result.loc[0, "strategy_version"] == "v1"


def test_run_strategies_returns_empty_standard_columns():
    result = run_strategies(pd.DataFrame(), strategies=[])

    assert result.empty
    assert result.columns.tolist() == SIGNAL_COLUMNS


def test_run_strategies_only_runs_enabled_strategies(tmp_path):
    config_path = tmp_path / "strategies.yaml"
    config_path.write_text(
        """
trend_pullback:
  enabled: true
breakout_volume:
  enabled: false
support_rebound:
  enabled: false
""",
        encoding="utf-8",
    )
    daily_factors = pd.DataFrame(
        {
            "trade_date": ["2026-01-02", "2026-01-02"],
            "code": ["TREND", "BREAK"],
            "pct_chg_1d": [0.055, 0.08],
            "pct_chg_5d": [0.06, 0.08],
            "volume_ratio_5": [1.2, 2.0],
            "close_position_20": [0.92, 0.96],
            "above_ma5": [True, True],
            "above_ma10": [True, False],
            "above_ma20": [True, True],
            "amount_ma5": [1.0, 1.0],
        }
    )

    result = run_strategies(daily_factors, trade_date="2026-01-02", config_path=str(config_path))

    assert result["strategy_name"].tolist() == ["trend_pullback"]
