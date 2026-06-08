import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.build_candidate_pool import _build_and_save_candidate_pool, _parse_args, build_candidate_pool
from src.strategy.candidate_selector import CANDIDATE_COLUMNS


def _factor_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2026-01-02", "2026-01-02"],
            "code": ["600000", "000001"],
            "close": [11.0, 20.0],
            "pct_chg_1d": [0.02, 0.03],
            "pct_chg_3d": [0.03, 0.01],
            "pct_chg_5d": [0.05, 0.20],
            "pct_chg_10d": [0.06, 0.02],
            "ma5": [10.0, 19.0],
            "ma10": [10.0, 21.0],
            "ma20": [10.0, 21.0],
            "volume_ma5": [1000.0, 2000.0],
            "amount_ma5": [200000000.0, 200000000.0],
            "volume_ratio_5": [2.0, 4.0],
            "high_20": [12.0, 21.0],
            "low_20": [8.0, 16.0],
            "close_position_20": [0.8, 0.6],
            "above_ma5": [True, True],
            "above_ma10": [True, False],
            "above_ma20": [True, False],
        }
    )


def test_build_candidate_pool_reads_generates_and_saves(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_daily_factors(_factor_rows())
    store.save_stock_basic(
        pd.DataFrame(
            {
                "code": ["600000", "000001"],
                "name": ["浦发银行", "平安银行"],
                "market": ["SH", "SZ"],
                "board": ["main", "main"],
                "list_status": ["L", "L"],
            }
        )
    )

    result = build_candidate_pool(
        trade_date="2026-01-02",
        top_n=1,
        min_amount_ma5=100000000.0,
        db_path=str(db_path),
    )
    saved = store.load_candidate_pool(trade_date="2026-01-02")

    assert len(result) == 1
    assert result.loc[0, "code"] == "000001"
    assert result.loc[0, "name"] == "平安银行"
    pd.testing.assert_frame_equal(saved, result)


def test_build_candidate_pool_prefers_strategy_signals(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_daily_factors(_factor_rows())
    store.save_stock_basic(
        pd.DataFrame(
            {
                "code": ["600000", "000001"],
                "name": ["浦发银行", "平安银行"],
                "market": ["SH", "SZ"],
                "board": ["main", "main"],
                "list_status": ["L", "L"],
            }
        )
    )
    store.save_strategy_signals(
        pd.DataFrame(
            {
                "trade_date": ["2026-01-02", "2026-01-02"],
                "code": ["600000", "600000"],
                "strategy_name": ["trend_pullback", "breakout_volume"],
                "signal_strength": [20.0, 30.0],
                "entry_reason": ["reason_a", "reason_b"],
                "risk_flags": ["near_20d_high", "extended_position"],
            }
        )
    )

    result = build_candidate_pool(
        trade_date="2026-01-02",
        top_n=10,
        min_amount_ma5=100000000.0,
        db_path=str(db_path),
    )

    assert result["code"].tolist() == ["600000"]
    assert set(result.loc[0, "strategy_names"].split(",")) == {"trend_pullback", "breakout_volume"}
    assert result.loc[0, "signal_count"] == 2
    assert "risk_flags" in result.columns


def test_build_candidate_pool_uses_signals_weighted_mode_with_evaluation(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_daily_factors(_factor_rows())
    store.save_strategy_signals(
        pd.DataFrame(
            {
                "trade_date": ["2026-01-02", "2026-01-02"],
                "code": ["600000", "000001"],
                "strategy_name": ["trend_pullback", "support_rebound"],
                "strategy_version": ["v1", "v1"],
                "signal_strength": [20.0, 20.0],
            }
        )
    )
    store.save_strategy_version_evaluation(
        pd.DataFrame(
            {
                "strategy_name": ["trend_pullback", "support_rebound"],
                "strategy_version": ["v1", "v1"],
                "recommendation": ["enable_observation", "observe"],
                "risk_level": ["low", "low"],
            }
        )
    )

    result, _, signal_count, evaluation_count, _, _, mode = _build_and_save_candidate_pool(
        trade_date="2026-01-02",
        top_n=10,
        min_amount_ma5=100000000.0,
        db_path=str(db_path),
    )

    assert mode == "signals_weighted"
    assert signal_count == 2
    assert evaluation_count == 2
    assert result["code"].tolist()[0] == "600000"
    assert result.set_index("code").loc["600000", "avg_strategy_weight"] == 1.2


def test_build_candidate_pool_uses_signals_unweighted_mode_without_evaluation(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_daily_factors(_factor_rows())
    store.save_strategy_signals(
        pd.DataFrame(
            {
                "trade_date": ["2026-01-02"],
                "code": ["600000"],
                "strategy_name": ["trend_pullback"],
                "signal_strength": [20.0],
            }
        )
    )

    result, _, signal_count, evaluation_count, _, _, mode = _build_and_save_candidate_pool(
        trade_date="2026-01-02",
        top_n=10,
        min_amount_ma5=100000000.0,
        db_path=str(db_path),
    )

    assert mode == "signals_unweighted"
    assert signal_count == 1
    assert evaluation_count == 0
    assert result["code"].tolist() == ["600000"]
    assert result.loc[0, "avg_strategy_weight"] == 1.0


def test_build_candidate_pool_falls_back_to_factors_without_signals(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_daily_factors(_factor_rows())

    result = build_candidate_pool(
        trade_date="2026-01-02",
        top_n=10,
        min_amount_ma5=100000000.0,
        db_path=str(db_path),
    )

    assert result["code"].tolist() == ["000001", "600000"]
    assert result["strategy_names"].isna().all()


def test_build_candidate_pool_applies_latest_market_high_risk(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_daily_factors(_factor_rows())
    store.save_market_regime(
        pd.DataFrame(
            {
                "trade_date": ["2026-01-02"],
                "market_regime": ["weak"],
                "risk_level": ["high"],
                "limit_up_count": [1],
                "limit_down_count": [30],
            }
        )
    )

    result = build_candidate_pool(
        trade_date="2026-01-02",
        top_n=10,
        min_amount_ma5=100000000.0,
        db_path=str(db_path),
    )

    assert "market_high_risk" in result.loc[0, "risk_flags"]


def test_build_candidate_pool_handles_empty_daily_factors(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.init_tables()

    result = build_candidate_pool(db_path=str(db_path))
    saved = store.load_candidate_pool()

    assert result.empty
    assert result.columns.tolist() == CANDIDATE_COLUMNS
    assert saved.empty


def test_build_candidate_pool_min_amount_ma5_help_mentions_thousand_yuan(capsys):
    try:
        _parse_args(["--help"])
    except SystemExit:
        pass

    output = capsys.readouterr().out
    assert "--min-amount-ma5" in output
    assert "thousand yuan" in output
