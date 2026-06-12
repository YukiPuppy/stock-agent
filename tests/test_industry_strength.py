import pandas as pd

from src.research.industry_strength import build_industry_strength


def test_build_industry_strength_generates_score_and_level():
    rows = []
    for index in range(6):
        rows.append(
            {
                "trade_date": f"2025-01-{index + 1:02d}",
                "industry_code": "801010.SI",
                "industry_name": "农林牧渔",
                "close": 100 + index,
                "pct_change": 1.0,
                "amount": 1000 + index * 100,
            }
        )
        rows.append(
            {
                "trade_date": f"2025-01-{index + 1:02d}",
                "industry_code": "801020.SI",
                "industry_name": "采掘",
                "close": 100 - index,
                "pct_change": -1.0,
                "amount": 500,
            }
        )

    result = build_industry_strength(pd.DataFrame(rows))
    latest = result[result["trade_date"] == "20250106"].set_index("industry_code")

    assert "industry_strength_score" in result.columns
    assert result["trade_date"].str.fullmatch(r"\d{8}").all()
    assert latest.loc["801010.SI", "industry_strength_level"] in {"strong", "neutral"}
    assert "weak_industry" in latest.loc["801020.SI", "industry_risk_flags"]
