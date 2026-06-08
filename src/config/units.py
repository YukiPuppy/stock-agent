"""Canonical units used by stock-agent data tables."""

DATA_UNIT_VERSION = "daily_bars_v2_tushare_units"
OFFICIAL_DATA_PROVIDER = "tushare"
DAILY_BARS_VOLUME_UNIT = "手"
DAILY_BARS_AMOUNT_UNIT = "千元"
DAILY_FACTORS_AMOUNT_MA5_UNIT = "千元"
MIN_AMOUNT_MA5_UNIT = "千元"
ACTUAL_TRADES_AMOUNT_UNIT = "元"
POSITIONS_AMOUNT_UNIT = "元"

DAILY_BARS_UNIT_DESCRIPTION = (
    "正式行情源为 Tushare Pro；daily_bars 使用 Tushare Pro 标准单位："
    "volume 为手，amount 为千元。"
)
