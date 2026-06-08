from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


TABLE_LABELS = {
    "stock_basic": "股票基础信息",
    "trade_calendar": "交易日历",
    "daily_bars": "个股日线行情",
    "daily_basic": "每日基础指标",
    "stock_limits": "涨跌停价格",
    "suspend_daily": "停复牌信息",
    "index_daily": "指数日线行情",
    "limit_list_daily": "涨跌停与短线情绪",
    "market_regime": "市场环境判断",
    "moneyflow": "个股资金流原始数据",
    "moneyflow_factors": "资金流因子",
    "sw_industry_classification": "申万行业分类",
    "stock_industry_map": "股票行业映射",
    "sw_daily": "申万行业行情",
    "industry_strength": "行业强度因子",
    "daily_factors": "个股综合因子",
    "factor_diagnostics": "因子诊断",
    "strategy_signals": "策略信号",
    "candidate_pool": "候选股池",
    "trade_plan": "交易计划",
    "strategy_admission": "策略准入结果",
    "data_quality_report": "数据质量报告",
    "provider_compare_result": "数据源对齐结果",
    "strategy_version_evaluation": "策略版本评价",
    "parameter_search_results": "参数搜索结果",
    "parameter_search_performance": "参数搜索表现",
    "parameter_search_backtest_results": "参数搜索回测结果",
    "walk_forward_validation": "滚动样本外验证",
    "historical_trade_plans": "历史交易计划",
    "trade_plan_backtest_results": "交易计划回测明细",
    "trade_plan_backtest_performance": "交易计划回测表现",
    "actual_trades": "实盘交易记录",
    "execution_review": "执行复盘结果",
    "actual_trade_performance": "实盘交易表现",
    "daily_review": "盘后复盘",
    "period_review": "周期复盘",
    "positions": "当前持仓",
    "position_review": "持仓风险检查",
}

FIELD_LABELS = {
    "table_name": "数据表",
    "table": "数据表",
    "row_count": "行数",
    "rows": "行数",
    "date_range": "日期范围",
    "date_column": "日期字段",
    "has_data": "是否有数据",
    "status": "状态",
    "report_type": "报告类型",
    "latest_path": "最新报告路径",
    "key": "配置项",
    "value": "配置值",
    "trade_date": "交易日期",
    "plan_date": "计划日期",
    "as_of_date": "统计日期",
    "code": "股票代码",
    "name": "股票名称",
    "rank": "排名",
    "open": "开盘价",
    "high": "最高价",
    "low": "最低价",
    "close": "收盘价",
    "pre_close": "昨收价",
    "pct_chg": "涨跌幅",
    "pct_chg_5d": "5日涨跌幅",
    "volume": "成交量",
    "amount": "成交额",
    "turnover_rate": "换手率",
    "turnover_rate_f": "自由流通换手率",
    "volume_ratio_daily_basic": "量比",
    "volume_ratio_5": "5日量比",
    "total_mv": "总市值",
    "circ_mv": "流通市值",
    "up_limit": "涨停价",
    "down_limit": "跌停价",
    "is_suspended": "是否停牌",
    "is_limit_up_close": "是否涨停收盘",
    "is_limit_down_close": "是否跌停收盘",
    "limit_up_distance": "距涨停空间",
    "limit_down_distance": "距跌停空间",
    "amount_ma5": "5日平均成交额",
    "moneyflow_score": "资金流评分",
    "main_net_amount": "主力净流入金额",
    "main_net_amount_ratio": "主力净流入占比",
    "moneyflow_risk_flags": "资金流风险标记",
    "industry_code": "行业代码",
    "industry_name": "行业名称",
    "industry_strength_score": "行业强度评分",
    "industry_strength_level": "行业强度等级",
    "industry_return_3d": "行业3日涨跌幅",
    "industry_return_5d": "行业5日涨跌幅",
    "industry_risk_flags": "行业风险标记",
    "industry_amount_ratio_5": "行业5日成交额占比",
    "pct_change": "涨跌幅",
    "market_regime": "市场环境",
    "risk_level": "风险等级",
    "regime_reason": "市场环境说明",
    "limit_up_count": "涨停家数",
    "limit_down_count": "跌停家数",
    "break_board_count": "炸板数量",
    "sentiment_score": "短线情绪评分",
    "strategy_name": "策略名称",
    "strategy_names": "策略名称",
    "strategy_version": "策略版本",
    "strategy_versions": "策略版本",
    "signal_strength": "信号强度",
    "signal_count": "信号数量",
    "active_signal_count": "有效信号数量",
    "total_weighted_signal_strength": "加权信号强度",
    "avg_strategy_weight": "平均策略权重",
    "recommendations": "策略评价建议",
    "entry_reason": "入选理由",
    "reason": "理由",
    "score": "综合评分",
    "risk_flags": "风险标记",
    "action": "计划动作",
    "buy_price_low": "买入区间下限",
    "buy_price_high": "买入区间上限",
    "entry_low": "买入区间下限",
    "entry_high": "买入区间上限",
    "stop_loss": "止损价",
    "take_profit": "止盈参考价",
    "take_profit_1": "第一止盈参考",
    "take_profit_2": "第二止盈参考",
    "position_ratio": "建议仓位比例",
    "position_low": "建议仓位下限",
    "position_high": "建议仓位上限",
    "plan_reason": "计划理由",
    "invalid_condition": "失效条件",
    "t_plus_1_risk": "T+1 风险说明",
    "holding_volume": "持仓数量",
    "available_volume": "可用数量",
    "frozen_volume": "冻结数量",
    "cost_price": "成本价",
    "latest_price": "最新价",
    "market_value": "持仓市值",
    "floating_pnl": "浮动盈亏",
    "floating_pnl_pct": "浮动盈亏比例",
    "t_plus_1_status": "T+1 状态",
    "position_status": "持仓状态",
    "position_risk_level": "持仓风险等级",
    "position_flags": "持仓风险标记",
    "position_comment": "持仓说明",
    "next_action_hint": "后续动作提示",
    "planned_stop_loss": "计划止损价",
    "planned_take_profit_1": "计划第一止盈",
    "planned_take_profit_2": "计划第二止盈",
    "execution_score": "执行评分",
    "off_plan_count": "计划外交易数量",
    "deviation_count": "执行偏差数量",
    "chase_count": "追高偏差数量",
    "return_1d": "1日收益率",
    "return_3d": "3日收益率",
    "return_5d": "5日收益率",
    "max_drawdown_3d": "3日最大回撤",
    "actual_trade_count": "实盘交易数量",
    "avg_execution_score": "平均执行评分",
    "avg_return_3d": "平均3日收益率",
    "plan_trade_avg_return_3d": "计划内交易平均3日收益率",
    "off_plan_avg_return_3d": "计划外交易平均3日收益率",
    "admission_score": "准入评分",
    "admission_status": "准入状态",
    "admission_recommendation": "准入建议",
    "admission_reason": "准入理由",
}

VALUE_LABELS = {
    "strong": "强势",
    "neutral": "中性",
    "weak": "弱势",
    "low": "低",
    "medium": "中",
    "high": "高",
    "success": "成功",
    "failed": "失败",
    "skipped": "跳过",
    "ok": "正常",
    "warning": "警告",
    "missing": "缺失",
    "true": "是",
    "false": "否",
}

RISK_FLAG_LABELS = {
    "missing_daily_basic": "每日基础指标缺失",
    "missing_market_value": "市值数据缺失",
    "missing_volume_ratio_daily_basic": "量比数据缺失",
    "missing_turnover_rate": "换手率缺失",
    "missing_limit_data": "涨跌停数据缺失",
    "suspended": "停牌风险",
    "limit_up_close": "涨停收盘，次日可能买入困难",
    "limit_down_close": "跌停收盘，流动性风险较高",
    "market_high_risk": "市场环境偏弱",
    "main_outflow": "主力资金流出",
    "strong_main_outflow": "主力资金明显流出",
    "strong_main_inflow": "主力资金明显流入",
    "missing_moneyflow": "资金流数据缺失",
    "strong_industry": "所属行业强势",
    "weak_industry": "所属行业弱势",
    "missing_industry_strength": "行业强度数据缺失",
}


def get_table_label(name: str) -> str:
    return TABLE_LABELS.get(str(name), str(name))


def get_field_label(name: str) -> str:
    return FIELD_LABELS.get(str(name), str(name))


def translate_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return VALUE_LABELS[str(value).lower()]
    text = str(value)
    return VALUE_LABELS.get(text.lower(), text)


def translate_risk_flags(value) -> str:
    if value is None or (not isinstance(value, Iterable) and pd.isna(value)):
        return ""
    if isinstance(value, str):
        flags = [item.strip() for item in value.replace(";", ",").split(",")]
    elif isinstance(value, Iterable):
        flags = [str(item).strip() for item in value]
    else:
        flags = [str(value).strip()]
    translated = [RISK_FLAG_LABELS.get(flag, flag) for flag in flags if flag]
    return "；".join(translated)


def translate_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    translated = df.copy(deep=True)
    for column in translated.columns:
        if "risk_flags" in str(column) or str(column) in {"position_flags"}:
            translated[column] = translated[column].map(translate_risk_flags)
        else:
            translated[column] = translated[column].map(_translate_cell_value)
    return translated.rename(columns={column: get_field_label(column) for column in translated.columns})


def _translate_cell_value(value):
    if isinstance(value, bool):
        return translate_value(value)
    if isinstance(value, str) and value.lower() in VALUE_LABELS:
        return translate_value(value)
    return value
