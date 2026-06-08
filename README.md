# stock-agent

## 一、项目定位

`stock-agent` 是一个面向 A 股策略研究与日度交易计划辅助的系统。正式数据源使用 Tushare Pro，系统用于数据更新、因子构建、策略研究、日度计划、复盘分析和 LLM 辅助解释。

本项目不会连接券商接口或替用户提交订单，不构成投资建议。系统输出只用于研究、记录和辅助决策，最终判断与执行由用户自行负责。

## 二、核心原则

- 程序负责计算，包括数据更新、因子构建、策略信号、候选池、交易计划、回测和健康检查。
- Agent 负责解释结果、审查风险、总结复盘，并提出后续研究建议。
- 用户负责最终判断和执行，不能把系统输出直接等同于交易指令。
- 小样本测试结果只能验证流程是否可运行，不能直接用于实盘决策。
- 策略建议必须经过回测、样本外验证和交易计划级回测，确认风险、容量、回撤和执行约束后，才能进入人工评估。

## 三、环境配置

项目通过 `.env` 管理数据源和 LLM 配置。示例：

```env
DEFAULT_DATA_PROVIDER=tushare
TUSHARE_TOKEN=your_tushare_token
TUSHARE_API_URL=http://lianghua.nanyangqiankun.top
TUSHARE_ALLOW_NON_OFFICIAL_API_URL=true
DATA_FETCH_DISABLE_PROXY=true

ENABLE_LLM_REPORT_AGENT=true
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-v4-flash
LLM_API_KEY=your_deepseek_api_key
LLM_BASE_URL=https://api.deepseek.com
LLM_TIMEOUT_SECONDS=60
LLM_DISABLE_PROXY=true
```

注意事项：

- 不要提交真实 Tushare token。
- 不要提交真实 DeepSeek API key。
- `DATA_FETCH_DISABLE_PROXY=true` 用于数据拉取时绕过 Clash 等本地代理。
- `LLM_DISABLE_PROXY=true` 用于 DeepSeek 调用时绕过代理。
- 如果不需要 LLM 报告，可以关闭 `ENABLE_LLM_REPORT_AGENT`。

## 四、数据说明

主要数据表：

- `stock_basic`：股票基础信息。
- `trade_calendar`：交易日历。
- `daily_bars`：日线行情。
- `daily_basic`：日度基础指标。
- `stock_limits`：涨跌停价格。
- `suspend_daily`：停复牌信息。
- `index_daily`：指数日线行情。
- `limit_list_daily`：涨跌停明细。
- `market_regime`：市场环境状态。
- `moneyflow`：个股资金流。
- `moneyflow_factors`：资金流因子。
- `sw_industry_classification`：申万行业分类。
- `stock_industry_map`：股票与行业映射。
- `sw_daily`：申万行业日线数据。
- `industry_strength`：行业强弱指标。
- `daily_factors`：日度综合因子。
- `factor_diagnostics`：因子诊断结果。

关键单位：

- `daily_bars.volume` = 手。
- `daily_bars.amount` = 千元。
- `daily_factors.amount_ma5` = 千元。
- `actual_trades.amount` = 元。
- `positions` 金额字段 = 元。

## 五、常用工作流

### 1. 数据更新

小样本测试：

```bash
uv run python -m src.pipeline.run_data_update_workflow \
  --start-date 20250101 \
  --end-date 20250110 \
  --mode test
```

夜间全量挂机：

```bash
mkdir -p logs

nohup uv run python -m src.pipeline.run_data_update_workflow \
  --start-date 20240901 \
  --end-date 20250110 \
  --mode full \
  --sleep-seconds 0.5 \
  > logs/data_update_full_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

查看日志：

```bash
tail -f logs/data_update_full_*.log
```

说明：

- `daily_basic` / `moneyflow` 在部分网络环境下可能超时。
- 当前项目已确认 Tushare 第三方代理地址配置和绕过 Clash 逻辑正常。
- 家宽下大响应接口可能不稳定，手机热点可作为补充。
- 当前阶段暂不启用 `timeout` / `retry` / `failed_dates` 机制。

### 2. 因子构建

```bash
uv run python -m src.pipeline.run_factor_build_workflow
```

该流程会构建：

- `moneyflow_factors`
- `market_regime`
- `industry_strength`
- `daily_factors`
- `factor_diagnostics`

### 3. 每日盘后总流程

默认不更新数据，只运行因子、计划、健康检查和 LLM：

```bash
uv run python -m src.pipeline.run_daily_ops_workflow
```

小样本完整流程：

```bash
uv run python -m src.pipeline.run_daily_ops_workflow \
  --update-data \
  --start-date 20250101 \
  --end-date 20250110 \
  --data-update-mode test
```

说明：

- 这是日常盘后生成明日交易计划的主入口。
- 输出 `candidate_pool`、`trade_plan`、`daily_report` 和 LLM 报告。
- 该流程不会连接券商接口或替用户提交订单。

### 4. 策略研究总流程

```bash
uv run python -m src.pipeline.run_strategy_ops_workflow
```

指定训练和验证区间：

```bash
uv run python -m src.pipeline.run_strategy_ops_workflow \
  --train-start-date 2024-09-01 \
  --train-end-date 2024-12-01 \
  --validation-start-date 2024-12-02 \
  --validation-end-date 2025-01-10
```

说明：

- 用于阶段性策略研究。
- 输出策略评价、参数搜索、样本外验证、策略准入和策略研究 Agent 报告。
- 研究结果不直接启用策略，必须人工确认。

### 5. LLM Agent 总控

```bash
uv run python -m src.pipeline.run_llm_agents_workflow
```

各 Agent 作用：

- `ReportAgent`：综合总结。
- `BacktestAnalysisAgent`：回测分析。
- `RiskReviewAgent`：风险审查。
- `DailyReviewAgent`：交易执行复盘。
- `MarketRegimeAgent`：市场环境解释。
- `IndustryInsightAgent`：行业强弱解释。
- `FactorInsightAgent`：因子诊断解释。
- `StrategyResearchAgent`：策略研究建议。
- `ParameterIterationAgent`：参数候选建议。

### 6. 系统验收

```bash
uv run python -m src.pipeline.run_system_acceptance_workflow
```

说明：

- 用于检查配置、数据表、报告、策略研究表和正式配置文件。
- 不拉取数据、不构建因子、不运行策略、不调用 LLM。

## 六、日常使用建议

1. 晚上或盘后更新数据。
2. 运行 `run_daily_ops_workflow`。
3. 查看 `trade_plan` / `daily_report` / `llm_risk_review` / `llm_report_summary`。
4. 第二天盘中只关注计划内标的。
5. 不临时扩大标的范围，不执行计划外标的。
6. 收盘后导入 `actual_trades` 做复盘。

## 七、主要输出文件

`reports` 目录下常见文件：

- `data_update_workflow_*.md`
- `factor_build_workflow_*.md`
- `daily_ops_workflow_*.md`
- `strategy_ops_workflow_*.md`
- `system_acceptance_*.md`
- `daily_report_*.md`
- `llm_agents_index_*.md`
- `llm_report_summary_*.md`
- `llm_backtest_analysis_*.md`
- `llm_risk_review_*.md`
- `llm_daily_review_*.md`
- `llm_market_regime_*.md`
- `llm_industry_insight_*.md`
- `llm_factor_insight_*.md`
- `llm_strategy_research_*.md`
- `llm_parameter_iteration_*.md`

## 八、正式配置文件保护

以下正式配置文件不能由 Agent 自动修改：

- `configs/active_strategies.json`
- `configs/active_strategies_candidate.json`
- `configs/strategy_versions.json`
- `configs/parameter_search_space.json`

说明：

- `StrategyResearchAgent` 和 `ParameterIterationAgent` 只生成 `reports` 下的 candidate json。
- candidate json 必须人工确认后，才能进入正式配置。
- 正式配置变更应保留清晰记录，便于复盘策略版本和参数来源。

## 九、常见问题

### 1. `daily_basic` / `moneyflow` 超时怎么办？

先不用过度纠结，这通常可能是家宽到第三方代理链路的问题。可以分日期补拉，或者切换手机热点后再运行数据更新流程。

### 2. 为什么候选池为空？

常见原因包括数据不足、策略条件过严、样本日期太短、策略未准入。先检查数据更新范围、`daily_factors`、策略配置和策略准入报告。

### 3. 为什么 `daily_factors` 里有 NaN？

常见原因包括样本期太短、扩展数据没覆盖、接口字段缺失。部分 NaN 可以接受，但需要结合 `factor_diagnostics` 判断是否影响后续计算。

### 4. Agent 会不会直接推荐买入？

不会。Agent 只解释结果、审查风险和提出研究建议，不生成交易指令，也不会替代用户做最终判断。

### 5. 小样本测试结果能不能实盘？

不能。小样本只验证流程，不验证策略有效性。策略进入真实使用前，必须经过更完整的历史回测、样本外验证、交易计划级回测和人工评估。

## 十、开发验收命令

```bash
uv run python -m pytest
uv run python -m src.pipeline.run_system_acceptance_workflow
uv run python -m src.pipeline.run_daily_ops_workflow
uv run python -m src.pipeline.run_strategy_ops_workflow
```
