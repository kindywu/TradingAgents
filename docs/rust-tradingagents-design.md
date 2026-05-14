# TradingAgents 重构需求分析文档

> **文档类型**: 需求分析（Requirements Analysis）
> **源项目**: TradingAgents v0.2.4
> **源语言**: Python (LangChain + LangGraph)
> **目标**: 重构到其他语言（如 Rust/Go/TypeScript 等）
> **日期**: 2026-05-14

---

## 目录

1. [系统概述](#1-系统概述)
2. [业务流程](#2-业务流程)
3. [功能需求](#3-功能需求)
   - [3.1 图编排引擎](#31-图编排引擎)
   - [3.2 LLM 客户端层](#32-llm-客户端层)
   - [3.3 智能体系统](#33-智能体系统)
   - [3.4 工具与数据源](#34-工具与数据源)
   - [3.5 结构化输出](#35-结构化输出)
   - [3.6 决策记忆日志](#36-决策记忆日志)
   - [3.7 事后反思](#37-事后反思)
   - [3.8 断点续跑](#38-断点续跑)
   - [3.9 CLI 交互界面](#39-cli-交互界面)
   - [3.10 报告生成](#310-报告生成)
   - [3.11 配置系统](#311-配置系统)
4. [数据模型](#4-数据模型)
5. [接口契约](#5-接口契约)
6. [非功能性需求](#6-非功能性需求)
7. [安全需求](#7-安全需求)
8. [测试需求](#8-测试需求)
9. [迁移约束](#9-迁移约束)
10. [附录](#10-附录)

---

## 1. 系统概述

### 1.1 项目定义

TradingAgents 是一个**多智能体 LLM 金融交易分析框架**。系统通过编排 12 个专业 AI 智能体协作，对给定股票在指定日期进行综合交易分析，最终输出结构化投资决策（Buy/Overweight/Hold/Underweight/Sell）。

### 1.2 核心价值

| 维度 | 描述 |
|------|------|
| **多智能体协作** | 12 个角色各司其职，模拟真实投资团队工作流 |
| **多源数据融合** | 技术面、基本面、新闻舆情、社交媒体情绪、内部交易数据 |
| **辩论机制** | 牛熊辩论 + 风险三方辩论，多视角碰撞 |
| **持续学习** | 事后反思机制，历史经验注入未来决策 |
| **多 LLM 支持** | 10+ 提供商，双模型架构（快思考+慢思考） |

### 1.3 重构目标

- **功能等价**: 所有现有功能必须在新实现中完整保留
- **性能提升**: 降低启动延迟和内存占用
- **部署简化**: 单二进制分发，消除 Python 环境依赖
- **类型安全**: 编译期类型检查，减少运行时错误
- **并发能力**: 支持多个 ticker 并行分析

---

## 2. 业务流程

### 2.1 完整执行生命周期

```
┌─────────────────────────────────────────────────────────────────┐
│                      1. 初始化阶段                               │
├─────────────────────────────────────────────────────────────────┤
│  加载配置 → 创建 LLM 客户端(快+慢) → 初始化数据源 → 构建图       │
│  → 读取记忆日志 → 解析 pending 条目 → 获取收益 → 反思 → 注入上下文│
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      2. 分析执行阶段                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  I. 分析师团队 (顺序执行)                              │       │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐│       │
│  │  │ Market   │→│ Social   │→│ News     │→│Fundament││       │
│  │  │ Analyst  │  │ Analyst  │  │ Analyst  │  │Analyst  ││       │
│  │  │          │  │          │  │          │  │         ││       │
│  │  │ tools:   │  │ tools:   │  │ tools:   │  │ tools:  ││       │
│  │  │ stock    │  │ news     │  │ news     │  │fundament││       │
│  │  │indicators│  │          │  │global_news│ │balance  ││       │
│  │  │          │  │          │  │insider   │  │cashflow ││       │
│  │  └──────────┘  └──────────┘  └──────────┘  │income   ││       │
│  │                                              └────────┘│       │
│  └──────────────────────────────────────────────────────┘       │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  II. 研究辩论 (多轮循环)                                │       │
│  │  ┌──────────────┐        ┌──────────────┐             │       │
│  │  │ Bull         │ ←───→  │ Bear         │             │       │
│  │  │ Researcher   │ 辩论N轮 │ Researcher   │             │       │
│  │  └──────────────┘        └──────────────┘             │       │
│  │              │                  │                      │       │
│  │              └────────┬─────────┘                      │       │
│  │                       ▼                                │       │
│  │              ┌────────────────┐                        │       │
│  │              │Research Manager│ → ResearchPlan         │       │
│  │              └────────────────┘                        │       │
│  └──────────────────────────────────────────────────────┘       │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  III. 交易决策                                         │       │
│  │  ┌────────────────┐                                   │       │
│  │  │    Trader      │ → TraderProposal                  │       │
│  │  └────────────────┘                                   │       │
│  └──────────────────────────────────────────────────────┘       │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  IV. 风险辩论 (三角多轮循环)                            │       │
│  │  ┌──────────────┐                                      │       │
│  │  │  Aggressive  │←──────→┌──────────────────┐         │       │
│  │  │   Analyst    │         │  Conservative    │         │       │
│  │  └──────────────┘         │    Analyst       │         │       │
│  │       ↑    │              └──────────────────┘         │       │
│  │       │    └────────────→ ┌──────────────────┐         │       │
│  │       └───────────────────│    Neutral       │         │       │
│  │              N轮循环       │    Analyst       │         │       │
│  │                           └──────────────────┘         │       │
│  │              │                  │                       │       │
│  │              └────────┬─────────┘                       │       │
│  │                       ▼                                 │       │
│  │              ┌────────────────────┐                     │       │
│  │              │ Portfolio Manager  │ → PortfolioDecision │       │
│  │              └────────────────────┘                     │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      3. 后处理阶段                               │
├─────────────────────────────────────────────────────────────────┤
│  信号提取 → 状态持久化(JSON) → 追加 pending 决策到记忆日志       │
│  → 清除 checkpoint → 展示/导出报告                               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 关键时序约束

| 约束 | 说明 |
|------|------|
| **分析师顺序** | 默认按 market → social → news → fundamentals 顺序执行（每个分析师产生完整报告后，下一个才开始）。架构上可以改为并行 |
| **分析师工具循环** | 每个分析师内部 LLM 与工具之间循环调用，直到 LLM 不再要求工具调用 |
| **分析师间消息清理** | 每个分析师完成后，messages 清空并置入 "Continue" 占位消息（Anthropic 兼容性要求） |
| **辩论轮次** | Bull↔Bear 最多 `max_debate_rounds * 2` 回合（默认 2 回合即 1 轮）；Risk 三角最多 `max_risk_discuss_rounds * 3` 回合（默认 3 回合） |
| **结构化输出串行** | Research Manager → Trader → Portfolio Manager 严格顺序，下游依赖上游输出 |
| **反思延迟** | 反思在下一次同 ticker 运行时执行（Phase B），因为需要实际市场结果数据（需等待交易日过后） |
| **checkpoint 生命周期** | 节点成功后立即保存 checkpoint；整个 pipeline 成功后清除 |

### 2.3 快速/深度模型分工

```
quick_thinking_llm (快速模型, e.g. GPT-5.4-mini):
  ├── Market Analyst
  ├── Social Media Analyst
  ├── News Analyst
  ├── Fundamentals Analyst
  ├── Bull Researcher
  ├── Bear Researcher
  ├── Trader
  ├── Aggressive Risk Analyst
  ├── Conservative Risk Analyst
  ├── Neutral Risk Analyst
  └── Reflector (事后反思)

deep_thinking_llm (深度模型, e.g. GPT-5.4):
  ├── Research Manager
  └── Portfolio Manager
```

---

## 3. 功能需求

### 3.1 图编排引擎

#### FR-1.1 节点管理

- **FR-1.1.1**: 系统必须支持注册不同类型执行节点（智能体节点、工具调用节点、消息清理节点）
- **FR-1.1.2**: 每个节点必须具有唯一标识（名称或 ID）
- **FR-1.1.3**: 节点必须支持异步执行
- **FR-1.1.4**: 节点执行失败时必须有明确的错误传播机制

#### FR-1.2 边与路由

- **FR-1.2.1**: 系统必须支持**固定边**：节点 A 执行后无条件转移到节点 B
- **FR-1.2.2**: 系统必须支持**条件边**：节点 A 执行后根据状态决定下一节点
- **FR-1.2.3**: 条件路由函数必须是**纯函数**：接收状态引用，返回下一节点标识

#### FR-1.3 条件路由逻辑

必须实现以下三种条件路由模式：

**模式 1 — 分析师工具循环：**
```
输入: AgentState
判断: last_message 是否包含 tool_calls
  是 → 路由到工具执行节点 (tools_{analyst_type})
  否 → 路由到消息清理节点 (Msg Clear {analyst_type})
```

**模式 2 — 牛/熊辩论循环：**
```
输入: AgentState
判断:
  IF debate_state.count >= 2 * max_debate_rounds → Research Manager
  ELSE IF current_response 以 "Bull" 开头 → Bear Researcher
  ELSE → Bull Researcher
```

**模式 3 — 风险三角辩论循环：**
```
输入: AgentState
判断:
  IF risk_state.count >= 3 * max_risk_discuss_rounds → Portfolio Manager
  ELSE IF latest_speaker == "Aggressive" → Conservative Analyst
  ELSE IF latest_speaker == "Conservative" → Neutral Analyst
  ELSE → Aggressive Analyst
```

#### FR-1.4 状态传播

- **FR-1.4.1**: 所有节点共享一个可变状态对象
- **FR-1.4.2**: 每个节点返回状态更新（部分更新），图引擎负责合并
- **FR-1.4.3**: 状态必须支持序列化/反序列化（用于 checkpoint）

#### FR-1.5 执行模式

- **FR-1.5.1**: 必须支持**流式执行**（逐节点输出中间状态，用于 UI 实时更新）
- **FR-1.5.2**: 必须支持**批量执行**（返回到达 END 后的最终状态）
- **FR-1.5.3**: 必须支持**递归深度限制**（可配置，防止无限循环）
- **FR-1.5.4**: 必须支持**消息去重**（基于消息 ID，防止流式模式下重复处理）

#### FR-1.6 消息管理

- **FR-1.6.1**: 支持消息追加（AI/人类/工具/系统消息）
- **FR-1.6.2**: 支持消息批量删除（用于分析师间清理上下文）
- **FR-1.6.3**: 删除消息后必须插入最小占位消息（"Continue"），以满足 Anthropic API 要求（不允许空的 messages 列表）

---

### 3.2 LLM 客户端层

#### FR-2.1 多提供商支持

系统必须支持以下 10 个 LLM 提供商，并能通过配置切换：

| # | 提供商 | API 类型 | 默认端点 | 认证 Env Var | 特殊需求 |
|---|--------|----------|----------|-------------|----------|
| 1 | **OpenAI** | Responses API (`/v1/responses`) | `api.openai.com` | `OPENAI_API_KEY` | `reasoning_effort` 参数; 结构化输出用 `json_schema` |
| 2 | **Anthropic** | Messages API | `api.anthropic.com` | `ANTHROPIC_API_KEY` | `effort` 参数; 结构化输出用 tool-use; content 是数组格式 |
| 3 | **Google** | Gemini API | `generativelanguage.googleapis.com` | `GOOGLE_API_KEY` | `thinking_level` (Gemini 3) / `thinking_budget` (Gemini 2.5) |
| 4 | **xAI** | OpenAI-compatible `/v1` | `api.x.ai` | `XAI_API_KEY` | 标准 Chat Completions |
| 5 | **DeepSeek** | OpenAI-compatible `/v1` | `api.deepseek.com` | `DEEPSEEK_API_KEY` | **thinking-mode 回传** (见 FR-2.8) |
| 6 | **Qwen** | OpenAI-compatible `/v1` | `dashscope-intl.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` | — |
| 7 | **GLM** | OpenAI-compatible `/v1` | `api.z.ai/api/paas/v4/` | `ZHIPU_API_KEY` | — |
| 8 | **OpenRouter** | OpenAI-compatible `/v1` | `openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | — |
| 9 | **Ollama** | OpenAI-compatible `/v1` | `localhost:11434/v1` | 无 | 结构化输出可能不可用，需自动降级 |
| 10 | **Azure** | Azure OpenAI | 用户指定 | Azure 认证 | 需 `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `OPENAI_API_VERSION` |

#### FR-2.2 统一 LLM 接口

所有提供商必须实现统一的调用接口：

```
接口: LlmClient
方法:
  + chat(messages: List<Message>) -> Response
      基础对话：发送消息列表，返回 AI 响应

  + chat_with_tools(messages: List<Message>, tools: List<ToolDef>) -> Response
      带工具绑定的对话：发送消息+工具定义，返回 AI 响应（可能包含 tool_calls）

  + chat_structured<T: Schema>(messages: List<Message>, schema: JsonSchema) -> T
      结构化输出：发送消息+schema，返回反序列化的类型实例

  + validate_model() -> bool
      模型名称校验：检查是否在已知模型列表中
```

#### FR-2.3 内容规范化

- **FR-2.3.1**: 所有提供商的 `response.content` 必须归一化为纯文本字符串
- **FR-2.3.2**: 对于返回列表结构内容的提供商（OpenAI Responses API、Gemini 3），必须提取并拼接所有 `type: "text"` 块的内容
- **FR-2.3.3**: 非文本块（如 reasoning/metadata）应在归一化过程中丢弃

#### FR-2.4 结构化输出支持

- **FR-2.4.1**: 系统必须支持通过 JSON Schema 约束 LLM 输出
- **FR-2.4.2**: 对于不支持结构化输出的提供商/模型，必须**自动降级**为自由文本生成
- **FR-2.4.3**: 降级时记录 WARNING 日志，不阻塞主流程
- **FR-2.4.4**: 结构化输出调用失败时（JSON 解析错误等），应**自动重试一次**自由文本路径

#### FR-2.5 自定义端点

- **FR-2.5.1**: 所有提供商必须支持通过配置指定自定义 `base_url`
- **FR-2.5.2**: 自定义 URL 优先级高于提供商默认 URL

#### FR-2.6 双模型架构

- **FR-2.6.1**: 系统必须维护两个独立的 LLM 客户端实例（快速 + 深度）
- **FR-2.6.2**: 两个实例可以使用不同模型但必须使用**同一提供商**
- **FR-2.6.3**: 必须支持提供商特定的思考参数配置（OpenAI `reasoning_effort`、Anthropic `effort`、Google `thinking_level`）

#### FR-2.7 HTTP 客户端

- **FR-2.7.1**: 必须支持自定义 HTTP 客户端注入（连接池、代理、TLS 配置）
- **FR-2.7.2**: 必须内置重试逻辑（最少 3 次，指数退避：1s、2s、4s）
- **FR-2.7.3**: 必须收集每次调用的 token 使用统计（输入/输出 token 数）

#### FR-2.8 DeepSeek Thinking-Mode 特殊处理

- **FR-2.8.1**: 当 DeepSeek 返回 `reasoning_content` 字段时，必须在下一轮请求中将该字段作为 assistant message 的一部分原样返回给 API
- **FR-2.8.2**: 若不回传 `reasoning_content`，DeepSeek API 将返回 HTTP 400 错误
- **FR-2.8.3**: `deepseek-reasoner` 模型不支持 `tool_choice`，结构化输出对该模型必须自动降级

#### FR-2.9 模型目录

- **FR-2.9.1**: 必须维护各提供商的已知模型列表（用于 CLI 选择器）
- **FR-2.9.2**: 支持自定义模型 ID 输入（不在已知列表中时发出 Warning 但允许使用）
- **FR-2.9.3**: Ollama 和 OpenRouter 不做模型名校验（任何模型名均接受）

---

### 3.3 智能体系统

#### FR-3.1 智能体清单

系统必须实现以下 12 个智能体：

| # | 智能体 | 使用的模型 | 绑定工具 | 结构化输出 | 输出写入字段 |
|---|--------|----------|---------|-----------|------------|
| 1 | **Market Analyst** | Quick | `get_stock_data`, `get_indicators` | — | `market_report` |
| 2 | **Social Media Analyst** | Quick | `get_news` | — | `sentiment_report` |
| 3 | **News Analyst** | Quick | `get_news`, `get_global_news`, `get_insider_transactions` | — | `news_report` |
| 4 | **Fundamentals Analyst** | Quick | `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement` | — | `fundamentals_report` |
| 5 | **Bull Researcher** | Quick | — | — | `investment_debate_state` |
| 6 | **Bear Researcher** | Quick | — | — | `investment_debate_state` |
| 7 | **Research Manager** | Deep | — | `ResearchPlan` | `investment_plan` |
| 8 | **Trader** | Quick | — | `TraderProposal` | `trader_investment_plan` |
| 9 | **Aggressive Risk Analyst** | Quick | — | — | `risk_debate_state` |
| 10 | **Conservative Risk Analyst** | Quick | — | — | `risk_debate_state` |
| 11 | **Neutral Risk Analyst** | Quick | — | — | `risk_debate_state` |
| 12 | **Portfolio Manager** | Deep | — | `PortfolioDecision` | `final_trade_decision` |

#### FR-3.2 智能体分类模式

所有 12 个智能体遵循以下三种模式之一：

**类型 A — 带工具调用的分析师（Agent 1-4）：**
```
流程:
  1. 从 AgentState 提取 company_of_interest, trade_date
  2. 构建 instrument_context（保留交易所后缀 e.g. .TO, .HK）
  3. 构建系统提示（含角色定义 + 工具描述 + 输出格式要求 + 语言指令）
  4. 绑定工具到 LLM
  5. 进入工具调用循环：LLM 返回 tool_calls → 执行工具 → 将结果加入 messages → 再次调用 LLM
  6. 循环终止条件：LLM 返回纯文本（无 tool_calls）
  7. 将最终文本写入对应报告字段
  8. 要求输出包含 Markdown 表格（整理关键信息）
```

**类型 B — 辩论参与者（Agent 5-6, 9-11）：**
```
流程:
  1. 读取所有分析师报告（4个）和上游决策
  2. 读取辩论历史和对方的最新论点
  3. 构建辩论提示（含角色立场 + 反驳策略 + 论据来源）
  4. 调用 LLM（无工具绑定）
  5. 将新论点追加到辩论历史
  6. 更新自己的 *_history 字段
  7. 递增辩论计数
  8. 发言人标识: "Bull Analyst:", "Bear Analyst:", "Aggressive Risk Analyst:" 等
```

**类型 C — 结构化决策者（Agent 7-8, 12）：**
```
流程:
  1. 读取上游输出（辩论历史/投资计划/交易计划等）
  2. 构建决策提示（含评级体系说明 + 上下文）
  3. 绑定结构化输出 schema
  4. 调用 LLM 生成类型化输出
  5. 渲染为 Markdown 文本
  6. 写入对应状态字段
```

#### FR-3.3 提示管理

- **FR-3.3.1**: 所有智能体的系统提示必须从代码中**分离**，存储在外部文件中
- **FR-3.3.2**: 提示必须支持**变量插值**（至少支持 `{company_of_interest}`, `{trade_date}`, `{report}`, `{history}`, `{current_response}` 等占位符）
- **FR-3.3.3**: 提示修改不需要重新编译
- **FR-3.3.4**: 必须支持多语言输出指令（`get_language_instruction()` 逻辑）

#### FR-3.4 智能体选择

- **FR-3.4.1**: 四个分析师可选配（用户可任意组合，如仅选 Market + News）
- **FR-3.4.2**: 其余 8 个智能体固定执行，不可跳过
- **FR-3.4.3**: 若未选择任何分析师，必须报错

#### FR-3.5 工具调用循环

- **FR-3.5.1**: 工具调用循环必须在通用逻辑中实现，而非在每个分析师中重复
- **FR-3.5.2**: 循环终止条件：LLM 返回的消息中 `tool_calls` 为空或不存在
- **FR-3.5.3**: 每次工具调用结果以 `ToolMessage` 形式追加到消息历史
- **FR-3.5.4**: 工具调用循环必须有最大迭代次数限制（防止无限循环）

---

### 3.4 工具与数据源

#### FR-4.1 工具清单

系统必须实现以下 9 个数据获取工具（作为 LLM Function Calling 工具）：

```
类别: core_stock_apis (核心行情)
├── get_stock_data(symbol, start_date, end_date) → OHLCV 数据 (CSV 格式)

类别: technical_indicators (技术指标)
├── get_indicators(symbol, indicator, curr_date, look_back_days) → 指标值序列

类别: fundamental_data (基本面)
├── get_fundamentals(ticker, curr_date) → 公司基本面摘要 (28个字段)
├── get_balance_sheet(ticker, freq, curr_date) → 资产负债表 (CSV)
├── get_cashflow(ticker, freq, curr_date) → 现金流表 (CSV)
├── get_income_statement(ticker, freq, curr_date) → 利润表 (CSV)

类别: news_data (新闻与事件)
├── get_news(ticker, start_date, end_date) → 公司新闻
├── get_global_news(curr_date, look_back_days, limit) → 全球宏观新闻
└── get_insider_transactions(ticker) → 内部交易数据 (CSV)
```

#### FR-4.2 工具-智能体绑定

| 智能体 | 绑定的工具 |
|--------|----------|
| Market Analyst | `get_stock_data`, `get_indicators` |
| Social Media Analyst | `get_news` |
| News Analyst | `get_news`, `get_global_news`, `get_insider_transactions` |
| Fundamentals Analyst | `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement` |

#### FR-4.3 技术指标

支持的 13 个技术指标（通过 yfinance/stockstats 计算）：

| 类别 | 指标名 | 说明 |
|------|--------|------|
| 移动平均 | `close_50_sma` | 50日简单移动平均 |
| 移动平均 | `close_200_sma` | 200日简单移动平均 |
| 移动平均 | `close_10_ema` | 10日指数移动平均 |
| MACD | `macd` | MACD 线 |
| MACD | `macds` | MACD 信号线 |
| MACD | `macdh` | MACD 柱状图 |
| 动量 | `rsi` | 相对强弱指标 |
| 动量 | `mfi` | 资金流量指标 |
| 波动性 | `boll` | 布林带中轨 (20 SMA) |
| 波动性 | `boll_ub` | 布林带上轨 |
| 波动性 | `boll_lb` | 布林带下轨 |
| 波动性 | `atr` | 平均真实波幅 |
| 成交量 | `vwma` | 成交量加权移动平均 |

每个指标必须附带：指标描述文本、使用建议、注意事项。

#### FR-4.4 数据供应商

**FR-4.4.1**: 必须支持至少两个数据供应商：

| 供应商 | 数据来源 | 认证需求 | 速率限制 |
|--------|---------|---------|---------|
| **yfinance** | Yahoo Finance (公开 HTTP API) | 无 | 有 (`YFRateLimitError`) |
| **Alpha Vantage** | Alpha Vantage API | `ALPHA_VANTAGE_API_KEY` | 严格（免费版 25 req/day） |

**FR-4.4.2**: 必须支持按数据类别配置供应商（`data_vendors` 配置）：
```text
core_stock_apis:     "yfinance" | "alpha_vantage"
technical_indicators: "yfinance" | "alpha_vantage"
fundamental_data:     "yfinance" | "alpha_vantage"
news_data:            "yfinance" | "alpha_vantage"
```

**FR-4.4.3**: 必须支持按工具级别覆盖供应商配置（`tool_vendors` 配置）：
```text
e.g. "get_stock_data": "alpha_vantage"  # 仅该工具使用 alpha_vantage
```

#### FR-4.5 供应商路由与回退

- **FR-4.5.1**: 当主供应商因**速率限制**（`AlphaVantageRateLimitError` / `YFRateLimitError`）失败时，必须自动切换到备用供应商
- **FR-4.5.2**: 其他类型的错误（网络错误、数据不存在等）不应触发回退，直接向上传播
- **FR-4.5.3**: 回退链顺序：配置的主供应商 → 配置中列出的其他供应商 → 所有其余可用供应商
- **FR-4.5.4**: 所有供应商均失败时，抛出明确的错误信息

#### FR-4.6 数据缓存

- **FR-4.6.1**: OHLCV 数据必须缓存到本地文件（减少重复网络请求）
- **FR-4.6.2**: 缓存按 ticker 组织，存储在 `data_cache_dir`
- **FR-4.6.3**: 缓存必须有过期策略

#### FR-4.7 防止前视偏差

- **FR-4.7.1**: 所有数据获取必须按 `trade_date` 过滤，LLM 不能看到未来数据
- **FR-4.7.2**: 财务报表数据必须过滤掉 `trade_date` 之后的列
- **FR-4.7.3**: OHLCV 数据加载只包含 `trade_date` 之前的历史

---

### 3.5 结构化输出

#### FR-5.1 三个结构化模式

**ResearchPlan** (研究经理):
```text
字段:
  - recommendation: enum {Buy, Overweight, Hold, Underweight, Sell}
  - rationale: string (自然语言推理，对话风格)
  - strategic_actions: string (给交易员的行动方案)

渲染格式:
  **Recommendation**: {recommendation}
  **Rationale**: {rationale}
  **Strategic Actions**: {strategic_actions}
```

**TraderProposal** (交易员):
```text
字段:
  - action: enum {Buy, Hold, Sell}
  - reasoning: string (2-4句推理)
  - entry_price: float? (入场价)
  - stop_loss: float? (止损价)
  - position_sizing: string? (仓位建议, e.g. "5% of portfolio")

渲染格式:
  **Action**: {action}
  **Reasoning**: {reasoning}
  [**Entry Price**: {entry_price}]
  [**Stop Loss**: {stop_loss}]
  [**Position Sizing**: {position_sizing}]
  FINAL TRANSACTION PROPOSAL: **{ACTION}**
```

**PortfolioDecision** (投资组合经理):
```text
字段:
  - rating: enum {Buy, Overweight, Hold, Underweight, Sell}
  - executive_summary: string (2-4句行动计划)
  - investment_thesis: string (详细推理，锚定分析师证据)
  - price_target: float? (目标价)
  - time_horizon: string? (持有期, e.g. "3-6 months")

渲染格式:
  **Rating**: {rating}
  **Executive Summary**: {executive_summary}
  **Investment Thesis**: {investment_thesis}
  [**Price Target**: {price_target}]
  [**Time Horizon**: {time_horizon}]
```

#### FR-5.2 模式管理

- **FR-5.2.1**: 每个模式必须有对应的 JSON Schema 生成能力
- **FR-5.2.2**: 每个模式必须有 Markdown 渲染函数
- **FR-5.2.3**: 渲染后的 Markdown 必须保持向后兼容（下游消费者和记忆日志依赖固定格式）

#### FR-5.3 优雅降级路径

```
优先级 1: structured_llm.invoke(prompt, schema) → 解析为类型实例 → render()
优先级 2 (降级): plain_llm.invoke(prompt) → 自由文本 → 直接使用 content
```

降级触发条件：
- 提供商不支持结构化输出（e.g. Ollama 旧模型）
- `deepseek-reasoner` 模型（无 `tool_choice` 支持）
- JSON 解析失败（弱模型输出格式不正确）
- API 返回异常

---

### 3.6 决策记忆日志

#### FR-6.1 存储格式

- **FR-6.1.1**: 记忆日志必须以**追加式 Markdown 文件**存储
- **FR-6.1.2**: 文件路径默认为 `~/.tradingagents/memory/trading_memory.md`（可通过环境变量/配置覆盖）
- **FR-6.1.3**: 使用 HTML 注释 `<!-- ENTRY_END -->` 作为条目分隔符（LLM 输出中不会出现的硬分隔符）

#### FR-6.2 日志条目格式

```
[{date} | {ticker} | {rating} | {status_or_return} | {alpha} | {holding_days}d]

DECISION:
{final_trade_decision 的完整 Markdown}

REFLECTION:
{反思文本}

<!-- ENTRY_END -->
```

Tag 行字段（管道分隔）：
1. `date` — 交易日期 (YYYY-MM-DD)
2. `ticker` — 股票代码
3. `rating` — 评级 (Buy/Overweight/Hold/Underweight/Sell)
4. 状态 — `pending` 或原始收益率 (e.g. `+3.2%`)
5. alpha — `pending` 或 Alpha vs SPY (e.g. `+1.5%`)
6. 持有天数 — `pending` 或 `5d`

#### FR-6.3 两阶段生命周期

**Phase A — 存储决策（`store_decision`）：**

执行时机: `propagate()` 完成时（每次分析运行结束时）

- FR-6.3.1: 从 PM 的 `final_trade_decision` 文本中提取评级
- FR-6.3.2: 构建 `[date | ticker | rating | pending]` tag
- FR-6.3.3: 以追加模式写入日志文件（tag + DECISION section）
- FR-6.3.4: **幂等性保证**：若文件中已存在相同 `[date | ticker | pending]` 条目，则跳过（防止重复写入）

**Phase B — 解析结果（`batch_update_with_outcomes`）：**

执行时机: 下一次同 ticker 的 `propagate()` 开始时

- FR-6.3.5: 加载所有 `pending` 条目
- FR-6.3.6: 仅为**同 ticker** 的 pending 条目获取实际收益数据（通过 yfinance）
- FR-6.3.7: 对每个条目调用 LLM 生成反思
- FR-6.3.8: 批量原子更新所有条目（更新 tag + 追加 REFLECTION section）
- FR-6.3.9: **原子写入保证**：先写临时文件，再 `rename` 替换原文件（防止写入中途崩溃导致文件损坏）

#### FR-6.4 上下文注入

- **FR-6.4.1**: `get_past_context(ticker)` 在每次运行时提取历史经验
- **FR-6.4.2**: 同 ticker 条目：最多 5 条，包含完整决策 + 反思
- **FR-6.4.3**: 跨 ticker 条目：最多 3 条，仅包含反思摘要
- **FR-6.4.4**: 提取的上下文注入到 Portfolio Manager 的提示中

#### FR-6.5 日志轮换

- **FR-6.5.1**: 当 `memory_log_max_entries` 配置为数值时，已解析条目数量超过该值则删除最旧的
- **FR-6.5.2**: **Pending 条目永不被删除**（代表未完成的工作）
- **FR-6.5.3**: `memory_log_max_entries` 为 `null/None` 时，禁用轮换

#### FR-6.6 条目解析

- **FR-6.6.1**: 必须从 Markdown 文本中正确解析 tag 行的各字段（管道分隔）
- **FR-6.6.2**: 必须从正文中正确提取 DECISION 和 REFLECTION 部分
- **FR-6.6.3**: 解析必须容忍格式变体（额外的空白、缺失的可选字段）

---

### 3.7 事后反思

#### FR-7.1 反思生成

- **FR-7.1.1**: 反思使用 `quick_thinking_llm` 生成（节省成本）
- **FR-7.1.2**: 反思必须覆盖三个要点：
  1. 方向性判断是否正确？（引用 Alpha 数据）
  2. 投资论文的哪部分成立/失败？
  3. 一个具体可操作的教训
- **FR-7.1.3**: 反思输出为 2-4 句纯文本（无列表、无标题、无 Markdown）
- **FR-7.1.4**: 反思内容紧凑，适合注入到未来 LLM 提示的上下文中

#### FR-7.2 收益计算

- **FR-7.2.1**: 计算 `raw_return = (close[N] - close[0]) / close[0]`，默认 N=5 天
- **FR-7.2.2**: 计算 `alpha = raw_return - spy_return`（SPY 作为基准）
- **FR-7.2.3**: 若价格数据不可用（太新、退市、网络错误），返回 None 并在下次运行时重试
- **FR-7.2.4**: 数据获取使用 yfinance（不依赖配置的数据供应商，因为收益计算是内部逻辑）

#### FR-7.3 反思失败处理

- **FR-7.3.1**: LLM 反思调用失败时，应重试（至少 3 次）
- **FR-7.3.2**: 反思失败不应阻塞主分析流程（该条目保持 pending 状态）

---

### 3.8 断点续跑

#### FR-8.1 检查点存储

- **FR-8.1.1**: 检查点默认存储为 SQLite 数据库（每个 ticker 一个独立 DB 文件）
- **FR-8.1.2**: 存储位置：`{data_cache_dir}/checkpoints/{TICKER}.db`
- **FR-8.1.3**: Thread ID 必须是确定性的：`SHA256(ticker_upper:date)[:16]`
- **FR-8.1.4**: 检查点启用通过 `checkpoint_enabled` 配置控制

#### FR-8.2 检查点生命周期

```
每个节点执行完成后 → save_checkpoint(stage, state)
整个 Pipeline 执行成功 → clear_checkpoint(ticker, date)
执行中崩溃/中断 → checkpoint 保留（用于下次 resume）
```

#### FR-8.3 断点恢复

- **FR-8.3.1**: `propagate()` 启动时检查是否存在同 ticker+date 的 checkpoint
- **FR-8.3.2**: 若存在：加载保存的状态，从保存的 stage 继续执行
- **FR-8.3.3**: 若不存在：从初始状态开始执行
- **FR-8.3.4**: 恢复后，已完成的节点不再重新执行

#### FR-8.4 检查点管理

- **FR-8.4.1**: 支持清除单个 ticker+date 的检查点
- **FR-8.4.2**: 支持清除所有检查点
- **FR-8.4.3**: 成功完成分析后自动清除检查点

#### FR-8.5 状态序列化

- **FR-8.5.1**: 完整的 `AgentState` 必须可序列化为 JSON
- **FR-8.5.2**: 序列化后的状态必须能完整反序列化恢复
- **FR-8.5.3**: 序列化不丢失关键信息（消息历史、辩论状态、报告内容）

---

### 3.9 CLI 交互界面

#### FR-9.1 启动流程

CLI 必须提供以下交互式输入步骤：

| 步骤 | 提示 | 输入类型 | 默认值 | 校验 |
|------|------|---------|--------|------|
| 1 | Ticker Symbol | 文本输入 | SPY | 非空 |
| 2 | Analysis Date | 日期输入 YYYY-MM-DD | 今天 | 不能是未来日期 |
| 3 | Output Language | 单选 | English | — |
| 4 | Analyst Selection | 多选 | 全选 | 至少选一个 |
| 5 | Research Depth | 单选 | Shallow (1 round) | — |
| 6 | LLM Provider | 单选 | OpenAI | — |
| 7 | Quick-thinking Model | 单选 | 提供商默认 | — |
| 8 | Deep-thinking Model | 单选 | 提供商默认 | — |
| 9 | Thinking Config | 单选 (条件) | 无 | 仅当提供商支持时显示 |

#### FR-9.2 实时仪表盘

必须显示包含以下 5 个区域的 Live 界面：

| 区域 | 内容 | 刷新频率 |
|------|------|----------|
| **Header** | 欢迎信息 + 版权 | 静态 |
| **Progress** | 智能体状态表（5 团队 12 智能体），颜色编码：pending=黄 / in_progress=蓝(旋转) / completed=绿 / error=红 | 4 Hz |
| **Messages** | 最近消息和工具调用（最新 12 条，含时间戳） | 4 Hz |
| **Analysis** | 当前报告 Section（实时更新的 Markdown） | 4 Hz |
| **Footer** | 统计条：智能体进度 | LLM 调用次数 | 工具调用次数 | Token 统计 | 报告进度 | 耗时 | 4 Hz |

#### FR-9.3 状态追踪

- **FR-9.3.1**: 必须自动从 stream chunks 推断智能体状态转换 (pending → in_progress → completed)
- **FR-9.3.2**: 必须按 Section 追踪报告完成情况
- **FR-9.3.3**: "报告完成"的判断标准：报告有内容 AND 该 Section 最终化智能体状态为 completed

#### FR-9.4 统计收集

- **FR-9.4.1**: 必须收集 LLM 调用总次数
- **FR-9.4.2**: 必须收集工具调用总次数
- **FR-9.4.3**: 必须收集 Token 使用统计（输入/输出）
- **FR-9.4.4**: 必须记录总耗时（从 pipeline 开始到结束）

#### FR-9.5 命令行参数

```
tradingagents analyze [OPTIONS]

Options:
  --checkpoint         启用断点续跑 (checkpoint 模式)
  --clear-checkpoints  运行前清除所有已有检查点
```

#### FR-9.6 分析后交互

- **FR-9.6.1**: 提示用户是否保存报告（Y/N），若保存则选择路径
- **FR-9.6.2**: 提示用户是否在屏幕上显示完整报告（Y/N）
- **FR-9.6.3**: 完整报告按 Section 依次展示（避免终端截断）

#### FR-9.7 CLI 框架能力

- **FR-9.7.1**: 必须支持 Shell 自动补全
- **FR-9.7.2**: 必须支持帮助命令 (`--help`)
- **FR-9.7.3**: 支持非交互模式（直接传参数跳过交互式问答）

---

### 3.10 报告生成

#### FR-10.1 输出格式

系统必须生成以下格式的报告：

| 格式 | 内容 |
|------|------|
| **Markdown (.md)** | 各 Section 独立文件 + 完整合并文件 |
| **HTML (.html)** | 各 Section 独立文件 + 完整合并文件 + 索引页面 |
| **JSON** | 完整 AgentState 快照 |

#### FR-10.2 文件结构

```
{save_path}/
├── index.html                     # 索引页面（含目录链接）
├── complete_report.md             # 完整报告（合并所有 Section）
├── complete_report.html           # 完整报告 HTML
├── 1_analysts/
│   ├── market.md / market.html
│   ├── sentiment.md / sentiment.html
│   ├── news.md / news.html
│   └── fundamentals.md / fundamentals.html
├── 2_research/
│   ├── bull.md / bull.html
│   ├── bear.md / bear.html
│   └── manager.md / manager.html
├── 3_trading/
│   └── trader.md / trader.html
├── 4_risk/
│   ├── aggressive.md / aggressive.html
│   ├── conservative.md / conservative.html
│   ├── neutral.md / neutral.html
│   └── decision.md / decision.html
└── 5_portfolio/
    └── decision.md / decision.html
```

#### FR-10.3 HTML 要求

- **FR-10.3.1**: 所有 CSS 必须内嵌（无外部依赖）
- **FR-10.3.2**: 响应式布局（max-width: 900px）
- **FR-10.3.3**: 表格有边框和交替行背景色
- **FR-10.3.4**: 代码块有背景色和等宽字体
- **FR-10.3.5**: 页面间有相对路径导航链接

#### FR-10.4 JSON 状态日志

```text
路径: {results_dir}/{TICKER}/TradingAgentsStrategy_logs/full_states_log_{date}.json
内容: AgentState 的完整 JSON 序列化（含所有报告、辩论状态、最终决策）
```

#### FR-10.5 CLI 消息日志

```text
路径: {results_dir}/{TICKER}/{date}/message_tool.log
格式: [HH:MM:SS] [MessageType] content
      [HH:MM:SS] [Tool Call] tool_name(key=value, ...)
```

#### FR-10.6 实时报告写入

- **FR-10.6.1**: 分析进行中时，每个 Section 完成后立即写入对应 .md 文件
- **FR-10.6.2**: 完整报告在 analysis 完成后生成

#### FR-10.7 安全要求

- **FR-10.7.1**: HTML 输出中的 LLM 生成内容必须转义（防止 XSS）
- **FR-10.7.2**: ticker 值必须通过 `safe_ticker_component` 验证后才能用于文件路径

---

### 3.11 配置系统

#### FR-11.1 配置项清单

```text
目录配置:
  results_dir:  string          # 默认 ~/.tradingagents/logs (env: TRADINGAGENTS_RESULTS_DIR)
  data_cache_dir: string        # 默认 ~/.tradingagents/cache (env: TRADINGAGENTS_CACHE_DIR)
  memory_log_path: string       # 默认 ~/.tradingagents/memory/trading_memory.md (env: TRADINGAGENTS_MEMORY_LOG_PATH)

LLM 配置:
  llm_provider: string          # openai|anthropic|google|xai|deepseek|qwen|glm|openrouter|ollama|azure
  deep_think_llm: string        # 深度思考模型 ID
  quick_think_llm: string       # 快速思考模型 ID
  backend_url: string|null      # 自定义 API 端点

提供商特定:
  google_thinking_level: string|null    # high|minimal|low|medium
  openai_reasoning_effort: string|null  # high|medium|low
  anthropic_effort: string|null         # high|medium|low

工作流:
  checkpoint_enabled: bool      # 默认 false
  output_language: string       # 默认 "English"
  max_debate_rounds: int        # 默认 1
  max_risk_discuss_rounds: int  # 默认 1
  max_recur_limit: int          # 默认 100

记忆:
  memory_log_max_entries: int|null  # null = 不轮换

数据源:
  data_vendors: map[string]string    # 类别→供应商
  tool_vendors: map[string]string    # 工具→供应商 (覆盖类别配置)
```

#### FR-11.2 配置加载优先级

```
环境变量 > 配置文件 > 代码默认值
```

#### FR-11.3 环境变量

```
LLM 认证:
  OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
  XAI_API_KEY, DEEPSEEK_API_KEY, DASHSCOPE_API_KEY
  ZHIPU_API_KEY, OPENROUTER_API_KEY

Azure:
  AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT
  AZURE_OPENAI_DEPLOYMENT_NAME, OPENAI_API_VERSION

数据:
  ALPHA_VANTAGE_API_KEY

路径:
  TRADINGAGENTS_CACHE_DIR, TRADINGAGENTS_RESULTS_DIR
  TRADINGAGENTS_MEMORY_LOG_PATH
```

#### FR-11.4 配置安全

- **FR-11.4.1**: API Keys 必须从环境变量或 `.env` 文件加载
- **FR-11.4.2**: API Keys 绝对不能出现在日志/错误消息/报告输出中
- **FR-11.4.3**: 配置中的敏感字段在序列化时必须屏蔽

#### FR-11.5 配置运行时行为

- **FR-11.5.1**: 配置在程序启动时一次性加载
- **FR-11.5.2**: 运行时配置不可变（不需要支持热重载）
- **FR-11.5.3**: 配置必须支持 `Clone`（可能在多线程间共享）

---

## 4. 数据模型

### 4.1 全局状态结构

```
AgentState {
    // === 基础标识 ===
    messages: List<Message>               // 当前消息历史
    company_of_interest: string           // 分析的股票代码（含交易所后缀）
    trade_date: string                    // 分析日期 YYYY-MM-DD
    sender: string                        // 最后发送消息的智能体名称
    past_context: string                  // 从记忆日志提取的历史上下文

    // === 分析师报告 (4个) ===
    market_report: string                 // 市场技术分析报告
    sentiment_report: string              // 社交媒体情绪报告
    news_report: string                   // 新闻分析报告
    fundamentals_report: string           // 基本面分析报告

    // === 投资辩论 ===
    investment_debate_state: InvestDebateState
    investment_plan: string              // 研究经理的结构化输出渲染

    // === 交易 ===
    trader_investment_plan: string        // 交易员的结构化输出渲染

    // === 风险辩论 ===
    risk_debate_state: RiskDebateState
    final_trade_decision: string          // PM 的结构化输出渲染
}

InvestDebateState {
    bull_history: string                  // 牛市研究员的所有论点（累积）
    bear_history: string                  // 熊市研究员的所有论点（累积）
    history: string                       // 双方交替的完整对话历史
    current_response: string              // 最新一条论点
    judge_decision: string                // 研究经理的裁定
    count: int                            // 辩论回合计数
}

RiskDebateState {
    aggressive_history: string            // 激进分析师论点（累积）
    conservative_history: string          // 保守分析师论点（累积）
    neutral_history: string               // 中性分析师论点（累积）
    history: string                       // 三方完整对话历史
    latest_speaker: string                // 最近发言者："Aggressive"|"Conservative"|"Neutral"
    current_aggressive_response: string    // 最新激进论点
    current_conservative_response: string  // 最新保守论点
    current_neutral_response: string       // 最新中性论点
    judge_decision: string                // PM 最终决策
    count: int                            // 辩论回合计数
}
```

### 4.2 消息类型

```
Message = SystemMessage | HumanMessage | AIMessage | ToolMessage

SystemMessage {
    content: string
}

HumanMessage {
    id: UUID
    content: string
}

AIMessage {
    id: UUID
    content: string?                      // None 时表示仅 tool_calls
    tool_calls: List<ToolCall>?
    additional_kwargs: Map<string, any>   // 存储 reasoning_content 等
}

ToolMessage {
    id: UUID
    name: string
    content: string
    tool_call_id: string
}

ToolCall {
    id: string
    type: "function"
    function: FunctionCall
}

FunctionCall {
    name: string
    arguments: string                     // JSON string
}
```

### 4.3 评级枚举

```
PortfolioRating: Buy | Overweight | Hold | Underweight | Sell
TraderAction:    Buy | Hold | Sell
```

### 4.4 工具定义类型

```
ToolDef {
    name: string
    description: string
    parameters: JsonSchema               // JSON Schema 对象
}
```

### 4.5 记忆日志条目

```
MemoryLogEntry {
    date: Date
    ticker: string
    rating: PortfolioRating?
    pending: bool
    raw_return: float?
    alpha_return: float?
    holding_days: int?
    decision: string                      // 完整 Markdown
    reflection: string?                   // None = pending
}
```

---

## 5. 接口契约

### 5.1 图执行引擎

```
interface GraphNode {
    name(): string
    stage(): PipelineStage
    execute(state: &mut AgentState, llm: &LlmPair): Future<Result<()>>
}

interface ConditionRouter {
    route(state: &AgentState, config: &Config): PipelineStage
}

interface GraphExecutor {
    stream(initial_state: AgentState): Stream<AgentState>
    invoke(initial_state: AgentState): Future<Result<AgentState>>
}
```

### 5.2 LLM 客户端

```
interface LlmClient {
    chat(messages: List<Message>): Future<Result<Message>>
    chat_with_tools(messages: List<Message>, tools: List<ToolDef>): Future<Result<Message>>
    chat_structured<T>(messages: List<Message>, schema: JsonSchema): Future<Result<T>>
    validate_model(): bool
    model_name(): string
    provider_name(): string
}

struct LlmResponse {
    content: string?
    tool_calls: List<ToolCall>?
    token_usage: TokenUsage
    finish_reason: string
}

struct TokenUsage {
    input_tokens: int
    output_tokens: int
}
```

### 5.3 数据源

```
interface DataSource {
    get_stock_data(symbol, start_date, end_date): Future<Result<string>>
    get_indicators(symbol, indicator, curr_date, lookback_days): Future<Result<string>>
    get_fundamentals(ticker, curr_date): Future<Result<string>>
    get_balance_sheet(ticker, freq, curr_date): Future<Result<string>>
    get_cashflow(ticker, freq, curr_date): Future<Result<string>>
    get_income_statement(ticker, freq, curr_date): Future<Result<string>>
    get_news(ticker, start_date, end_date): Future<Result<string>>
    get_global_news(curr_date, lookback_days, limit): Future<Result<string>>
    get_insider_transactions(ticker): Future<Result<string>>
}
```

### 5.4 记忆日志

```
interface MemoryLog {
    store_decision(ticker, date, decision: PortfolioDecision): Result<()>
    get_pending_entries(ticker): List<MemoryLogEntry>
    batch_update_with_outcomes(updates: List<OutcomeUpdate>): Result<()>
    get_past_context(ticker, n_same=5, n_cross=3): string
    load_entries(): List<MemoryLogEntry>
}
```

---

## 6. 非功能性需求

### 6.1 性能

| 指标 | 目标 | 测量方法 |
|------|------|----------|
| 冷启动时间 | < 3s | 从程序启动到第一次 LLM 调用的时间 |
| 单次分析延迟 | 取决于 LLM API，本地开销 < 5s | 排除 LLM API 调用和网络 I/O 的 CPU 时间 |
| 内存占用 (空闲) | < 50MB | 程序启动后、开始分析前的 RSS |
| 内存占用 (分析中) | < 200MB | 分析执行期间的峰值 RSS |
| 并发分析 | 支持 ≥ 10 ticker 并行 | 无共享可变状态，独立 AgentState |
| LLM Token 吞吐 | 不增加额外限制 | 不对 API 调用添加不必要的序列化屏障 |

### 6.2 可靠性

| 指标 | 目标 |
|------|------|
| LLM 调用失败处理 | 自动重试 3 次，指数退避 (1s, 2s, 4s) |
| 数据源故障转移 | 主数据源失败自动切换备用 |
| 检查点可靠性 | 节点失败后可恢复，最多丢失 1 个节点的进度 |
| 日志写入可靠性 | 原子文件操作，崩溃不损坏日志 |
| 优雅降级 | 结构化输出不可用时自动回退自由文本 |

### 6.3 可维护性

| 指标 | 目标 |
|------|------|
| 模块耦合度 | 智能体间通过 AgentState 共享数据，无直接依赖 |
| 提示可更新性 | 提示模板修改不需要代码变更 |
| 提供商扩展性 | 新增 LLM 提供商只需实现 LlmClient 接口 |
| 数据源扩展性 | 新增数据供应商只需实现 DataSource 接口 |
| 日志可观测性 | 结构化日志，支持不同级别（DEBUG/INFO/WARN/ERROR） |
| 测试覆盖率 | 核心逻辑 ≥ 80%，解析器/工具 = 100% |

### 6.4 兼容性

| 指标 | 目标 |
|------|------|
| 报告格式兼容 | 生成的 Markdown 报告结构与 Python 版本一致 |
| 记忆日志兼容 | 能读取 Python 版本写入的 `trading_memory.md` |
| 环境变量兼容 | 所有环境变量命名与 Python 版本一致 |
| API 兼容 | （可选）提供与 Python 版 `TradingAgentsGraph` 兼容的编程接口 |

### 6.5 分发性

| 指标 | 目标 |
|------|------|
| 二进制大小 | < 50MB (release, stripped) |
| 平台支持 | Linux (x86_64, aarch64), macOS (x86_64, aarch64), Windows (x86_64) |
| 外部依赖 | 无系统级依赖（SQLite 使用 bundled），不需要 Python 运行时 |
| 安装方式 | 单二进制下载 + 解压即用 |

---

## 7. 安全需求

### 7.1 路径安全

- **SR-1.1**: ticker 值在用于文件系统路径前必须通过 `safe_ticker_component` 验证
- **SR-1.2**: 允许字符集：`[A-Za-z0-9._\-\^]`
- **SR-1.3**: 最大长度限制：32 字符
- **SR-1.4**: 拒绝全点号值（如 `.`, `..`, `...`）
- **SR-1.5**: 拒绝空字符串和非字符串类型

### 7.2 输出安全

- **SR-2.1**: HTML 报告中的 LLM 输出必须 HTML 转义
- **SR-2.2**: Markdown 报告中的 LLM 输出不需要转义（纯文本）
- **SR-2.3**: CLI 终端输出不需要额外转义

### 7.3 敏感信息保护

- **SR-3.1**: API Keys 不出现在任何日志输出中
- **SR-3.2**: API Keys 不出现在错误消息中
- **SR-3.3**: API Keys 不出现在导出的报告中
- **SR-3.4**: 配置 dump 时必须屏蔽 `api_key` 字段
- **SR-3.5**: `.env` 文件不应被提交到版本控制

### 7.4 网络安全

- **SR-4.1**: 所有 LLM API 调用必须使用 HTTPS
- **SR-4.2**: TLS 证书验证必须启用（除非用户显式配置跳过）
- **SR-4.3**: 支持企业代理配置

### 7.5 输入验证

- **SR-5.1**: 所有用户输入在用于 API 调用前进行验证
- **SR-5.2**: 日期格式必须严格校验（YYYY-MM-DD，不能是未来日期）
- **SR-5.3**: ticker 格式校验（拒绝含路径遍历字符的值）
- **SR-5.4**: 工具调用参数必须进行基本类型校验

---

## 8. 测试需求

### 8.1 测试层次

```
Layer 1 — 单元测试:
  - 纯函数: parse_rating, safe_ticker_component, render_*, condition routers
  - 类型转换: Message 序列化/反序列化
  - 配置解析: Config 加载、合并、验证
  - 记忆日志: 条目解析、格式化、轮换逻辑
  - 信号提取: 各种 Markdown 格式变体

Layer 2 — 集成测试:
  - 图执行引擎: 使用 mock LLM + mock 数据源的完整图执行
  - 数据源路由: 供应商选择、回退链
  - 文件 I/O: 记忆日志读写、报告生成
  - 检查点: SQLite 保存/恢复/清除
  - 智能体协作: 多个智能体顺序执行的正确性

Layer 3 — 端到端测试:
  - 完整分析流程 (mock LLM 响应)
  - CLI 参数解析和配置构建
  - 完整分析流程 (真实 LLM，可选，CI 中不强制)
```

### 8.2 Mock 策略

| Mock 对象 | 方式 |
|-----------|------|
| LLM 客户端 | 实现 `MockLlmClient`，返回预定义响应 |
| 数据源 | 实现 `MockDataSource`，从 fixture 文件加载 |
| 文件系统 | 使用 `tempfile` 创建临时目录 |
| 网络 | 使用 mock HTTP server（如 `mockito`/`wiremock`） |
| 时间 | 固定日期 `2026-05-14` |

### 8.3 测试数据

- 至少 5 个完整的 trader agents 输出样本（黄金数据集）
- 至少 10 个评级提取测试用例（覆盖所有格式变体）
- 至少 5 个记忆日志文件样本（含 pending 和 resolved 条目）
- 至少 3 个配置文件样本（覆盖不同提供商和数据源组合）

### 8.4 关键测试用例

| 测试对象 | 最少测试数 | 覆盖场景 |
|----------|----------|----------|
| `parse_rating` | 20 | 所有格式变体 (bold, non-bold, colon, hyphen, case, position) |
| `safe_ticker_component` | 15 | 合法值、路径遍历、特殊字符、长度限制、全点号 |
| 条件路由逻辑 | 12 | 所有三种模式的边界条件 |
| 结构化输出序列化 | 6 | 所有字段的组合（含 None 可选字段） |
| 工具执行循环 | 5 | 0/1/多 次工具调用、工具错误、最大迭代 |
| 记忆日志解析 | 10 | 完整/部分字段、pending/resolved、边界情况 |
| 检查点 | 6 | 保存、恢复、覆盖、清除、多 ticker、空状态 |

---

## 9. 迁移约束

### 9.1 必须保留的行为

- **C-1.1**: 智能体执行顺序和依赖关系
- **C-1.2**: 辩论循环的计数逻辑和终止条件
- **C-1.3**: 结构化输出的 schema 结构和渲染格式
- **C-1.4**: 记忆日志的文件格式和两阶段生命周期
- **C-1.5**: 反思的三要点结构
- **C-1.6**: 评级提取的双通道正则策略
- **C-1.7**: 消息清理 + "Continue" 占位符的行为（Anthropic 兼容）
- **C-1.8**: DeepSeek reasoning_content 回传逻辑

### 9.2 允许的改进

- 提示模板外置（从代码中分离到文件）
- 增加更多 LLM 提供商或新模型
- 改进错误消息的可读性
- 增加细粒度的日志级别
- 改进数据缓存策略
- 增加对新型 LLM 功能（如 prompt caching）的支持

### 9.3 不要求保留的

- Python 特定的语法和模式（装饰器、动态类型等）
- LangChain/LangGraph 的内部实现细节（仅保留外部行为）
- `backtrader` 和 `redis` 依赖（未在核心流程中实际使用）
- PyPI 包管理方式（改用目标语言的包管理）

---

## 10. 附录

### 附录 A: Python 依赖分析

| 依赖 | 是否核心 | 说明 |
|------|---------|------|
| `langgraph` | **核心** | 图编排引擎，需要完整重新实现 |
| `langchain-core` | **核心** | 消息类型和工具定义，需重新实现 |
| `langchain-openai` | **核心** | OpenAI LLM，需重新实现 HTTP 调用 |
| `langchain-anthropic` | **核心** | Anthropic LLM，需重新实现 |
| `langchain-google-genai` | **核心** | Google LLM，需重新实现 |
| `langgraph-checkpoint-sqlite` | **核心** | SQLite 检查点，可用原生 SQLite 库替代 |
| `yfinance` | **核心** | Yahoo Finance 数据，需重新实现 HTTP 调用 |
| `stockstats` | **核心** | 技术指标计算，需找等价库或自行实现 |
| `pandas` | **核心** | CSV 数据处理，可用轻量级 CSV 库替代 |
| `typer` + `rich` | **核心** | CLI 界面，需找等价 CLI 框架 |
| `pydantic` | **核心** | 数据验证/序列化，使用目标语言等价物 |
| `markdown` | 重要 | Markdown → HTML 转换 |
| `backtrader` | 非核心 | 回测框架（代码中未实际使用） |
| `redis` | 非核心 | 缓存（代码中未实际使用） |
| `parsel` | 非核心 | HTML 解析（代码中未实际使用） |

### 附录 B: Python 源文件清单（按功能模块）

```
图编排层 (graph/):
  trading_graph.py    — 主编排器: 初始化、执行、状态日志、记忆日志
  setup.py            — 图构建: 创建节点、添加边、编译图
  conditional_logic.py — 条件路由: 工具循环 + 辩论循环
  propagation.py      — 初始状态创建 + 图参数
  checkpointer.py     — SQLite 检查点: 保存/恢复/清除
  reflection.py       — 事后反思 LLM 调用
  signal_processing.py — 信号提取 (5-tier 评级)

智能体层 (agents/):
  analysts/market_analyst.py        — 市场技术分析师 + 工具循环
  analysts/social_media_analyst.py  — 社交媒体情绪分析师
  analysts/news_analyst.py          — 新闻分析师
  analysts/fundamentals_analyst.py  — 基本面分析师
  researchers/bull_researcher.py    — 牛市研究员 (辩论)
  researchers/bear_researcher.py    — 熊市研究员 (辩论)
  managers/research_manager.py      — 研究经理 (结构化输出)
  trader/trader.py                  — 交易员 (结构化输出)
  risk_mgmt/aggressive_debator.py   — 激进风险分析师
  risk_mgmt/conservative_debator.py — 保守风险分析师
  risk_mgmt/neutral_debator.py      — 中性风险分析师
  managers/portfolio_manager.py     — 投资组合经理 (最终决策)

通用层:
  schemas.py           — Pydantic 结构化输出模式
  utils/agent_states.py — AgentState + 子状态定义
  utils/agent_utils.py  — 工具导入、消息删除、ticker 处理
  utils/memory.py       — 决策记忆日志
  utils/rating.py       — 评级解析 (正则启发式)
  utils/structured.py   — 结构化输出 wrapper + 降级逻辑
  utils/core_stock_tools.py        — 行情工具
  utils/technical_indicators_tools.py — 指标工具
  utils/fundamental_data_tools.py  — 基本面工具
  utils/news_data_tools.py         — 新闻工具

LLM 客户端层 (llm_clients/):
  base_client.py       — 抽象基类 + 内容规范化
  factory.py           — 工厂函数 (延迟导入)
  openai_client.py     — OpenAI + 6 兼容提供商 + DeepSeek
  anthropic_client.py  — Anthropic Claude
  google_client.py     — Google Gemini
  azure_client.py      — Azure OpenAI
  model_catalog.py     — 已知模型列表
  validators.py        — 模型验证

数据源层 (dataflows/):
  interface.py         — 供应商路由 + 回退链
  config.py            — 运行时配置单例
  y_finance.py         — yfinance OHLCV/指标/基本面/财报
  yfinance_news.py     — yfinance 新闻
  stockstats_utils.py  — 技术指标计算 + 缓存
  alpha_vantage_*.py   — Alpha Vantage 实现 (6个文件)
  utils.py             — ticker 安全验证 + 日期工具

CLI 层 (cli/):
  main.py              — Typer 应用 + Rich UI Live 显示
  utils.py             — 交互式提示 + 提供商选择
  models.py            — AnalystType 枚举
  stats_handler.py     — LangChain 回调 (统计收集)
  config.py            — CLI 配置常量
  announcements.py     — 公告获取

测试层 (tests/):
  conftest.py                  — Fixtures (mock API keys, mock LLM)
  test_signal_processing.py    — 信号提取测试
  test_memory_log.py           — 记忆日志测试
  test_structured_agents.py    — 结构化输出智能体测试
  test_checkpoint_resume.py    — 断点续跑测试
  test_deepseek_reasoning.py   — DeepSeek 特性测试
  test_google_api_key.py       — Google API key 测试
  test_model_validation.py     — 模型验证测试
  test_safe_ticker_component.py — 安全 ticker 测试
  test_ticker_symbol_handling.py — ticker 处理测试
```

### 附录 C: 文件大小估算 (Python 源码)

| 模块 | 文件数 | 预估行数 |
|------|--------|----------|
| 图编排层 | 7 | ~600 |
| 智能体 (含 schemas/utils) | 20 | ~1800 |
| LLM 客户端 | 9 | ~800 |
| 数据源 | 14 | ~1500 |
| CLI | 6 | ~1400 |
| 测试 | 10 | ~1500 |
| **总计** | **~66** | **~7600** |

### 附录 D: 术语对照表

| Python 术语 | 通用概念 | 说明 |
|-------------|---------|------|
| LangGraph StateGraph | 有向状态图 | 节点 + 边构成的工作流图 |
| ToolNode | 工具执行器 | 接收 tool_calls 执行对应函数 |
| conditional_edges | 条件路由 | 根据状态值选择下一个节点 |
| MessagesState | 消息列表状态 | 带有累加规则的状态类型 |
| TypedDict | 类型化字典 | 键值对集合，有类型约束 |
| @tool 装饰器 | 工具注册 | 将函数注册为 LLM 可调用的工具 |
| with_structured_output | 结构化输出绑定 | 将 JSON Schema 绑定到 LLM |
| ChatPromptTemplate | 提示模板 | 带变量插值的消息模板 |
| SqliteSaver | SQLite 检查点存储 | 将状态持久化到 SQLite |
| stream_mode="values" | 流式执行模式 | 每节点完成后输出完整状态 |
| recursion_limit | 递归深度限制 | 防止无限循环的保护机制 |
| Phase A / Phase B | 两阶段日志 | 先存储 pending，后解析结果 |

---

> **文档版本**: 1.0
> **最后更新**: 2026-05-14
> **基准代码版本**: TradingAgents v0.2.4
>
> 本文档基于对 TradingAgents 源代码的完整阅读和分析编写，覆盖了全部 ~66 个 Python 源文件的功能需求。
