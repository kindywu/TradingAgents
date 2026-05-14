# TradingAgents Rust 实现设计文档

本文档设计如何用 Rust 重新实现 TradingAgents，保持与 Python 版本完全一致的业务流程，但对 LangChain/LangGraph 这类 Python 生态独有的通用框架做**最小化硬编码替代**——流程写死，必要的抽象保留。

---

## 目录

1. [架构总览](#架构总览)
2. [核心抽象：Rust 里的 "LCEL"](#核心抽象rust-里的-lcel)
3. [LLM 客户端抽象](#llm-客户端抽象)
4. [Tool Calling：Rust 里的工具系统](#tool-callingrust-里的工具系统)
5. [Structured Output：Typed 输出](#structured-outputtyped-输出)
6. [图执行引擎：硬编码的 StateGraph](#图执行引擎硬编码的-stategraph)
7. [Agent 状态定义](#agent-状态定义)
8. [Analyst 节点 + 工具循环](#analyst-节点--工具循环)
9. [辩论循环：Bull vs Bear](#辩论循环bull-vs-bear)
10. [风险管理辩论 + 最终决策](#风险管理辩论--最终决策)
11. [Checkpoint / 断点续跑](#checkpoint-断点续跑)
12. [Memory / 反思系统](#memory-反思系统)
13. [数据供应商抽象](#数据供应商抽象)
14. [配置系统](#配置系统)
15. [完整代码骨架](#完整代码骨架)
16. [Python vs Rust 对照表](#python-vs-rust-对照表)

---

## 架构总览

Python 版依赖 LangGraph 做图调度、LangChain 做 LLM 抽象。Rust 生态没有等价物，但我们也不需要一个通用的图执行引擎——TradingAgents 的流程是**固定的**：

```
START
  → MarketAnalyst (↻ 工具循环)
  → SocialAnalyst  (↻ 工具循环)
  → NewsAnalyst    (↻ 工具循环)
  → FundamentalsAnalyst (↻ 工具循环)
  → BullResearcher ↔ BearResearcher (↻ 辩论, max N 轮)
  → ResearchManager
  → Trader
  → AggressiveRisk ↔ ConservativeRisk ↔ NeutralRisk (↻ 辩论, max N 轮)
  → PortfolioManager
  → END
```

**核心设计决策：**

| 问题 | 决策 |
|------|------|
| 图执行引擎 | **硬编码**。一个 `run_pipeline()` 函数包含完整的顺序逻辑 + 循环。不用 DAG 调度器。 |
| LLM 抽象 | **trait + enum 分发**。`trait LlmClient`，`enum Provider { OpenAI, Anthropic, Google, ... }` |
| 状态管理 | **一个巨大的 `AgentState` struct**，节点函数签名 `(state: &mut AgentState) -> Result<()>` |
| 工具系统 | **`#[async_trait]` + serde_json**。手动解析 `tool_calls`、执行、拼回 messages。 |
| 断点续跑 | **SQLite + serde_json**。每个节点执行完 `save_checkpoint()`，崩溃后 `load_checkpoint()` 从断点继续。 |

> Python 的 LangGraph 是"通用图引擎 + 业务节点"，Rust 版是"硬编码的业务流程 + 抽象 trait"。

---

## 核心抽象：Rust 里的 "LCEL"

LangChain 的 LCEL 本质上就是：`PromptTemplate → LLM → OutputParser`。在 Rust 中我们用 trait 表达。

### 消息类型

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum Message {
    #[serde(rename = "system")]
    System {
        content: String,
    },
    #[serde(rename = "user")]
    Human {
        content: String,
    },
    #[serde(rename = "assistant")]
    AI {
        content: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
    },
    #[serde(rename = "tool")]
    Tool {
        content: String,
        tool_call_id: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String, // "function"
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String, // JSON string
}
```

`Message` enum 对应 LangChain 的 `SystemMessage / HumanMessage / AIMessage / ToolMessage`。用 `serde` 序列化，与 OpenAI API 的 messages 格式完全对齐。这是 Rust 版整个系统的"通用货币"——所有 LLM provider 都读写这个格式。

### Prompt 模板

不需要完整的 Jinja2 引擎。用简单的 `str::replace` 即可：

```rust
pub struct PromptTemplate {
    template: String,
}

impl PromptTemplate {
    pub fn new(template: impl Into<String>) -> Self {
        Self { template: template.into() }
    }

    /// 替换 {var_name} 占位符
    pub fn format(&self, vars: &[(&str, &str)]) -> String {
        let mut result = self.template.clone();
        for (key, value) in vars {
            result = result.replace(&format!("{{{}}}", key), value);
        }
        result
    }
}
```

真正的 TradingAgents 的 prompt 都非常长（几十到几百行），存储在 `prompts/` 目录下作为 `.md` 文件，运行时 `include_str!` 加载。占位符只有少数几个：`{ticker}`, `{date}`, `{reports}`, `{history}`, `{current_response}` 等。

### Pipeline trait

对应 LCEL 的 `|` 管道：

```rust
#[async_trait]
pub trait Pipeline {
    type Input;
    type Output;

    async fn invoke(&self, input: Self::Input) -> Result<Self::Output>;
}

// Prompt → LLM → String
pub struct PromptChain {
    template: PromptTemplate,
    llm: Box<dyn LlmClient>,
}

#[async_trait]
impl Pipeline for PromptChain {
    type Input = Vec<(&'static str, String)>;
    type Output = String;

    async fn invoke(&self, vars: Self::Input) -> Result<String> {
        let prompt = self.template.format(&vars.iter().map(|(k,v)| (*k, v.as_str())).collect::<Vec<_>>());
        let messages = vec![Message::Human { content: prompt }];
        let response = self.llm.chat(&messages).await?;
        Ok(response.content)
    }
}
```

实际代码中，大部分节点都用 `llm.chat(messages)` 直接调用，不需要这个 pipeline trait。只有需要"模板 + LLM + 结构化输出"三件套的节点（Research Manager, Trader, Portfolio Manager）才用到组合。

---

## LLM 客户端抽象

### trait 定义

```rust
use async_trait::async_trait;

#[async_trait]
pub trait LlmClient: Send + Sync {
    /// 发送消息列表，返回 AI 消息
    async fn chat(&self, messages: &[Message]) -> Result<Message>;

    /// 发送消息列表 + 工具定义，返回 AI 消息（可能包含 tool_calls）
    async fn chat_with_tools(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
    ) -> Result<Message>;

    /// 结构化输出（JSON mode / tool_choice）
    async fn chat_structured<T: DeserializeOwned + Send>(
        &self,
        messages: &[Message],
        schema: serde_json::Value,
    ) -> Result<T>;

    /// 检查模型名是否有效
    fn validate_model(&self) -> bool;
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value, // JSON Schema
}
```

### Provider 实现

```rust
pub enum Provider {
    OpenAI,
    Anthropic,
    Google,
    DeepSeek,
    Xai,
    Qwen,
    Glm,
    OpenRouter,
    Ollama,
}

impl Provider {
    pub fn default_base_url(&self) -> &str {
        match self {
            Self::OpenAI => "https://api.openai.com/v1",
            Self::Anthropic => "https://api.anthropic.com",
            Self::DeepSeek => "https://api.deepseek.com/v1",
            // ...
        }
    }

    pub fn api_key_env(&self) -> &str {
        match self {
            Self::OpenAI => "OPENAI_API_KEY",
            Self::Anthropic => "ANTHROPIC_API_KEY",
            Self::DeepSeek => "DEEPSEEK_API_KEY",
            // ...
        }
    }
}
```

### OpenAI-compatible 客户端（最通用的实现）

```rust
use reqwest::Client;
use serde_json::Value;

pub struct OpenAiCompatClient {
    provider: Provider,
    model: String,
    base_url: String,
    api_key: String,
    http: Client,
    extra_params: HashMap<String, Value>,
}

#[async_trait]
impl LlmClient for OpenAiCompatClient {
    async fn chat(&self, messages: &[Message]) -> Result<Message> {
        let body = serde_json::json!({
            "model": self.model,
            "messages": messages,
        });
        let resp = self.http
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;
        let json: Value = resp.json().await?;
        let choice = &json["choices"][0]["message"];
        Ok(Message::AI {
            content: choice["content"].as_str().map(String::from),
            tool_calls: parse_tool_calls(choice),
        })
    }

    async fn chat_with_tools(&self, messages: &[Message], tools: &[ToolDef]) -> Result<Message> {
        let body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "tools": tools.iter().map(|t| serde_json::json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
            })).collect::<Vec<_>>(),
        });
        // ... 同上
    }

    async fn chat_structured<T: DeserializeOwned + Send>(
        &self,
        messages: &[Message],
        schema: serde_json::Value,
    ) -> Result<T> {
        let body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                }
            },
        });
        let resp = self.http.post(...).json(&body).send().await?;
        let text = resp["choices"][0]["message"]["content"].as_str().unwrap();
        Ok(serde_json::from_str(text)?)
    }

    fn validate_model(&self) -> bool {
        // 检查 model 是否在已知列表中
        true // Ollama/OpenRouter 跳过校验
    }
}
```

**Anthropic 客户端**需要适配 Anthropic 的 Messages API（header `x-api-key` 而非 `Authorization: Bearer`，content 是 `[{"type": "text", "text": "..."}]` 数组而非纯字符串），以及对 `tool_use` 格式的 tool_calls 做归一化。

**DeepSeek 客户端**需要处理 `reasoning_content` 字段的回显（think mode），这与 OpenAI-compatible 的 base class 共享大部分逻辑，只在 `parse_response()` 时多做一步保存 reasoning_content。

### 工厂函数

```rust
pub fn create_llm_client(provider: Provider, model: &str, base_url: Option<&str>) -> Box<dyn LlmClient> {
    let base_url = base_url.unwrap_or_else(|| provider.default_base_url()).to_string();
    let api_key = std::env::var(provider.api_key_env()).unwrap_or_default();

    match provider {
        Provider::Anthropic => Box::new(AnthropicClient::new(model, &base_url, &api_key)),
        // OpenAI-compatible 的 8 个 provider 全部走同一个 struct
        _ => Box::new(OpenAiCompatClient::new(provider, model, &base_url, &api_key)),
    }
}
```

### 双 LLM 架构

```rust
pub struct LlmPair {
    pub quick: Box<dyn LlmClient>,  // 分析师、研究员、交易员
    pub deep: Box<dyn LlmClient>,   // 研究经理、投资组合经理
}

impl LlmPair {
    pub fn from_config(config: &Config) -> Self {
        let provider = config.llm_provider.clone();
        Self {
            quick: create_llm_client(provider, &config.quick_think_model, None),
            deep: create_llm_client(provider, &config.deep_think_model, None),
        }
    }
}
```

---

## Tool Calling：Rust 里的工具系统

LangChain 用 `@tool` 装饰器 + 函数签名自动推断 schema。Rust 用 **proc macro** 做不到同样丝滑（无法在编译期读函数的 docstring 作为运行时描述），所以我们用 **声明式注册**：

### 工具定义与注册

```rust
use std::collections::HashMap;
use serde_json::Value;

pub type ToolFn = Arc<dyn Fn(Value) -> Pin<Box<dyn Future<Output = Result<String>> + Send>> + Send + Sync>;

pub struct Tool {
    pub def: ToolDef,
    pub handler: ToolFn,
}

pub struct ToolRegistry {
    tools: HashMap<String, Tool>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: HashMap::new() }
    }

    pub fn register(
        &mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
        handler: impl Fn(Value) -> Pin<Box<dyn Future<Output = Result<String>> + Send>> + Send + Sync + 'static,
    ) {
        let name = name.into();
        self.tools.insert(name.clone(), Tool {
            def: ToolDef {
                name,
                description: description.into(),
                parameters,
            },
            handler: Arc::new(handler),
        });
    }

    pub fn get_defs(&self) -> Vec<ToolDef> {
        self.tools.values().map(|t| t.def.clone()).collect()
    }

    pub async fn execute(&self, name: &str, args: Value) -> Result<String> {
        let tool = self.tools.get(name)
            .ok_or_else(|| anyhow::anyhow!("Unknown tool: {}", name))?;
        (tool.handler)(args).await
    }

    pub fn filter(&self, names: &[&str]) -> Self {
        // 返回只包含指定名称的子 ToolRegistry
        // ...
    }
}
```

### 工具注册示例

```rust
fn register_market_tools(registry: &mut ToolRegistry) {
    registry.register(
        "get_stock_data",
        "获取股票历史 OHLCV 数据",
        serde_json::json!({
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "股票代码，如 AAPL"},
                "start_date": {"type": "string", "description": "开始日期 YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "结束日期 YYYY-MM-DD"},
            },
            "required": ["ticker", "start_date", "end_date"],
        }),
        |args| {
            Box::pin(async move {
                let ticker = args["ticker"].as_str().unwrap();
                let start = args["start_date"].as_str().unwrap();
                let end = args["end_date"].as_str().unwrap();
                get_stock_data(ticker, start, end).await
            })
        },
    );

    registry.register(
        "get_indicators",
        "获取技术指标，包括 RSI, MACD, SMA, Bollinger Bands",
        serde_json::json!({
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "indicators": {"type": "string", "description": "逗号分隔的指标名"},
            },
            "required": ["ticker", "start_date", "end_date", "indicators"],
        }),
        |args| Box::pin(async move { get_indicators(&args).await }),
    );
}
```

### 工具执行循环

对应 LangGraph 的 `ToolNode` + 条件边回到 agent：

```rust
/// 工具循环：Agent 和 ToolNode 之间来回，直到 LLM 不再要求调工具
pub async fn agent_tool_loop(
    llm: &dyn LlmClient,
    tools: &ToolRegistry,
    messages: &mut Vec<Message>,
) -> Result<String> {
    let tool_defs = tools.get_defs();

    loop {
        // Agent 调用 LLM（带工具绑定）
        let response = llm.chat_with_tools(messages, &tool_defs).await?;

        match &response.tool_calls {
            Some(calls) if !calls.is_empty() => {
                // 把 AI 消息加入历史
                messages.push(response.clone());

                // 执行每个工具调用
                for tc in calls {
                    let args: Value = serde_json::from_str(&tc.function.arguments)?;
                    let result = tools.execute(&tc.function.name, args).await?;

                    messages.push(Message::Tool {
                        content: result,
                        tool_call_id: tc.id.clone(),
                    });
                }
                // 循环回到 LLM
            }
            _ => {
                // LLM 给了文本回复，没有工具调用 → 结束循环
                let content = response.content.clone().unwrap_or_default();
                messages.push(response);
                return Ok(content);
            }
        }
    }
}
```

这样，一个 Analyst 节点就是：准备 messages → 调 `agent_tool_loop()` → 得到报告文本。

---

## Structured Output：Typed 输出

Python 版用 Pydantic v2 + `with_structured_output()`。Rust 版用 `serde` + JSON mode：

### Schema 定义

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PortfolioRating {
    Buy,
    Overweight,
    Hold,
    Underweight,
    Sell,
}

// ==================== ResearchPlan ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchPlan {
    pub recommendation: PortfolioRating,
    pub rationale: String,
    pub strategic_actions: String,
}

impl ResearchPlan {
    /// 渲染为 markdown（给下游节点消费）
    pub fn render(&self) -> String {
        format!(
            "**Recommendation**: {}\n\n**Rationale**:\n{}\n\n**Strategic Actions**:\n{}",
            format!("{:?}", self.recommendation),
            self.rationale,
            self.strategic_actions,
        )
    }
}

// ==================== TraderProposal ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TraderAction {
    Buy,
    Hold,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraderProposal {
    pub action: TraderAction,
    pub reasoning: String,
    pub entry_price: Option<f64>,
    pub stop_loss: Option<f64>,
    pub position_sizing: Option<String>,
}

// ==================== PortfolioDecision ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioDecision {
    pub rating: PortfolioRating,
    pub executive_summary: String,
    pub investment_thesis: String,
    pub price_target: Option<f64>,
    pub time_horizon: Option<String>,
}
```

### 结构化输出调用

```rust
/// 尝试结构化输出，失败回退到自由文本
pub async fn invoke_structured_or_freetext<T: DeserializeOwned>(
    structured_llm: &dyn LlmClient,
    plain_llm: &dyn LlmClient,
    messages: &[Message],
    schema_name: &str,
    agent_name: &str,
) -> Result<T> {
    let schema = generate_json_schema::<T>(schema_name);

    match structured_llm.chat_structured::<T>(messages, schema).await {
        Ok(result) => Ok(result),
        Err(e) => {
            log::warn!("{agent_name} structured output failed ({e}), falling back to free-text");
            // fallback: 用 plain_llm 做普通调用，自己解析
            let text_response = plain_llm.chat(messages).await?;
            // 尝试从文本中提取 JSON
            extract_json_from_text::<T>(&text_response.content.unwrap_or_default())
        }
    }
}

/// 生成 JSON Schema（运行时用 serde_json::Value 表示）
fn generate_json_schema<T: Serialize + DeserializeOwned>(name: &str) -> serde_json::Value {
    // 用 schemars crate 或直接手写
    // ...
}
```

---

## 图执行引擎：硬编码的 StateGraph

这是 Rust 版最核心的设计差异。不再有一个通用的 `StateGraph` 框架，所有流程逻辑硬编码在一个函数中。但我们需要结构化的阶段跳转和断点续跑。

### 阶段枚举（用于 checkpoint 路由）

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineStage {
    Init,
    MarketAnalyst,
    SocialAnalyst,
    NewsAnalyst,
    FundamentalsAnalyst,
    BullResearcher,
    BearResearcher,
    ResearchManager,
    Trader,
    AggressiveRisk,
    ConservativeRisk,
    NeutralRisk,
    PortfolioManager,
    Done,
}

impl PipelineStage {
    /// 按顺序的下一个阶段
    pub fn next(self) -> Option<Self> {
        use PipelineStage::*;
        match self {
            Init => Some(MarketAnalyst),
            MarketAnalyst => Some(SocialAnalyst),
            SocialAnalyst => Some(NewsAnalyst),
            NewsAnalyst => Some(FundamentalsAnalyst),
            FundamentalsAnalyst => Some(BullResearcher),
            // 辩论阶段的 next 由循环逻辑控制，不通过此方法
            ResearchManager => Some(Trader),
            Trader => Some(AggressiveRisk),
            PortfolioManager => Some(Done),
            Done => None,
            _ => None,
        }
    }
}
```

### 节点 trait

```rust
#[async_trait]
pub trait GraphNode: Send + Sync {
    /// 节点名（用于日志和 checkpoint）
    fn name(&self) -> &str;

    /// 节点对应的阶段
    fn stage(&self) -> PipelineStage;

    /// 执行节点逻辑，返回（）或错误
    async fn execute(&self, state: &mut AgentState, llm: &LlmPair) -> Result<()>;
}
```

### 工具循环节点的通用实现

所有四个 Analyst 节点的逻辑是完全一样的：`构造 system prompt → 进入 tool loop → 存储报告`。不同之处仅在于 system prompt 模板和工具集。

```rust
pub struct AnalystNode {
    stage: PipelineStage,
    name: String,
    system_prompt_template: &'static str,
    tools: ToolRegistry,
    report_field: ReportField, // 标记写入 state 的哪个字段
}

#[async_trait]
impl GraphNode for AnalystNode {
    fn name(&self) -> &str { &self.name }
    fn stage(&self) -> PipelineStage { self.stage }

    async fn execute(&self, state: &mut AgentState, llm: &LlmPair) -> Result<()> {
        let prompt = self.system_prompt_template
            .replace("{ticker}", &state.company_of_interest)
            .replace("{date}", &state.trade_date);

        let mut messages = vec![
            Message::System { content: prompt },
            Message::Human {
                content: format!("分析 {} 在 {} 的情况", state.company_of_interest, state.trade_date),
            },
        ];

        // 工具循环
        let report = agent_tool_loop(&*llm.quick, &self.tools, &mut messages).await?;

        // 写入 state
        self.report_field.write(state, report);

        // 清空 messages（为了 Anthropic 兼容性，避免重复发送工具结果）
        state.messages = vec![Message::Human { content: "Continue".into() }];

        Ok(())
    }
}
```

### 流水线主函数

```rust
pub async fn run_pipeline(
    state: &mut AgentState,
    llm: &LlmPair,
    config: &Config,
    checkpointer: &Option<Checkpointer>,
) -> Result<PortfolioDecision> {
    // 如果是断点续跑，从 checkpoint 恢复 stage
    let start_stage = if let Some(cp) = checkpointer {
        cp.get_current_stage().unwrap_or(PipelineStage::Init)
    } else {
        PipelineStage::Init
    };

    let nodes: Vec<Box<dyn GraphNode>> = build_nodes(config);

    let mut stage = start_stage;
    while stage != PipelineStage::Done {
        // 找到对应节点
        let node = nodes.iter().find(|n| n.stage() == stage)
            .ok_or_else(|| anyhow::anyhow!("No node for stage {:?}", stage))?;

        println!("[{}] Running...", node.name());

        // 执行节点
        node.execute(state, llm).await?;

        // 保存 checkpoint
        if let Some(cp) = checkpointer {
            cp.save(stage, state).await?;
        }

        // 推进 stage
        stage = next_stage(stage, state, config);
    }

    // 解析最终决策
    let decision: PortfolioDecision = serde_json::from_str(&state.final_trade_decision)?;
    Ok(decision)
}

/// 决定下一个阶段（包含辩论循环的条件逻辑）
fn next_stage(current: PipelineStage, state: &AgentState, config: &Config) -> PipelineStage {
    use PipelineStage::*;

    match current {
        // Analyst 链：跳过未选中的 analyst
        MarketAnalyst if !config.selected_analysts.contains(&"market") => SocialAnalyst,
        SocialAnalyst if !config.selected_analysts.contains(&"social") => NewsAnalyst,
        NewsAnalyst if !config.selected_analysts.contains(&"news") => FundamentalsAnalyst,
        FundamentalsAnalyst if !config.selected_analysts.contains(&"fundamentals") => BullResearcher,

        // 投资辩论循环
        BullResearcher => {
            let debate = &state.investment_debate_state;
            if debate.count >= 2 * config.max_debate_rounds as i32 {
                ResearchManager
            } else {
                BearResearcher  // Bear 回应
            }
        }
        BearResearcher => {
            let debate = &state.investment_debate_state;
            if debate.count >= 2 * config.max_debate_rounds as i32 {
                ResearchManager
            } else {
                BullResearcher  // Bull 回应
            }
        }

        // 风险辩论循环（三角循环）
        AggressiveRisk => {
            let debate = &state.risk_debate_state;
            if debate.count >= 3 * config.max_risk_discuss_rounds as i32 {
                PortfolioManager
            } else {
                ConservativeRisk
            }
        }
        ConservativeRisk => {
            let debate = &state.risk_debate_state;
            if debate.count >= 3 * config.max_risk_discuss_rounds as i32 {
                PortfolioManager
            } else {
                NeutralRisk
            }
        }
        NeutralRisk => {
            let debate = &state.risk_debate_state;
            if debate.count >= 3 * config.max_risk_discuss_rounds as i32 {
                PortfolioManager
            } else {
                AggressiveRisk
            }
        }

        // 其他阶段线性递进
        current => current.next().unwrap_or(Done),
    }
}
```

---

## Agent 状态定义

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    // --- 基础信息 ---
    pub company_of_interest: String,
    pub trade_date: String,
    pub sender: String,
    pub messages: Vec<Message>,

    // --- 分析师报告 ---
    pub market_report: String,
    pub sentiment_report: String,
    pub news_report: String,
    pub fundamentals_report: String,

    // --- 投资辩论 ---
    pub investment_debate_state: InvestDebateState,
    pub investment_plan: String,

    // --- 交易计划 ---
    pub trader_investment_plan: String,

    // --- 风险辩论 ---
    pub risk_debate_state: RiskDebateState,
    pub final_trade_decision: String,

    // --- 记忆 ---
    pub past_context: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InvestDebateState {
    pub bull_history: String,
    pub bear_history: String,
    pub history: String,
    pub current_response: String,
    pub judge_decision: String,
    pub count: i32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RiskDebateState {
    pub aggressive_history: String,
    pub conservative_history: String,
    pub neutral_history: String,
    pub history: String,
    pub latest_speaker: String,
    pub current_aggressive_response: String,
    pub current_conservative_response: String,
    pub current_neutral_response: String,
    pub judge_decision: String,
    pub count: i32,
}
```

Python 版 `AgentState` 继承 `MessagesState` 以获得特殊的消息累加 reducer。Rust 版不需要 reducer——因为每个 analyst 执行完毕后我们直接 `state.messages = vec![HumanMessage("Continue")]` 清空消息列表。辩论节点手动 append 到 `history` 字段而非依赖 messages 累加。

---

## Analyst 节点 + 工具循环

四个分析师节点的构建函数：

```rust
fn build_analyst_nodes(config: &Config, data_vendor: &DataVendor) -> Vec<Box<dyn GraphNode>> {
    let mut nodes: Vec<Box<dyn GraphNode>> = Vec::new();

    for analyst_type in &config.selected_analysts {
        let (stage, name, prompt, tools) = match analyst_type.as_str() {
            "market" => (
                PipelineStage::MarketAnalyst,
                "Market Analyst",
                include_str!("../prompts/market_analyst.md"),
                create_market_tools(data_vendor),
            ),
            "social" => (
                PipelineStage::SocialAnalyst,
                "Social Media Analyst",
                include_str!("../prompts/social_media_analyst.md"),
                create_social_tools(data_vendor),
            ),
            "news" => (
                PipelineStage::NewsAnalyst,
                "News Analyst",
                include_str!("../prompts/news_analyst.md"),
                create_news_tools(data_vendor),
            ),
            "fundamentals" => (
                PipelineStage::FundamentalsAnalyst,
                "Fundamentals Analyst",
                include_str!("../prompts/fundamentals_analyst.md"),
                create_fundamentals_tools(data_vendor),
            ),
            _ => continue,
        };

        nodes.push(Box::new(AnalystNode::new(stage, name, prompt, tools, analyst_type)));
    }

    nodes
}
```

每个 analyst 的 `execute()` 流程（前面已展示）完全一致，差异仅在于 prompt 和 tool set。这比 Python 版更简洁——Python 版为了 LangGraph 的节点函数签名，每个 analyst 都是一个独立文件。

---

## 辩论循环：Bull vs Bear

### Bull Researcher

```rust
pub struct BullResearcherNode;

#[async_trait]
impl GraphNode for BullResearcherNode {
    fn name(&self) -> &str { "Bull Researcher" }
    fn stage(&self) -> PipelineStage { PipelineStage::BullResearcher }

    async fn execute(&self, state: &mut AgentState, llm: &LlmPair) -> Result<()> {
        let debate = &state.investment_debate_state;

        // 构造 prompt：包含所有分析师报告、历史辩论记录、最后一条发言
        let prompt = BullResearcherPrompt::build(
            &state.market_report,
            &state.sentiment_report,
            &state.news_report,
            &state.fundamentals_report,
            &debate.history,
            &debate.current_response,
        );

        let messages = vec![
            Message::System { content: prompt },
            Message::Human { content: "Present your bullish argument.".into() },
        ];

        let response = llm.quick.chat(&messages).await?;
        let content = response.content.unwrap_or_default();
        let bull_argument = format!("Bull Analyst:\n{}\n---", content);

        // 更新 state
        let mut new_debate = debate.clone();
        new_debate.bull_history = format!("{}{}", debate.bull_history, bull_argument);
        new_debate.history = format!("{}{}", debate.history, bull_argument);
        new_debate.current_response = bull_argument;
        new_debate.count += 1;

        state.investment_debate_state = new_debate;
        state.sender = "Bull Researcher".into();

        Ok(())
    }
}
```

Bear Researcher 结构完全对称，只是把 `bull_history` 换成 `bear_history`，前缀换成 `"Bear Analyst:"`，`current_response` 使用正确的前缀。

### Research Manager（裁判）

```rust
pub struct ResearchManagerNode;

#[async_trait]
impl GraphNode for ResearchManagerNode {
    fn name(&self) -> &str { "Research Manager" }
    fn stage(&self) -> PipelineStage { PipelineStage::ResearchManager }

    async fn execute(&self, state: &mut AgentState, llm: &LlmPair) -> Result<()> {
        let debate = &state.investment_debate_state;

        let prompt = ResearchManagerPrompt::build(
            &state.company_of_interest,
            &state.trade_date,
            &state.market_report,
            &state.sentiment_report,
            &state.news_report,
            &state.fundamentals_report,
            &debate.history,
        );

        let messages = vec![Message::System { content: prompt }];

        // 结构化输出
        let plan: ResearchPlan = invoke_structured_or_freetext(
            &*llm.deep, &*llm.deep, &messages, "research_plan", "ResearchManager",
        ).await?;

        state.investment_plan = plan.render();
        let mut new_debate = debate.clone();
        new_debate.judge_decision = plan.render();
        state.investment_debate_state = new_debate;
        state.sender = "Research Manager".into();

        Ok(())
    }
}
```

---

## 风险管理辩论 + 最终决策

### 风险辩论三角循环

三个风险分析师（AggressiveRisk、ConservativeRisk、NeutralRisk）结构完全对称，与 Bull/Bear 辩论类似：

- 每个收到：四个分析师报告 + `trader_investment_plan` + `history` + **最近两个对手的发言**
- 每个追加自己的论点到 `history` 和自己的 `*_history` 字段
- `count` 递增，`latest_speaker` 更新

```rust
pub struct AggressiveRiskNode;

#[async_trait]
impl GraphNode for AggressiveRiskNode {
    fn name(&self) -> &str { "Aggressive Risk Analyst" }
    fn stage(&self) -> PipelineStage { PipelineStage::AggressiveRisk }

    async fn execute(&self, state: &mut AgentState, llm: &LlmPair) -> Result<()> {
        let risk = &state.risk_debate_state;

        let prompt = RiskAnalystPrompt::build(
            "aggressive",
            &state.market_report,
            &state.sentiment_report,
            &state.news_report,
            &state.fundamentals_report,
            &state.trader_investment_plan,
            &risk.history,
            &risk.current_conservative_response,
            &risk.current_neutral_response,
        );

        let messages = vec![
            Message::System { content: prompt },
            Message::Human { content: "Present your aggressive risk argument.".into() },
        ];

        let response = llm.quick.chat(&messages).await?;
        let content = response.content.unwrap_or_default();
        let argument = format!("Aggressive Risk Analyst:\n{}\n---", content);

        let mut new_risk = risk.clone();
        new_risk.aggressive_history = format!("{}{}", risk.aggressive_history, argument);
        new_risk.history = format!("{}{}", risk.history, argument);
        new_risk.current_aggressive_response = argument;
        new_risk.latest_speaker = "Aggressive".into();
        new_risk.count += 1;

        state.risk_debate_state = new_risk;
        state.sender = "Aggressive Risk Analyst".into();

        Ok(())
    }
}
```

### Portfolio Manager（最终决策者）

```rust
pub struct PortfolioManagerNode;

#[async_trait]
impl GraphNode for PortfolioManagerNode {
    fn name(&self) -> &str { "Portfolio Manager" }
    fn stage(&self) -> PipelineStage { PipelineStage::PortfolioManager }

    async fn execute(&self, state: &mut AgentState, llm: &LlmPair) -> Result<()> {
        let risk = &state.risk_debate_state;

        let prompt = PortfolioManagerPrompt::build(
            &state.company_of_interest,
            &state.trade_date,
            &state.market_report,
            &state.sentiment_report,
            &state.news_report,
            &state.fundamentals_report,
            &state.investment_plan,
            &state.trader_investment_plan,
            &risk.history,
            &state.past_context,   // 注入历史交易记忆！
        );

        let messages = vec![Message::System { content: prompt }];

        // 结构化输出
        let decision: PortfolioDecision = invoke_structured_or_freetext(
            &*llm.deep, &*llm.deep, &messages, "portfolio_decision", "PortfolioManager",
        ).await?;

        state.final_trade_decision = decision.render();
        let mut new_risk = risk.clone();
        new_risk.judge_decision = decision.render();
        state.risk_debate_state = new_risk;
        state.sender = "Portfolio Manager".into();

        Ok(())
    }
}
```

### 信号提取

Portfolio Manager 输出 markdown 后，需要从中提取 5 级评级（Buy/Overweight/Hold/Underweight/Sell）。Python 版用正则启发式：

```rust
/// 从 markdown 文本中提取 PortfolioRating（启发式正则匹配）
pub fn extract_rating(text: &str) -> Option<PortfolioRating> {
    // 1. 先找 "**Rating**: Buy" 这样的结构化字段
    let re = Regex::new(r"\*\*Rating\*\*:\s*(\w+)").unwrap();
    if let Some(caps) = re.captures(text) {
        return PortfolioRating::from_str(&caps[1]).ok();
    }

    // 2. 找不到就用全文关键词搜索
    let text_lower = text.to_lowercase();
    if text_lower.contains("buy") { return Some(PortfolioRating::Buy); }
    if text_lower.contains("overweight") { return Some(PortfolioRating::Overweight); }
    if text_lower.contains("sell") { return Some(PortfolioRating::Sell); }
    if text_lower.contains("underweight") { return Some(PortfolioRating::Underweight); }
    if text_lower.contains("hold") { return Some(PortfolioRating::Hold); }

    None
}
```

---

## Checkpoint / 断点续跑

Python 版依赖 LangGraph 的 `SqliteSaver` 在每个节点执行完毕后自动保存。Rust 版需要手动实现——但逻辑非常简单。

### 设计

```rust
use rusqlite::Connection;

pub struct Checkpointer {
    db: Connection,
    thread_id: String,
}

impl Checkpointer {
    /// 打开/创建 checkpoint 数据库
    pub fn new(db_path: &str, ticker: &str, date: &str) -> Result<Self> {
        let db = Connection::open(db_path)?;

        db.execute_batch("
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                state_json TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (thread_id)
            );
        ")?;

        let thread_id = generate_thread_id(ticker, date);

        Ok(Self { db, thread_id })
    }

    /// 生成确定性 thread_id（SHA256 前16位）
    fn thread_id(&self) -> &str { &self.thread_id }

    /// 保存当前状态
    pub async fn save(&self, stage: PipelineStage, state: &AgentState) -> Result<()> {
        let json = serde_json::to_string(state)?;
        self.db.execute(
            "INSERT OR REPLACE INTO checkpoints (thread_id, stage, state_json) VALUES (?1, ?2, ?3)",
            rusqlite::params![self.thread_id, serde_json::to_string(&stage)?, json],
        )?;
        Ok(())
    }

    /// 获取当前阶段（用于断点续跑）
    pub fn get_current_stage(&self) -> Option<PipelineStage> {
        self.db.query_row(
            "SELECT stage FROM checkpoints WHERE thread_id = ?1",
            rusqlite::params![self.thread_id],
            |row| {
                let s: String = row.get(0)?;
                Ok(serde_json::from_str(&s).unwrap())
            },
        ).ok()
    }

    /// 加载保存的状态
    pub fn load_state(&self) -> Option<AgentState> {
        self.db.query_row(
            "SELECT state_json FROM checkpoints WHERE thread_id = ?1",
            rusqlite::params![self.thread_id],
            |row| {
                let json: String = row.get(0)?;
                Ok(serde_json::from_str(&json).unwrap())
            },
        ).ok()
    }

    /// 是否存在 checkpoint（即是否可以从断点续跑）
    pub fn has_checkpoint(&self) -> bool {
        self.get_current_stage().is_some()
    }

    /// 运行完成后删除 checkpoint
    pub fn clear(&self) -> Result<()> {
        self.db.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?1",
            rusqlite::params![self.thread_id],
        )?;
        Ok(())
    }
}
```

### 使用方式

```rust
pub async fn propagate(
    ticker: &str,
    date: &str,
    config: &Config,
) -> Result<(AgentState, PortfolioRating)> {
    let llm = LlmPair::from_config(config);

    let checkpointer = if config.checkpoint_enabled {
        let path = format!("{}/checkpoints/{}.db", config.data_cache_dir, ticker);
        Some(Checkpointer::new(&path, ticker, date)?)
    } else {
        None
    };

    // 尝试从 checkpoint 恢复
    let mut state = if let Some(ref cp) = checkpointer {
        if cp.has_checkpoint() {
            cp.load_state().unwrap_or_else(|| Propagator::initial_state(ticker, date, config))
        } else {
            Propagator::initial_state(ticker, date, config)
        }
    } else {
        Propagator::initial_state(ticker, date, config)
    };

    // 运行流水线
    let decision = run_pipeline(&mut state, &llm, config, &checkpointer).await?;

    // 成功后清除 checkpoint
    if let Some(cp) = &checkpointer {
        cp.clear()?;
    }

    // 存储决策到 Memory Log
    MemoryLog::store_decision(ticker, date, &decision)?;

    Ok((state, decision.rating))
}
```

### 与 Python 版的关键差异

Python 版的 checkpoint 由 LangGraph 在每个节点执行完后**自动触发**。Rust 版需要我们在 `run_pipeline()` 中每个节点后**手动调用** `cp.save()`。但好在流程是硬编码的，所以不会遗漏。

对于 `interrupt_before`（人工审核暂停）功能，可以加一个需要在某阶段前暂停的 flag：

```rust
pub async fn run_pipeline(
    state: &mut AgentState,
    llm: &LlmPair,
    config: &Config,
    checkpointer: &Option<Checkpointer>,
) -> Result<PortfolioDecision> {
    let start_stage = /* 恢复逻辑 */;
    let nodes = build_nodes(config);

    let mut stage = start_stage;
    while stage != PipelineStage::Done {
        // 如果配置了 interrupt_before 且当前阶段在列表中，暂停并返回
        if config.interrupt_before.contains(&stage) {
            log::info!("Interrupted before {:?}. Resume by calling propagate() again.", stage);
            return Err(anyhow::anyhow!("Interrupted before {:?}", stage));
        }

        let node = nodes.iter().find(|n| n.stage() == stage).unwrap();
        node.execute(state, llm).await?;

        if let Some(cp) = checkpointer {
            cp.save(stage, state).await?;
        }

        stage = next_stage(stage, state, config);
    }

    // ...
}
```

这样，调用方在 `propagate()` 返回 `Interrupted` 错误后，可以人工检查 state 后再调用一次 `propagate()`——checkpoint 机制保证它从断点继续。

---

## Memory / 反思系统

### 数据结构

```rust
use chrono::NaiveDate;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub date: NaiveDate,
    pub ticker: String,
    pub rating: Option<PortfolioRating>,   // None = pending
    pub raw_return: Option<f64>,
    pub alpha_return: Option<f64>,
    pub holding_days: Option<i32>,
    pub decision: String,                   // PortfolioDecision 的 markdown
    pub reflection: Option<String>,         // None = pending
}

pub struct MemoryLog {
    path: PathBuf,
}
```

### 核心逻辑

```rust
impl MemoryLog {
    /// 追加一条 pending 决策
    pub fn store_decision(
        ticker: &str,
        date: &str,
        decision: &PortfolioDecision,
    ) -> Result<()> {
        let mut log = Self::load()?;

        // 幂等性：如果已有同 ticker+date 的 pending 条目，跳过
        let already_exists = log.iter().any(|e|
            e.ticker == ticker && e.date.to_string() == date && e.rating.is_none()
        );
        if already_exists {
            return Ok(());
        }

        log.push(MemoryEntry {
            date: NaiveDate::parse_from_str(date, "%Y-%m-%d")?,
            ticker: ticker.to_string(),
            rating: None,
            raw_return: None,
            alpha_return: None,
            holding_days: None,
            decision: decision.render(),
            reflection: None,
        });

        Self::save(&log)?;
        Ok(())
    }

    /// 解析 pending 条目——获取真实收益，生成反思
    pub async fn resolve_pending(ticker: &str, reflector: &Reflector, llm: &dyn LlmClient) -> Result<()> {
        let mut log = Self::load()?;

        for entry in log.iter_mut().filter(|e| e.ticker == ticker && e.rating.is_none()) {
            // 获取实际收益（用 yfinance 或其他数据源）
            let (raw_return, alpha_return) = fetch_actual_returns(
                &entry.ticker,
                &entry.date.to_string(),
                entry.holding_days.unwrap_or(30),
            ).await?;

            // 生成反思
            let reflection = reflector.reflect(
                &entry.decision,
                raw_return,
                alpha_return,
                &entry.date.to_string(),
                llm,
            ).await?;

            entry.raw_return = Some(raw_return);
            entry.alpha_return = Some(alpha_return);
            entry.reflection = Some(reflection);
        }

        Self::save(&log)?;
        Ok(())
    }

    /// 获取过往上下文（注入到 PortfolioManager prompt）
    pub fn get_past_context(
        ticker: &str,
        n_same: usize,   // 最多 5 条同 ticker
        n_cross: usize,  // 最多 3 条跨 ticker
    ) -> Result<String> {
        let log = Self::load()?;

        let resolved: Vec<&MemoryEntry> = log.iter()
            .filter(|e| e.rating.is_some() && e.reflection.is_some())
            .collect();

        let same_ticker: Vec<_> = resolved.iter()
            .filter(|e| e.ticker == ticker)
            .take(n_same)
            .collect();

        let cross_ticker: Vec<_> = resolved.iter()
            .filter(|e| e.ticker != ticker)
            .take(n_cross)
            .collect();

        let mut ctx = String::new();

        for e in &same_ticker {
            ctx.push_str(&format!(
                "[{} | {} | {} | {:.2}% | {:.2}% | {}d]\n\nDECISION:\n{}\n\nREFLECTION:\n{}\n\n---\n\n",
                e.date, e.ticker,
                e.rating.as_ref().map(|r| format!("{:?}", r)).unwrap_or_default(),
                e.raw_return.unwrap_or(0.0) * 100.0,
                e.alpha_return.unwrap_or(0.0) * 100.0,
                e.holding_days.unwrap_or(0),
                e.decision,
                e.reflection.as_deref().unwrap_or(""),
            ));
        }

        for e in &cross_ticker {
            ctx.push_str(&format!(
                "[{} | {} | {} | {:.2}%]\n\nREFLECTION:\n{}\n\n---\n\n",
                e.date, e.ticker,
                e.rating.as_ref().map(|r| format!("{:?}", r)).unwrap_or_default(),
                e.alpha_return.unwrap_or(0.0) * 100.0,
                e.reflection.as_deref().unwrap_or(""),
            ));
        }

        Ok(ctx)
    }
}
```

### Reflector

```rust
pub struct Reflector;

impl Reflector {
    pub async fn reflect(
        &self,
        decision: &str,
        raw_return: f64,
        alpha_return: f64,
        date: &str,
        llm: &dyn LlmClient,
    ) -> Result<String> {
        let prompt = format!(
            "You are a trading performance reviewer.\n\n\
             Decision on {}:\n{}\n\n\
             Raw return: {:.2}%, Alpha vs SPY: {:.2}%\n\n\
             Write a concise 2-4 sentence reflection on:\n\
             1. Was the directional call correct?\n\
             2. What held or failed in the investment thesis?\n\
             3. One concrete lesson for next time.\n\n\
             Reflection:",
            date, decision, raw_return * 100.0, alpha_return * 100.0,
        );

        let messages = vec![Message::Human { content: prompt }];
        let response = llm.chat(&messages).await?;
        Ok(response.content.unwrap_or_default())
    }
}
```

---

## 数据供应商抽象

Python 版支持 yfinance 和 Alpha Vantage，带 fallback chain。Rust 版简化——优先用 yfinance（通过 HTTP 调用 Yahoo Finance API 或用 `yahoo_finance` crate），也可配 Alpha Vantage。

```rust
#[async_trait]
pub trait DataVendor: Send + Sync {
    async fn get_stock_data(&self, ticker: &str, start: &str, end: &str) -> Result<String>;
    async fn get_indicators(&self, ticker: &str, start: &str, end: &str, indicators: &str) -> Result<String>;
    async fn get_fundamentals(&self, ticker: &str) -> Result<String>;
    async fn get_balance_sheet(&self, ticker: &str) -> Result<String>;
    async fn get_cashflow(&self, ticker: &str) -> Result<String>;
    async fn get_income_statement(&self, ticker: &str) -> Result<String>;
    async fn get_news(&self, ticker: &str, start: &str, end: &str) -> Result<String>;
    async fn get_global_news(&self, start: &str, end: &str) -> Result<String>;
    async fn get_insider_transactions(&self, ticker: &str) -> Result<String>;
}

pub struct FallbackDataVendor {
    primary: Box<dyn DataVendor>,
    fallback: Box<dyn DataVendor>,
}

#[async_trait]
impl DataVendor for FallbackDataVendor {
    async fn get_stock_data(&self, ticker: &str, start: &str, end: &str) -> Result<String> {
        match self.primary.get_stock_data(ticker, start, end).await {
            Ok(data) => Ok(data),
            Err(e) => {
                log::warn!("Primary vendor failed ({e}), falling back");
                self.fallback.get_stock_data(ticker, start, end).await
            }
        }
    }
    // ... 其他方法同理
}
```

实际实现推荐直接调 Yahoo Finance 的 HTTP API（它是公开的），不需要额外的 Rust crate：
- 历史数据：`https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start}&period2={end}&interval=1d`
- 财报：`https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{ticker}?type=annualIncomeStatementHistory`

---

## 配置系统

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    // LLM
    pub llm_provider: Provider,
    pub quick_think_model: String,
    pub deep_think_model: String,

    // 分析师选择
    pub selected_analysts: Vec<String>,

    // 辩论轮数
    pub max_debate_rounds: u32,
    pub max_risk_discuss_rounds: u32,

    // 工具/数据
    pub data_vendor: String,          // "yfinance" | "alpha_vantage"
    pub fallback_vendor: Option<String>,

    // Checkpoint
    pub checkpoint_enabled: bool,
    pub data_cache_dir: String,

    // 输出
    pub output_language: String,
    pub results_dir: String,

    // 记忆
    pub memory_log_enabled: bool,
    pub memory_log_path: String,

    // 人工审核
    pub interrupt_before: Vec<PipelineStage>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            llm_provider: Provider::OpenAI,
            quick_think_model: "gpt-4o".into(),
            deep_think_model: "gpt-4o".into(),
            selected_analysts: vec!["market".into(), "social".into(), "news".into(), "fundamentals".into()],
            max_debate_rounds: 1,
            max_risk_discuss_rounds: 1,
            data_vendor: "yfinance".into(),
            fallback_vendor: None,
            checkpoint_enabled: false,
            data_cache_dir: "~/.tradingagents/cache".into(),
            output_language: "中文".into(),
            results_dir: "~/.tradingagents/logs".into(),
            memory_log_enabled: true,
            memory_log_path: "~/.tradingagents/memory/trading_memory.md".into(),
            interrupt_before: vec![],
        }
    }
}
```

---

## 完整代码骨架

```
tradingagents-rs/
├── Cargo.toml
├── src/
│   ├── main.rs                  # CLI 入口（clap）
│   ├── config.rs                # 配置加载（toml/env）
│   ├── pipeline.rs              # 硬编码的 run_pipeline() + next_stage()
│   ├── state.rs                 # AgentState + 子状态 struct
│   ├── message.rs               # Message enum + ToolCall + ToolDef
│   ├── checkpoint.rs            # Checkpointer（SQLite）
│   ├── memory.rs                # MemoryLog + MemoryEntry
│   ├── reflection.rs            # Reflector
│   ├── signal.rs                # extract_rating() 评级提取
│   ├── llm/
│   │   ├── mod.rs               # trait LlmClient + create_llm_client() factory
│   │   ├── openai_compat.rs     # OpenAI-compatible client（8 个 provider 共用）
│   │   ├── anthropic.rs         # Anthropic client
│   │   └── google.rs            # Google Gemini client
│   ├── tools/
│   │   ├── mod.rs               # Tool + ToolRegistry
│   │   ├── market.rs            # 注册市场数据工具
│   │   ├── social.rs            # 注册社交媒体工具
│   │   ├── news.rs              # 注册新闻工具
│   │   └── fundamentals.rs      # 注册基本面工具
│   ├── agents/
│   │   ├── mod.rs               # trait GraphNode
│   │   ├── analyst.rs           # AnalystNode（四个分析师共用）
│   │   ├── bull_researcher.rs   # BullResearcherNode
│   │   ├── bear_researcher.rs   # BearResearcherNode
│   │   ├── research_manager.rs  # ResearchManagerNode
│   │   ├── trader.rs            # TraderNode
│   │   ├── aggressive_risk.rs   # AggressiveRiskNode
│   │   ├── conservative_risk.rs # ConservativeRiskNode
│   │   ├── neutral_risk.rs      # NeutralRiskNode
│   │   └── portfolio_manager.rs # PortfolioManagerNode
│   ├── data/
│   │   ├── mod.rs               # trait DataVendor
│   │   ├── yahoo.rs             # Yahoo Finance HTTP 调用
│   │   └── alphavantage.rs      # Alpha Vantage API
│   ├── schemas/
│   │   └── mod.rs               # ResearchPlan, TraderProposal, PortfolioDecision
│   └── prompts/                 # include_str!() 加载的 prompt 模板
│       ├── market_analyst.md
│       ├── social_media_analyst.md
│       ├── news_analyst.md
│       ├── fundamentals_analyst.md
│       ├── bull_researcher.md
│       ├── bear_researcher.md
│       ├── research_manager.md
│       ├── trader.md
│       ├── aggressive_risk.md
│       ├── conservative_risk.md
│       ├── neutral_risk.md
│       └── portfolio_manager.md
└── tests/
    ├── pipeline_test.rs
    ├── checkpoint_test.rs
    ├── memory_test.rs
    └── signal_test.rs
```

---

## Python vs Rust 对照表

| 概念 | Python (LangChain/LangGraph) | Rust (本设计) |
|------|---------------------------|--------------|
| **LLM 调用** | `ChatOpenAI.invoke(messages)` | `llm.chat(&messages).await` |
| **多 provider 支持** | LangChain 的 `Chat*` 类 | `trait LlmClient` + `create_llm_client()` factory |
| **工具定义** | `@tool` 装饰器，自动推断 schema | `ToolRegistry::register()` 声明式注册 |
| **工具循环** | LangGraph ToolNode + 条件边 | `agent_tool_loop()` 普通 while 循环 |
| **结构化输出** | `llm.with_structured_output(PydanticModel)` | `llm.chat_structured::<T>(schema)` + JSON mode |
| **状态管理** | LangGraph `MessagesState` + reducer | `AgentState` struct, 节点函数 `&mut state` |
| **图调度** | LangGraph `StateGraph` 自动调度 | `run_pipeline()` + `next_stage()` 硬编码流程 |
| **条件边** | `add_conditional_edges("node", router, mapping)` | `next_stage()` 中的 match + if 判断 |
| **辩论循环** | 条件边自动 B 跳 | `next_stage()` 中 `count >= 2*max_rounds` 判断 |
| **并行执行** | LangGraph 从 START 分叉自动并行 | 需要手动 `tokio::join!`（暂不需要，当前是顺序的） |
| **Checkpoint** | LangGraph `SqliteSaver` 自动保存 | `Checkpointer::save()` 手动调，节点执行后保存 |
| **断点续跑** | `graph.invoke(None, config)` | `cp.load_state()` + 从保存的 stage 继续 |
| **消息历史** | MessagesState reducer 自动累加 | Analyst 节点结束时手动清空 `state.messages` |
| **Prompt 模板** | `ChatPromptTemplate.from_messages()` | `str::replace("{var}", value)` |
| **Memory Log** | 手写的 `TradingMemoryLog` | 同结构翻译为 Rust，追加式 markdown 文件 |
| **Reflection** | `Reflector.reflect_on_final_decision()` | 同逻辑翻译，`llm.chat()` 调 LLM |

---

## 关键简化点

1. **没有通用的 DAG 调度器**：LangGraph 是一个通用的有向图执行框架，支持任意拓扑。TradingAgents 的图拓扑完全固定，所以用一个 `run_pipeline()` 函数 + `next_stage()` 路由函数就足够了。这比 LangGraph 简单一个数量级。

2. **没有 Reducer 机制**：Python 的 `MessagesState` 有特殊的列表合并 reducer。Rust 不需要——我们手动管理 message 列表的追加和清空。

3. **没有 LCEL**：`|` 管道符在 Rust 中没有等价物。但我们的节点逻辑足够简单（system prompt → LLM → 存结果），不需要管道组合。

4. **没有 `@tool` 装饰器**：Rust 的 proc macro 可以实现类似效果，但过于复杂。声明式注册 `ToolRegistry` 足够。

5. **并行执行被省略**：Python 版四个 Analyst 从 `START` 分叉后并行执行。但从日志看，实际配置中它们是**顺序**的（`sequential`），所以 Rust 版直接顺序执行。如果将来需要并行，用 `tokio::join!` 即可。

6. **Prompt 模板极简化**：不需要 Jinja2。所有 prompt 是预写好的 markdown 文件，只有 `{ticker}`、`{date}` 等少数占位符需要替换。

---

## 依赖（Cargo.toml 关键项）

```toml
[dependencies]
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.12", features = ["json"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
async-trait = "0.1"
anyhow = "1"
chrono = { version = "0.4", features = ["serde"] }
rusqlite = { version = "0.31", features = ["bundled"] }
regex = "1"
log = "0.4"
clap = { version = "4", features = ["derive"] }
toml = "0.8"

[dev-dependencies]
pretty_assertions = "1"
mockito = "1"    # mock HTTP for testing LLM calls
tempfile = "3"
```
