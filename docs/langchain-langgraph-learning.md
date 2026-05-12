# LangChain & LangGraph 入门学习指南

本文档聚焦 TradingAgents 项目实际用到的 LangChain/LangGraph 技术点，帮助你快速理解核心概念及其在本项目中的用法。

---

## 目录

1. [核心概念速览](#核心概念速览)
2. [LangGraph：构建 Agent 工作流](#langgraph构建-agent-工作流)
3. [LangChain：LLM 调用与工具绑定](#langchainllm-调用与工具绑定)
4. [本项目实战模式](#本项目实战模式)

---

## 核心概念速览

| 技术 | 来源 | 作用 | 本项目中的位置 |
|------|------|------|---------------|
| `StateGraph` | LangGraph | 定义工作流图（节点 + 边） | `tradingagents/graph/setup.py` |
| `MessagesState` | LangGraph | 带消息累加器的状态基类 | `tradingagents/agents/utils/agent_states.py` |
| `ToolNode` | LangGraph | 把工具函数包装成图节点 | `tradingagents/graph/trading_graph.py` |
| `add_conditional_edges` | LangGraph | 条件路由（if-else 分支） | `tradingagents/graph/setup.py` |
| `SqliteSaver` | LangGraph | 断点续跑（崩溃恢复） | `tradingagents/graph/checkpointer.py` |
| `ChatPromptTemplate` | LangChain | 模板化 prompt 构造 | `tradingagents/agents/analysts/` |
| `llm.bind_tools()` | LangChain | 让 LLM 能调用工具 | `tradingagents/agents/analysts/` |
| `llm.with_structured_output()` | LangChain | 让 LLM 输出结构化 JSON | `tradingagents/agents/utils/structured.py` |
| `@tool` 装饰器 | LangChain | 定义 LLM 可调用的工具函数 | `tradingagents/dataflows/core_stock_tools.py` |

---

## LangGraph：构建 Agent 工作流

LangGraph 是一个**有状态图**框架。你把工作流定义成一张图（节点 = 步骤，边 = 流转方向），然后编译、运行。LangGraph 自动管理状态在各个节点间的传递和更新。

### 2.1 StateGraph — 定义图

```python
from langgraph.graph import END, START, StateGraph
from tradingagents.agents.utils.agent_states import AgentState

workflow = StateGraph(AgentState)
```

`StateGraph(AgentState)` 创建一张图，`AgentState` 定义了图中流转的数据结构。

### 2.2 添加节点

```python
workflow.add_node("Market Analyst", market_analyst_node)
workflow.add_node("tools_market", ToolNode([get_stock_data, get_indicators]))
```

每个节点是一个**函数**，签名为 `(state: AgentState) -> dict`。输入当前状态，返回一个 dict（部分状态更新），LangGraph 会自动合并到全局状态中。

本项目中的节点函数都是**工厂函数**生成的闭包：

```python
# tradingagents/agents/analysts/market_analyst.py
def create_market_analyst(llm):
    def market_analyst_node(state: AgentState) -> dict:
        # 使用闭包捕获的 llm 进行推理
        ...
        return {"messages": [result], "market_report": report}
    return market_analyst_node
```

LLM 实例通过闭包注入，而不是通过 state 传递。这是 LangGraph 的推荐模式。

### 2.3 添加边 — 定义流转

**静态边**（固定流转）：

```python
workflow.add_edge(START, "Market Analyst")           # 入口
workflow.add_edge("tools_market", "Market Analyst")  # 工具执行完回到分析师
workflow.add_edge("Portfolio Manager", END)          # 终点
```

`START` 和 `END` 是 LangGraph 的哨兵值，分别代表图的入口和出口。

**条件边**（动态路由）：

```python
workflow.add_conditional_edges(
    "Market Analyst",       # 源节点
    should_continue_market,  # 路由函数
    {
        "tools_market": "tools_market",     # 返回值 → 目标节点
        "Msg Clear Market": "Msg Clear Market",
    }
)
```

路由函数是纯 Python 函数，签名为 `(state: AgentState) -> str`，返回值为下一个节点名：

```python
# tradingagents/graph/conditional_logic.py
def should_continue_market(self, state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools_market"       # LLM 要调工具 → 去执行工具
    return "Msg Clear Market"       # LLM 给出最终回复 → 进入下一阶段
```

### 2.4 MessagesState — 消息累加器

这是 LangGraph 最重要的内置类型之一：

```python
from langgraph.graph import MessagesState

class AgentState(MessagesState):
    company_of_interest: str
    market_report: str
    # ... 更多字段
```

`MessagesState` 的核心是 `messages` 字段，它有一个特殊的 **reducer**：`add_messages`。

**普通字段**：返回 `{"market_report": "xxx"}` → 覆盖旧值
**`messages` 字段**：返回 `{"messages": [new_msg]}` → **追加**到现有消息列表末尾

```python
# 初始状态
state["messages"] = [HumanMessage("分析 AAPL")]

# 节点返回
node_return = {"messages": [AIMessage("让我查一下数据...", tool_calls=[...])]}

# 自动合并后
state["messages"] = [HumanMessage("分析 AAPL"), AIMessage("让我查一下数据...", tool_calls=[...])]
```

这就是多轮 Agent 对话能持续累积上下文的原因。**没有 `MessagesState`，每次返回都会覆盖整个消息列表。**

### 2.5 子状态（Nested TypedDict）

本项目用 `TypedDict` 定义嵌套子状态：

```python
class InvestDebateState(TypedDict):
    history: Annotated[str, "辩论历史"]
    current_response: Annotated[str, "当前发言方"]
    count: Annotated[int, "辩论轮数"]

class AgentState(MessagesState):
    investment_debate_state: InvestDebateState
```

**注意**：子状态没有自定义 reducer，所以返回时会**整体替换**。节点必须返回包含所有字段的完整 dict：

```python
# 在 Bull Researcher 节点中
return {
    "investment_debate_state": {
        "history": old_history + "\n" + new_argument,  # 追加
        "current_response": "Bull Researcher",          # 更新
        "count": old_count + 1,                         # 递增
    }
}
```

### 2.6 编译与运行

```python
# 编译图
graph = workflow.compile(checkpointer=saver)  # 有断点续跑
graph = workflow.compile()                     # 无断点续跑

# 两种运行方式
# 1. stream — 逐步返回（调试用）
for chunk in graph.stream(init_state, stream_mode="values", config=config):
    print(chunk)

# 2. invoke — 一次返回最终结果（生产用）
final_state = graph.invoke(init_state, config=config)
```

`config` 中包含 `thread_id`（用于断点续跑）和 `recursion_limit`（防止无限循环的安全上限，本项目默认 100）。

### 2.7 断点续跑（Checkpoints）

LangGraph 内置的状态持久化机制。本项目使用 SQLite：

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 创建
with SqliteSaver.from_conn_string("checkpoints/AAPL.db") as saver:
    saver.setup()  # 创建 writes 和 checkpoints 表
    graph = workflow.compile(checkpointer=saver)

    # 运行，thread_id 决定"谁的状态"
    graph.invoke(init_state, config={"configurable": {"thread_id": "abc123"}})
```

**原理**：
- 每个节点执行完后，LangGraph 自动将状态写入 SQLite
- 如果崩溃，下次用相同 `thread_id` 运行会自动从最后完成的节点继续
- 本项目用 `SHA256("TICKER:DATE")` 生成 thread_id，所以同一天重新跑同一个股票会自动续跑
- 跑完后清除 checkpoint，避免下次误续

### 2.8 本项目图结构总览

```
START
  │
  ├─ Market Analyst ←→ tools_market ──→ Msg Clear Market
  ├─ Social Analyst ←→ tools_social ──→ Msg Clear Social      (可选)
  ├─ News Analyst ←→ tools_news ──→ Msg Clear News            (可选)
  ├─ Fundamentals Analyst ←→ tools_fundamentals ──→ Msg Clear (可选)
  │
  ├─ Bull Researcher ←→ Bear Researcher    (辩论循环, N轮)
  │
  ├─ Research Manager   (结构化输出: ResearchPlan)
  ├─ Trader             (结构化输出: TraderProposal)
  │
  ├─ Aggressive ←→ Conservative ←→ Neutral  (风险评估循环, N轮)
  │
  └─ Portfolio Manager  (结构化输出: PortfolioDecision) → END
```

---

## LangChain：LLM 调用与工具绑定

### 3.1 ChatPromptTemplate — 模板化 Prompt

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

system_message = "你是一个市场分析师。用中文回答。"

prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder(variable_name="messages"),  # 这里插入消息历史
])

# 构造输入
chain = prompt | llm  # LCEL (LangChain Expression Language)
result = chain.invoke({"messages": state["messages"]})
```

`MessagesPlaceholder` 会在 prompt 中插入完整的消息历史（system → user → assistant → tool → ...），让 LLM 能看到完整的对话上下文。这是构建多轮对话 Agent 的关键组件。

### 3.2 bind_tools — 让 LLM 调用工具

```python
from langchain_core.tools import tool

# 1. 定义工具函数
@tool
def get_stock_data(ticker: str, start_date: str, end_date: str) -> str:
    """获取股票历史价格数据。"""
    ...

# 2. 绑定到 LLM
tools = [get_stock_data, get_indicators]
llm_with_tools = llm.bind_tools(tools)

# 3. LLM 的响应会包含 tool_calls
result = llm_with_tools.invoke(prompt)
# result.tool_calls → [{"name": "get_stock_data", "args": {"ticker": "AAPL", ...}, "id": "..."}]
# 或者 result.content → "根据数据分析，建议买入..."（不需要工具时）
```

配合 LangGraph 的 `ToolNode`：

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)  # 自动解析 tool_calls 并执行对应函数
```

**工具调用循环的完整流程**：
1. 分析师节点调用 LLM → 返回 `tool_calls`
2. 条件边判断 `tool_calls` 存在 → 路由到 `ToolNode`
3. `ToolNode` 执行工具，返回 `ToolMessage`
4. 静态边回到分析师节点 → LLM 看到工具结果，继续推理
5. 循环直到 LLM 返回纯文本（没有 `tool_calls`）

### 3.3 with_structured_output — 结构化输出

让 LLM 返回 Pydantic 对象而非纯文本：

```python
from pydantic import BaseModel

class ResearchPlan(BaseModel):
    """研究计划"""
    recommendation: str   # Buy / Overweight / Hold / Underweight / Sell
    rationale: str        # 理由
    strategic_actions: str # 建议行动

structured_llm = llm.with_structured_output(ResearchPlan)
result = structured_llm.invoke(prompt)
# result 是 ResearchPlan 实例，不是字符串
# result.recommendation → "Buy"
```

**本项目的最佳实践**：结构化输出 + 回退到自由文本

```python
# tradingagents/agents/utils/structured.py
def bind_structured(llm, Schema, agent_name):
    try:
        return llm.with_structured_output(Schema)
    except (NotImplementedError, AttributeError):
        return None  # 某些模型（如 deepseek-reasoner）不支持，静默回退

def invoke_structured_or_freetext(structured_llm, plain_llm, prompt, render_fn):
    if structured_llm:
        try:
            result = structured_llm.invoke(prompt)
            return render_fn(result)  # 转为 markdown 供下游使用
        except Exception:
            pass  # 结构化调用失败，回退到自由文本
    return plain_llm.invoke(prompt).content
```

**为什么要渲染回 markdown？** 因为下游节点（辩论、日志、报告）都消费 markdown 字符串。结构化输出保证格式一致（标题、分段），但最终以 markdown 形式流通。

### 3.4 @tool 装饰器

```python
from langchain_core.tools import tool

@tool
def get_news(ticker: str, date: str) -> str:
    """获取指定股票在指定日期的新闻。

    Args:
        ticker: 股票代码，如 AAPL
        date: 日期，格式 YYYY-MM-DD
    """
    # 实际数据获取逻辑
    return news_text
```

`@tool` 装饰器自动从函数签名和 docstring 提取：
- **工具名**：函数名 (`get_news`)
- **工具描述**：docstring 第一行
- **参数 schema**：从类型注解和 `Args` 文档生成

这些信息会被 `bind_tools` 转成 LLM 能理解的 function calling schema。

---

## 本项目实战模式

### 模式一：Agent 工具循环（分析师）

```
Analyst Node ──(有 tool_calls?)──→ ToolNode ──→ Analyst Node
       │                                    (循环)
       │ (无 tool_calls: 分析完成)
       ▼
   Msg Clear ──→ 下一个分析师
```

关键代码结构：

```python
def create_analyst(llm):
    tools = [tool1, tool2]
    llm_with_tools = llm.bind_tools(tools)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("messages"),
    ])

    def node(state):
        result = prompt | llm_with_tools).invoke({"messages": state["messages"]})
        report = result.content if not result.tool_calls else ""
        return {"messages": [result], "report_field": report}

    return node
```

### 模式二：辩论循环（Bull/Bear Researcher）

```
Bull Researcher ──→ Bear Researcher ──→ Bull Researcher ──→ ...
                                            │ (达到最大轮数)
                                            ▼
                                       Research Manager
```

关键：子状态手动管理，每次返回完整 dict。

### 模式三：结构化输出（Manager/Trader）

```
Report → LLM.with_structured_output(Schema) → Pydantic 对象 → render() → markdown
                                                      │ (失败)
                                                      ▼
                                              LLM.invoke() → 自由文本
```

### 模式四：多 LLM 分层

本项目使用**两个 LLM 实例**：

| LLM 类型 | 用途 | 配置 |
|----------|------|------|
| `quick_thinking_llm` | 分析师、辩论者、交易员 | 快/便宜的模型 |
| `deep_thinking_llm` | Research Manager、Portfolio Manager | 强/贵的模型 |

这是成本与质量的平衡：大量中间步骤用便宜模型，最终决策用强模型。

---

## 关键文件索引

| 文件 | 学习要点 |
|------|---------|
| `tradingagents/graph/setup.py` | StateGraph 构建、条件边、工具循环 |
| `tradingagents/graph/trading_graph.py` | 图编译、运行、checkpoint 管理 |
| `tradingagents/graph/conditional_logic.py` | 条件路由函数（工具循环出口、辩论出口） |
| `tradingagents/agents/utils/agent_states.py` | MessagesState、子状态 TypedDict |
| `tradingagents/agents/utils/structured.py` | with_structured_output + 回退模式 |
| `tradingagents/agents/analysts/market_analyst.py` | 工具绑定 Agent 节点示例 |
| `tradingagents/agents/managers/portfolio_manager.py` | 结构化输出 + Memory 注入 |
| `tradingagents/agents/researchers/bull_researcher.py` | 纯文本辩论节点示例 |
| `tradingagents/llm_clients/` | 多 provider LLM 工厂模式 |
| `tradingagents/graph/checkpointer.py` | SqliteSaver 断点续跑 |
| `tradingagents/dataflows/core_stock_tools.py` | @tool 装饰器定义工具 |

---

## 推荐学习路径

1. **先看** `agent_states.py` — 理解状态结构（5 分钟）
2. **再看** `market_analyst.py` — 理解一个 Agent 节点怎么写（10 分钟）
3. **然后看** `conditional_logic.py` — 理解条件路由怎么工作（10 分钟）
4. **最后看** `setup.py` — 理解整张图怎么串起来（20 分钟）

阅读时重点关注：状态怎么更新、消息怎么流转、LLM 怎么调用、条件路由怎么判断。其余是业务逻辑，可以后续再看。
