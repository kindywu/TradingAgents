# LangChain & LangGraph 深入浅出教程

本教程面向零基础读者，从"为什么"开始，逐步深入到"怎么做"，用最简短的代码讲清楚每个概念。

---

## 目录

1. [为什么要用 LangChain？](#为什么要用-langchain)
2. [LangChain 基础：LLM 调用三板斧](#langchain-基础llm-调用三板斧)
3. [LCEL：用管道串联你的逻辑](#lcel用管道串联你的逻辑)
4. [RunnablePassthrough & RunnableLambda：自定义管道环节](#runnablepassthrough--runnablelambda自定义管道环节)
5. [Tool Calling：让 LLM 能动手做事](#tool-calling让-llm-能动手做事)
6. [Structured Output：让 LLM 输出格式化数据](#structured-output让-llm-输出格式化数据)
7. [流式输出：边生成边输出](#流式输出边生成边输出)
8. [RAG 基础：让 LLM 能查资料](#rag-基础让-llm-能查资料)
9. [为什么要用 LangGraph？](#为什么要用-langgraph)
10. [LangGraph 基础：有状态图](#langgraph-基础有状态图)
11. [MessagesState：消息累加器](#messagesstate消息累加器)
12. [自定义 Reducer：不只是追加](#自定义-reducer不只是追加)
13. [条件边：动态路由](#条件边动态路由)
14. [Send API：动态并行分发](#send-api动态并行分发)
15. [Agent 工具循环](#agent-工具循环)
16. [Fallbacks & 错误处理：LLM 挂了怎么办](#fallbacks--错误处理llm-挂了怎么办)
17. [多 Agent 协作](#多-agent-协作)
18. [断点续跑](#断点续跑)
19. [LangGraph 流式输出](#langgraph-流式输出)
20. [完整实战：构建一个股票分析 Agent](#完整实战构建一个股票分析-agent)

---

## 为什么要用 LangChain？

直接用原生 SDK 调用 LLM 其实很简单：

```python
from openai import OpenAI
client = OpenAI(api_key="your-api-key", base_url="https://api.deepseek.com/v1")
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "你好"}],
)
print(response.choices[0].message.content)
```

**但当你要做这些事情时，代码会迅速膨胀：**

- 管理多轮对话历史（手动拼接 messages 列表）
- 切换模型提供商（OpenAI → Anthropic → 国产模型，API 各不相同）
- 让 LLM 调用工具（解析 function call、执行函数、把结果塞回对话）
- 构建复杂的 prompt（模板复用、变量插值、对话历史注入）

LangChain 做的事情很简单：**把 LLM 开发中的重复模式抽象成标准接口**。它不"重"，核心代码量很小。

> 一句话：LangChain 是 LLM 应用的"标准库"，让你换模型、调工具、管对话不用每次都重复造轮子。

---

## LangChain 基础：LLM 调用三板斧

### 第一板：统一调用接口

无论底层是 OpenAI、Anthropic 还是国产模型，调用方式都一样：

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# 两个模型，用法完全一致
llm_ds = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com/v1")
llm_claude = ChatAnthropic(model="claude-sonnet-4-6")

# 统一调用
result = llm_ds.invoke("你好")       # → AIMessage(content="你好！有什么可以帮你的？")
result = llm_claude.invoke("你好")       # → AIMessage(content="你好！请问...")
```

所有 LLM 返回的都是统一的 `AIMessage` 对象，而不是各自厂商的原始响应格式。

### 第二板：消息类型

LangChain 用三种消息对象管理对话：

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

messages = [
    SystemMessage(content="你是一个幽默的助手"),
    HumanMessage(content="讲个笑话"),
]
response = llm.invoke(messages)  # 返回 AIMessage

# 多轮对话就是不断往列表追加
messages.append(response)                      # 追加 AI 回复
messages.append(HumanMessage(content="再来一个"))  # 追加用户消息
response = llm.invoke(messages)
```

`SystemMessage` = 系统指令，`HumanMessage` = 用户说的话，`AIMessage` = LLM 的回复。

### 第三板：Prompt 模板

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，用{language}回答"),
    ("human", "{question}"),
])

# 模板 + LLM 组合
chain = prompt | llm
result = chain.invoke({
    "role": "股票分析师",
    "language": "中文",
    "question": "介绍一下价值投资的核心思想",
})
# result.content → 一段中文回复

# 换个角色，复用同一个模板
result = chain.invoke({
    "role": "幼儿园老师",
    "language": "中文",
    "question": "什么是月亮？",
})
```

`{变量}` 语法让 prompt 像函数一样可复用。`|` 是 LCEL 管道符，把 prompt 输出传给 LLM。

---

## LCEL：用管道串联你的逻辑

LCEL（LangChain Expression Language）是 LangChain 的核心范式。它的语法就一个操作符：`|`（管道符）。

### 基本思想

```python
chain = step1 | step2 | step3
result = chain.invoke(data)
```

数据从左到右流动，每一步的输出是下一步的输入。很像 Unix 管道 `cat file | grep | sort`。

### 实际示例

```python
from langchain_core.output_parsers import StrOutputParser

# 三件套：模板 → LLM → 字符串解析
chain = prompt | llm | StrOutputParser()

# 等价于：
# 1. prompt.invoke({...}) → 拼好 messages
# 2. llm.invoke(messages) → AIMessage
# 3. StrOutputParser().invoke(aimsg) → 纯文本字符串

result = chain.invoke({"topic": "Python"})
# result 是 str，不是 AIMessage。StrOutputParser 自动提取了 .content
```

### 链式分支

```python
# 两个 prompt 各自生成内容，LLM 一次性处理
chain = (
    {
        "context": retriever | format_docs,  # 检索 + 格式化
        "question": lambda x: x["question"], # 透传问题
    }
    | prompt
    | llm
)
chain.invoke({"question": "什么是 RAG？"})
```

`{...}` 是一个 RunnableDict，里面的 key 可以各自独立处理，最后合并成一个 dict 传给 prompt。

> 一句话：LCEL 让你用 `|` 把 LLM 调用串成流水线，告别手写胶水代码。

---

## RunnablePassthrough & RunnableLambda：自定义管道环节

LCEL 的 `|` 要求每个环节都是 Runnable。如果某个环节只是一个普通函数，怎么办？

### RunnableLambda：把函数变成管道环节

```python
from langchain_core.runnables import RunnableLambda

def format_docs(docs: list) -> str:
    """把文档列表拼接成字符串"""
    return "\n\n".join(doc.page_content for doc in docs)

# 包装后就能用 | 串联
chain = retriever | RunnableLambda(format_docs) | prompt | llm
```

### RunnablePassthrough：透传数据

管道中经常需要透传某些数据——不修改，只是往后传：

```python
from langchain_core.runnables import RunnablePassthrough

# 典型用法：dict 中部分字段需要处理，部分字段直接透传
chain = (
    {
        "context": retriever | format_docs,     # 这个字段走检索+格式化
        "question": RunnablePassthrough(),       # 这个字段直接透传
    }
    | prompt | llm | StrOutputParser()
)

chain.invoke("什么是 RAG？")
# prompt 收到: {"context": "文档内容...", "question": "什么是 RAG？"}
```

### RunnablePassthrough.assign()：往 dict 追加新字段

```python
answer_chain = prompt | llm | StrOutputParser()

# assign：在原始输入 dict 基础上追加一个新 key
full_chain = RunnablePassthrough.assign(answer=answer_chain)

result = full_chain.invoke({"question": "什么是 RAG？"})
# result == {"question": "什么是 RAG？", "answer": "RAG 是检索增强生成..."}
```

很适合**在保持原有信息的同时补全 LLM 计算结果**。

> 一句话：`RunnableLambda` 让任何函数融入管道，`RunnablePassthrough` 让数据选择性透传，`assign` 在 dict 上追加计算结果。

---

## Tool Calling：让 LLM 能动手做事

LLM 本质上只能"说话"，不能查数据库、调 API、读文件。Tool Calling（函数调用）解决了这个问题：**LLM 告诉你它想调什么函数、传什么参数，由你（或框架）执行，然后把结果传回去。**

### 定义工具

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    # 实际项目中调天气 API，这里模拟
    return f"{city}今天晴天，25°C"

@tool
def calculate(expression: str) -> str:
    """计算数学表达式。例如: '2+3*4'"""
    return str(eval(expression))  # 仅示例，生产环境请用安全求值
```

`@tool` 装饰器从**函数签名**和**docstring**自动提取工具名、描述和参数 schema。这些信息会传给 LLM，让 LLM 知道有哪些工具可用、什么时候调用。

### 绑定工具到 LLM

```python
tools = [get_weather, calculate]
llm_with_tools = llm.bind_tools(tools)

# LLM 自动判断：回答问题需要工具吗？
result = llm_with_tools.invoke("北京今天天气怎么样？")
# result.tool_calls → [{"name": "get_weather", "args": {"city": "北京"}, "id": "call_1"}]
# result.content → ""（调用了工具，没有文字回复）

result = llm_with_tools.invoke("你好呀")
# result.tool_calls → []（不需要工具，直接回复）
# result.content → "你好！有什么可以帮你的？"
```

### 执行工具并返回结果

```python
# 1. LLM 决定调用 get_weather
ai_msg = llm_with_tools.invoke("北京天气？")

# 2. 执行工具
from langchain_core.messages import ToolMessage

tool_results = []
for tc in ai_msg.tool_calls:
    if tc["name"] == "get_weather":
        result = get_weather.invoke(tc["args"])  # 实际执行
        tool_results.append(ToolMessage(content=result, tool_call_id=tc["id"]))

# 3. 把工具结果+历史消息传给 LLM 继续推理
messages = [HumanMessage("北京天气？"), ai_msg] + tool_results
final = llm.invoke(messages)
# final.content → "北京今天晴天，25°C，是个好天气！"
```

整个过程：**用户提问 → LLM 要工具 → 你执行工具 → LLM 看结果 → LLM 最终回答。**

---

## Structured Output：让 LLM 输出格式化数据

很多时候你不想拿纯文本，想要结构化的 JSON/对象，方便程序处理。

### 方法一：with_structured_output

```python
from pydantic import BaseModel, Field

class WeatherReport(BaseModel):
    """天气报告"""
    city: str = Field(description="城市名")
    temperature: float = Field(description="摄氏温度")
    condition: str = Field(description="天气状况，如晴天/阴天/雨天")
    advice: str = Field(description="出行建议")

# 让 LLM 输出 Pydantic 对象
structured_llm = llm.with_structured_output(WeatherReport)

result = structured_llm.invoke("北京今天30度，大太阳")
# result 是 WeatherReport 实例，不是字符串
print(result.city)         # "北京"
print(result.temperature)  # 30.0
print(result.condition)    # "晴天"
```

### 方法二：TypedDict 方案

```python
from typing import TypedDict

class SentimentResult(TypedDict):
    sentiment: str    # "positive" / "negative" / "neutral"
    score: float      # 0.0 - 1.0
    reason: str

structured_llm = llm.with_structured_output(SentimentResult)
result = structured_llm.invoke("分析：这个产品太好用了")
# result["sentiment"] → "positive", result["score"] → 0.9
```

> 一句话：Structured Output 让 LLM 从"写作文"变成"填表格"，方便写代码处理。

---

## 流式输出：边生成边输出

长回复如果等 LLM 全部生成完再显示，用户会觉得很慢。流式输出可以逐 token 输出，像 ChatGPT 的打字效果。

### LLM 级别

```python
for chunk in llm.stream("用中文介绍 Python"):
    print(chunk.content, end="", flush=True)
```

### Chain 级别

LCEL chain 自动继承 `.stream()`——不需要任何额外配置：

```python
chain = prompt | llm | StrOutputParser()

for chunk in chain.stream({"topic": "Python"}):
    print(chunk, end="", flush=True)
```

### 异步流式

```python
async for chunk in chain.astream({"topic": "Python"}):
    print(chunk, end="", flush=True)
```

### 流式 + Tool Calling

当 LLM 调用工具时流式也正常工作——先流式输出 tool_calls 参数，工具执行完后继续流式输出最终回复。

> 一句话：`.stream()` 让 LLM 打字机式输出，LCEL chain 自动继承这个能力，无需额外配置。

---

## RAG 基础：让 LLM 能查资料

RAG（Retrieval Augmented Generation，检索增强生成）是 LangChain 的最大应用场景——**让 LLM 在回答前先查外部文档，用私有知识弥补训练数据的不足和时效性问题**。

### 核心流程

```
文档加载 → 文本切分 → 向量化 → 存入向量库 → 检索 → 增强 prompt → LLM 生成
```

### 最小实现

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 加载 + 切分文档
loader = TextLoader("knowledge.txt")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(loader.load())

# 2. 向量化 + 存入向量库
vectorstore = Chroma.from_documents(chunks, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. RAG Chain —— 回顾上一节的 RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_messages([
    ("system", "根据以下资料回答问题：\n\n{context}"),
    ("human", "{question}"),
])

chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        "question": RunnablePassthrough(),
    }
    | prompt | llm | StrOutputParser()
)

chain.invoke("公司去年的营收是多少？")
```

### 进阶方向

- **多路检索**：从多个数据源同时检索，合并结果
- **Query 改写**：把用户问题改写成更适合检索的形式
- **Reranking**：对检索结果二次排序，提高相关性

> 一句话：RAG = 检索 + 生成，先查资料再回答，让 LLM 的知识从"训练截止日"延伸到"你的私有文档"。

---

## 为什么要用 LangGraph？

LangChain 擅长处理**线性的、单向的**流程（A → B → C → D）。但构建 Agent 时，流程往往是**循环的、有分支的**：

- LLM 调用工具 → 看到结果 → 还想调工具 → 循环
- 两个 Agent 互相对话 → 你来我往 → 循环
- 出错了要重试 → 跳到错误处理节点 → 再回来

用 `if/while` 手写这些逻辑能工作，但会越来越乱。LangGraph 的思路：

**把你的 Agent 工作流画成一张有向图——节点是执行步骤，边是流转方向，状态在节点间自动传递和累积。LangGraph 管理状态，你只写节点的业务逻辑。**

---

## LangGraph 基础：有状态图

### 第一步：定义状态

```python
from typing import TypedDict

class MyState(TypedDict):
    messages: list
    user_name: str
    task_result: str
```

`TypedDict` 定义图中流转的数据结构。每个节点读这个 dict，返回部分字段，LangGraph 自动合并。

### 第二步：创建图

```python
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(MyState)
```

### 第三步：定义节点函数

```python
def greeter(state: MyState) -> dict:
    """每个节点接收 state，返回要更新的字段"""
    name = state["user_name"]
    return {"messages": [f"你好 {name}！"]}

def task_doer(state: MyState) -> dict:
    return {"task_result": "任务完成"}
```

节点函数的签名统一是 `(state) -> dict`。返回的 dict 会**合并**到全局 state 中。

### 第四步：加边，编译，运行

```python
# 添加节点
workflow.add_node("greeter", greeter)
workflow.add_node("task_doer", task_doer)

# 定义流转
workflow.add_edge(START, "greeter")      # 入口 → greeter
workflow.add_edge("greeter", "task_doer") # greeter → task_doer
workflow.add_edge("task_doer", END)      # task_doer → 出口

# 编译并运行
graph = workflow.compile()
result = graph.invoke({"user_name": "小明"})
# result → {"user_name": "小明", "messages": ["你好 小明！"], "task_result": "任务完成"}
```

流程图：

```
START → greeter → task_doer → END
```

`START` 和 `END` 是 LangGraph 内置的哨兵节点，代表图的起点和终点。

---

## MessagesState：消息累加器

这是 LangGraph 最重要的内置状态类型。先看问题：

```python
class MyState(TypedDict):
    messages: list

def node_a(state):
    return {"messages": [AIMessage("A说了一些话")]}

def node_b(state):
    return {"messages": [AIMessage("B说了一些话")]}

# 问题：node_b 返回后会覆盖 node_a 的结果！
# 最终 messages = [AIMessage("B说了一些话")]，node_a 的丢了
```

`MessagesState` 解决了这个问题——它的 `messages` 字段有一个**特殊的累加器（reducer）**：

```python
from langgraph.graph import MessagesState

class MyAgentState(MessagesState):
    # messages 字段已内置（带累加逻辑），你只需要加额外字段
    report: str
    decision: str
```

**关键区别**：

| 普通字段 | `messages` 字段 |
|---------|----------------|
| `return {"report": "xxx"}` → 覆盖旧值 | `return {"messages": [msg]}` → **追加**到列表末尾 |
| 新值替换旧值 | 新消息追加到已有消息后 |

```python
# 初始
state["messages"] = [HumanMessage("分析 AAPL")]

# node_a 返回
return {"messages": [AIMessage("让我查数据...")]}
# 自动合并后 → [HumanMessage("分析 AAPL"), AIMessage("让我查数据...")]

# node_b 返回
return {"messages": [AIMessage("数据拿到了，建议买入")]}
# 自动合并后 → [HumanMessage("分析 AAPL"), AIMessage("让我查数据..."), AIMessage("数据拿到了，建议买入")]
```

这就是 Agent 多轮对话能**持续累积上下文**的底层机制。没有这个累加器，每次返回都会覆盖整个对话历史。

---

## 自定义 Reducer：不只是追加

MessagesState 的消息累加器本质上是 **reducer**——一个决定"新值如何合并到旧值"的函数。你可以用 `Annotated` 为任何字段自定义 reducer。

### 语法

```python
from typing import Annotated
from operator import add

class MyState(TypedDict):
    # add reducer：新列表追加到旧列表
    history: Annotated[list, add]

    # 默认行为（无 Annotated）：新值直接覆盖旧值
    name: str

    # 取最大值
    max_score: Annotated[float, lambda current, update: max(current, update)]

    # 取最新非空值
    latest_report: Annotated[str, lambda current, update: update if update else current]

    # 累加计数器
    call_count: Annotated[int, lambda current, update: current + update]
```

### 执行逻辑

当节点 `return {"max_score": 0.8}` 时，LangGraph 不直接覆盖，而是：

```python
# 内部：reducer(current=0.3, update=0.8) → 0.8
new_max_score = reducer(0.3, 0.8)
```

### MessagesState 的秘密

```python
# MessagesState 本质上就是：
class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]
```

`add_messages` 是内置的特殊 reducer——不仅追加消息，还会用消息 `id` 去重和替换同 ID 的旧消息（同名覆盖）。如果你只需要简单列表追加，用 `operator.add` 就够。

> 一句话：`Annotated[类型, reducer函数]` 让你自定义每个字段的合并策略——追加、取最大、累加、只保留非空……全由你定。

---

## 条件边：动态路由

静态边是固定的 A → B。条件边是：**根据当前状态，决定下一步去哪**。

```python
import random

def random_router(state: MyState) -> str:
    """路由函数：返回下一个节点的名字"""
    if random.random() > 0.5:
        return "path_a"
    return "path_b"

# 加条件边
workflow.add_conditional_edges(
    "source_node",   # 源节点
    random_router,   # 路由函数（签名为 (state) -> str）
    {
        "path_a": "node_a",  # 返回 "path_a" → 去 node_a
        "path_b": "node_b",  # 返回 "path_b" → 去 node_b
    }
)
```

**路由函数的规则**：
- 签名为 `(state) -> str`
- 返回值必须是 `add_conditional_edges` 的第三个参数中定义的 key
- 逻辑可以是任意的 Python 代码：读 state 字段、调 LLM、查数据库……

**实际应用**——工具循环的出口判断：

```python
def should_call_tool(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"       # LLM 说需要调工具 → 去执行
    return "next_step"       # LLM 给出了最终回复 → 继续下一阶段
```

**多分支条件边**：

```python
def complexity_router(state) -> str:
    question = state["question"]
    if len(question) < 10:
        return "simple"
    elif "代码" in question:
        return "code"
    return "complex"

workflow.add_conditional_edges(
    "router",
    complexity_router,
    {
        "simple": "simple_handler",
        "code": "code_handler",
        "complex": "complex_handler",
    }
)
```

---

## Send API：动态并行分发

条件边是"根据状态选一条路"（一对一路由）。但有时你需要"根据状态生成 N 条并行路"（一对多分发），每条路的输入还不一样。这就是 Send API 的用途。

### 场景：同时分析多只股票

```python
from langgraph.types import Send

class BatchState(TypedDict):
    tickers: list[str]          # ["AAPL", "GOOGL", "TSLA"]
    reports: Annotated[list, add]  # add reducer 收集所有报告

def analyst(state):
    """分析单只股票——ticker 来自 Send 传入的参数"""
    ticker = state["ticker"]
    return {"reports": [f"{ticker} 分析完成：建议买入"]}

def fan_out(state) -> list[Send]:
    """为每只股票生成一个 analyst 实例"""
    return [Send("analyst", {"ticker": t}) for t in state["tickers"]]

workflow.add_node("analyst", analyst)
workflow.add_conditional_edges(START, fan_out, {"analyst": "analyst"})
workflow.add_edge("analyst", END)
```

3 只股票 → 3 个 `analyst` 实例并行执行，各自拿到不同的 `ticker` 参数，结果自动用 `add` reducer 收集。

### 与静态并行的对比

| | 静态并行（多边） | Send API |
|---|---|---|
| 并行数 | 固定，编译时确定 | 动态，运行时由数据决定 |
| 输入 | 所有实例共享同一个 state | 每个实例有独立参数 |
| 适用场景 | 固定的分析师团队 | 按列表元素数量动态分叉 |

```
静态并行:                 Send API (map-reduce):
  START                     START
  ↓   ↓   ↓                 ↓ fan_out → 生成 N 个 Send
  A   B   C                 ↓ analyst("AAPL") | analyst("GOOGL") | ...
  ↓   ↓   ↓                 ↓ 结果自动用 add reducer 收集到 reports
  汇总
```

> 一句话：Send API 实现 map-reduce 模式——运行时动态 fan-out 到 N 个并行实例，每个实例拿到不同参数。

---

## Agent 工具循环

把前面的内容串起来，实现一个完整的"LLM 自主调工具"Agent：

```
Human提问 → LLM思考 → 要不要调工具？
                        ↓ (要)        ↓ (不要)
                    ToolNode执行     最终回答
                        ↓
                    回到 LLM（看结果继续思考）
```

### 代码实现

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 1. 定义工具
@tool
def search(query: str) -> str:
    """搜索互联网信息"""
    return f"关于'{query}'的搜索结果：..."

@tool
def save_to_file(content: str, filename: str) -> str:
    """保存内容到文件"""
    with open(filename, "w") as f:
        f.write(content)
    return f"已保存到 {filename}"

tools = [search, save_to_file]

# 2. 给 LLM 绑定工具
llm = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com/v1")
llm_with_tools = llm.bind_tools(tools)

# 3. 核心节点：LLM 推理
def agent(state: MessagesState) -> dict:
    """Agent 节点：调用 LLM"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 4. 建图
workflow = StateGraph(MessagesState)

workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")

# 5. 加条件边：agent 执行完后，判断是去调工具还是结束
def should_continue(state: MessagesState) -> str:
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return END

workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")  # 工具执行完回到 agent

graph = workflow.compile()

# 6. 运行
result = graph.invoke({
    "messages": [HumanMessage("帮我搜索 LangGraph 最新动态，然后保存到 result.txt")]
})
```

### 运行时发生了什么

```
第1轮: agent 调 LLM → LLM 返回 tool_calls=[search("LangGraph 最新动态")]
      → 条件边路由到 tools
      → ToolNode 执行 search，返回 ToolMessage("关于 LangGraph...")
      → tools → agent（进入第2轮）

第2轮: agent 调 LLM（看到搜索结果）→ LLM 返回 tool_calls=[save_to_file("...", "result.txt")]
      → 条件边路由到 tools
      → ToolNode 执行 save_to_file，返回 ToolMessage("已保存到 result.txt")
      → tools → agent（进入第3轮）

第3轮: agent 调 LLM（看到保存成功）→ LLM 返回 "已帮你搜索并保存..."
      → 条件边路由到 END
```

这就是 LangGraph 最核心的模式：**Agent 工具循环**。

---

## Fallbacks & 错误处理：LLM 挂了怎么办

生产环境中，LLM 可能超时、限流、返回格式错误。LangChain 和 LangGraph 都提供了容错机制。

### LangChain：模型级降级

```python
primary_llm = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com/v1", max_retries=2)
backup_llm = ChatOpenAI(model="deepseek-v4-flash", base_url="https://api.deepseek.com/v1")

robust_llm = primary_llm.with_fallbacks([backup_llm])
# deepseek-chat 挂了 → 自动切到 deepseek-v4-flash
```

`with_fallbacks` 返回一个新的 Runnable，按顺序尝试，第一个成功的返回结果。Chain 级别也可以用：

```python
chain = prompt | primary_llm.with_fallbacks([backup_llm]) | StrOutputParser()
```

### LangGraph：节点内 try/except

```python
def safe_agent(state: MessagesState) -> dict:
    try:
        response = llm_with_tools.invoke(state["messages"])
    except Exception:
        return {"messages": [AIMessage(content="抱歉，服务暂时不可用。")]}
    return {"messages": [response]}
```

### ToolNode 自动重试

```python
from langgraph.prebuilt import ToolNode

tools_node = ToolNode(tools, retry_on=[ConnectionError], max_retries=3)
```

### 关键原则

- **模型降级**用 `with_fallbacks`——不影响图结构
- **业务逻辑异常**用节点内 try/except——返回有意义的状态而非崩溃
- **工具重试**用 ToolNode 的 `retry_on`——自动重试瞬态错误
- **别吃掉所有异常**——只处理你能恢复的类型，让真正的 bug 暴露出来

> 一句话：`with_fallbacks` 自动切备用模型，ToolNode `retry_on` 自动重试工具，节点 try/except 兜底——三层防护。

---

## 多 Agent 协作

工具循环是"一个 Agent + 工具"。更复杂的场景是**多个 Agent 像团队一样协作**。

### 模式一：流水线（串联）

每个 Agent 完成自己的任务，输出交给下一个：

```python
def researcher(state):  # 做研究
    return {"research": "研究发现..."}

def writer(state):      # 写报告
    return {"report": f"基于研究：{state['research']}，写出报告"}

def reviewer(state):    # 审核
    return {"final_report": state["report"] + "\n[已审核]"}

workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "reviewer")
workflow.add_edge("reviewer", END)
```

### 模式二：辩论（B 跳）

两个 Agent 互相辩论直到达成共识或达到轮数上限：

```python
def bull(state: DebateState) -> dict:
    """多头：提出看涨理由"""
    response = llm.invoke(f"你是多头分析师，针对以下观点反驳并给出看涨理由：{state.get('current')}")
    return {"history": state["history"] + f"\n多头：{response.content}", "round": state["round"] + 1}

def bear(state: DebateState) -> dict:
    """空头：提出看跌理由"""
    response = llm.invoke(f"你是空头分析师，反驳多头的观点：{state.get('current')}")
    return {"history": state["history"] + f"\n空头：{response.content}", "round": state["round"]}

def debate_router(state: DebateState) -> str:
    if state["round"] >= 3:   # 最多辩论 3 轮
        return "judge"        # 去裁判
    return "bull" if state["round"] % 2 == 0 else "bear"

workflow.add_conditional_edges("bull", debate_router, {
    "judge": "judge",
    "bear": "bear",
})
workflow.add_conditional_edges("bear", debate_router, {
    "judge": "judge",
    "bull": "bull",
})
```

流程图：

```
Bull → Bear → Bull → Bear → Judge → END
  ↑______________|                (3轮后)
```

### 模式三：并行 + 汇总

多个分析师同时工作，结果汇总到一个节点：

```python
# 四个分析师各自独立分析
for name in ["market_analyst", "news_analyst", "fundamentals_analyst", "social_analyst"]:
    workflow.add_node(name, create_analyst(llm))

# 所有分析师结果汇总到 manager
workflow.add_edge("market_analyst", "manager")
workflow.add_edge("news_analyst", "manager")
workflow.add_edge("fundamentals_analyst", "manager")
workflow.add_edge("social_analyst", "manager")

# START 连到所有分析师——它们并行执行
for name in ["market_analyst", "news_analyst", "fundamentals_analyst", "social_analyst"]:
    workflow.add_edge(START, name)
```

LangGraph 会自动并行执行从同一个源节点分叉出来的多个目标节点。汇总节点等待所有前驱完成后才执行。

---

## 断点续跑

LangGraph 内置了 checkpoint 机制：**每个节点执行完后，自动把整个 state 持久化**。崩溃后恢复的关键是——你不需要手动保存任何东西，节点跑完一步，状态就存一步。

### 先用一个简单流程来看

```python
from typing import TypedDict

class TaskState(TypedDict):
    research: str
    analysis: str
    report: str

def research_node(state: TaskState) -> dict:
    """第1步：做研究"""
    return {"research": "研究发现：AI 芯片需求增长 200%"}

def analyze_node(state: TaskState) -> dict:
    """第2步：分析——假设这里调用不稳定的外部 API，可能崩溃"""
    data = state["research"]
    # 模拟：如果外部 API 挂了，这里抛异常
    raise ConnectionError("API 超时！")
    # return {"analysis": f"分析结论：{data} → 建议重点投资"}

def report_node(state: TaskState) -> dict:
    """第3步：写报告"""
    return {"report": f"最终报告\n研究：{state['research']}\n分析：{state['analysis']}"}
```

### 场景一：崩溃恢复（最常用的场景）

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 建图
workflow = StateGraph(TaskState)
workflow.add_node("research", research_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("report", report_node)
workflow.add_edge(START, "research")
workflow.add_edge("research", "analyze")
workflow.add_edge("analyze", "report")
workflow.add_edge("report", END)

# 编译时挂 checkpointer
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "task_001"}}

# === 第一次运行：崩溃 ===
try:
    graph.invoke({"research": "", "analysis": "", "report": ""}, config)
except ConnectionError as e:
    print(f"💥 崩溃了：{e}")
    # 程序虽然崩了，但 research_node 的结果已经保存了！
```

**崩溃后，先查"伤势"——看看到底保存了什么：**

```python
# 查看当前状态——最核心的 API
snapshot = graph.get_state(config)

print("已保存的数据：")
print(snapshot.values)
# → {"research": "研究发现：AI 芯片需求增长 200%", "analysis": "", "report": ""}
#    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ research 的结果还在！analysis 还没跑所以是空的

print("下一步该执行哪个节点：")
print(snapshot.next)
# → ("analyze",)
#    ^^^^^^^^^ 精准告诉你：下次 invoke 会从 analyze 开始，不会重跑 research

# 查看历史：到底经历了哪些步骤？
print("执行历史：")
for c in graph.get_state_history(config):
    step = c.metadata.get("step")
    source = c.metadata.get("source")
    next_nodes = c.next
    print(f"  step {step} (来源: {source}) → 接下来执行: {next_nodes}")
# 输出类似：
#   step 1 (来源: loop) → 接下来执行: ("analyze",)
#   step 0 (来源: input) → 接下来执行: ("__start__",)
```

**确认无误，从断点继续：**

```python
# invoke(None) = "我不传新状态，用你上次存的继续往下走"
result = graph.invoke(None, config)

print("最终结果：")
# → {"research": "研究发现：AI 芯片需求增长 200%",
#     "analysis": "分析结论：研究发现... → 建议重点投资",
#     "report": "最终报告\n研究：...\n分析：..."}
```

**回顾整个恢复过程：**

```
第1次 invoke → research ✅（已保存）→ analyze 💥 崩溃
                         ↓
              get_state(config)  → 看到 research 结果在，下一步是 analyze
                         ↓
第2次 invoke(None) → analyze ✅（从 checkpoint 恢复状态继续）→ report ✅ → 完成
```

关键洞察：**`get_state(config)` 是你恢复时的"眼睛"**——让你在继续之前看清当前状态、确认下一步要跑什么。不是盲目地 `invoke(None)` 然后祈祷。

### 场景二：人工审核（interrupt_before）

有时候你不想崩溃后才介入，而是**主动要求在某个节点前暂停**，让人检查后再放行：

```python
# 在 analyze 前设置断点
graph = workflow.compile(checkpointer=memory, interrupt_before=["analyze"])

# 运行——跑到 analyze 前自动暂停
result = graph.invoke(
    {"research": "", "analysis": "", "report": ""},
    config={"configurable": {"thread_id": "review_001"}},
)

# 查看当前状态
snapshot = graph.get_state({"configurable": {"thread_id": "review_001"}})
print(snapshot.values)  # {"research": "研究发现：AI 芯片需求增长 200%", "analysis": "", ...}
print(snapshot.next)    # ("analyze",)  ← 下一步是 analyze，但现在暂停了

# 人工检查 research 结果没问题，放行
result = graph.invoke(None, {"configurable": {"thread_id": "review_001"}})
```

### 生产环境用 SQLite

MemorySaver 在进程中，重启就没了。生产环境用 SQLite 持久化到磁盘：

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as saver:
    saver.setup()  # 自动建表
    graph = workflow.compile(checkpointer=saver)

    config = {"configurable": {"thread_id": "prod_task_001"}}
    graph.invoke(initial_state, config)

# === 程序重启后 ===
with SqliteSaver.from_conn_string("checkpoints.db") as saver:
    graph = workflow.compile(checkpointer=saver)

    # checkpoint 还在磁盘上，直接用同一个 thread_id 恢复
    snapshot = graph.get_state({"configurable": {"thread_id": "prod_task_001"}})
    print(snapshot.values)  # 数据完好

    # 从断点继续
    result = graph.invoke(None, {"configurable": {"thread_id": "prod_task_001"}})
```

### 核心要点

| 场景 | 做法 |
|------|------|
| **崩溃恢复** | `graph.invoke(None, config)` —— 不传新状态，用 checkpoint 里的状态继续 |
| **查看保存了什么** | `graph.get_state(config)` —— 返回 values（数据）+ next（下一步节点） |
| **查看执行历史** | `graph.get_state_history(config)` —— 按时间倒序列出所有 checkpoint |
| **主动暂停** | `compile(interrupt_before=["节点名"])` —— 在指定节点前暂停，等人审核后再 `invoke(None)` |
| **持久化到磁盘** | `SqliteSaver.from_conn_string("xxx.db")` —— 重启不丢 |

一句话：**checkpoint 存的是整个 state，不只是 messages**。你在自定义 State 中加的所有字段（report、decision、analysis……）都会被持久化。恢复时 `invoke(None)` 就是告诉 LangGraph："用上次存的状态继续，从上次中断的地方开始"。

---

## LangGraph 流式输出

LangGraph 的 `.stream()` 支持多种流式模式，让你看到图内部正在发生什么。

### 三种核心模式

```python
config = {"configurable": {"thread_id": "demo"}}

# 模式1：values —— 每个节点执行后，输出完整的 state
for event in graph.stream(input_data, config, stream_mode="values"):
    print(event)  # 每个节点后输出完整 state

# 模式2：updates —— 每个节点执行后，只输出该节点返回的增量
for event in graph.stream(input_data, config, stream_mode="updates"):
    print(event)  # {"analyst": {"messages": [AIMessage(...)]}}

# 模式3：messages —— 输出 LLM token 流（打字机效果）
for event in graph.stream(input_data, config, stream_mode="messages"):
    # event = (AIMessageChunk, metadata)
    print(event[0].content, end="", flush=True)
```

### 组合模式

```python
# 同时用两种模式——UI 更新用 updates，聊天窗口用 messages
for event in graph.stream(input, config, stream_mode=["updates", "messages"]):
    mode, data = event
    if mode == "updates":
        print(f"节点完成: {data}")
    elif mode == "messages":
        print(data[0].content, end="", flush=True)
```

### 模式对比

| 模式 | 输出内容 | 用途 |
|------|---------|------|
| `values` | 完整 state | 调试、审计日志 |
| `updates` | 节点返回的增量 | 跟踪节点进度 |
| `messages` | LLM token 流 | 打字机效果、前端展示 |
| `debug` | 详细执行信息 | 开发调试 |

> 一句话：`stream_mode="messages"` 实现打字机效果，`stream_mode="updates"` 跟踪节点进度，两者可以组合使用。

---

## 完整实战：构建一个股票分析 Agent

把前面所有概念串起来，实现一个简化的股票分析 Agent。

### 整体架构

```
用户提问 → 股票数据 Agent（查价格、指标）
         → 新闻分析 Agent（搜索新闻）
         → 综合判断 Agent（汇总 + 决策）
         → 输出结构化结果
```

### 完整代码

```python
import os
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# ============================================================
# 第1步：定义工具
# ============================================================

@tool
def get_stock_price(ticker: str) -> str:
    """获取股票最新价格。ticker 如 AAPL, GOOGL"""
    # 模拟数据（实际项目中调用 yfinance 或 API）
    prices = {"AAPL": "188.50", "GOOGL": "142.30", "TSLA": "245.10", "MSFT": "378.20"}
    return f"{ticker} 当前价格：${prices.get(ticker, '未知')}"

@tool
def get_financial_indicators(ticker: str) -> str:
    """获取股票财务指标，包括 PE、PB、ROE 等"""
    # 模拟数据
    indicators = {
        "AAPL": "PE: 28.5, ROE: 45%, PB: 12.3, 营收增速: 15%",
        "GOOGL": "PE: 24.3, ROE: 23%, PB: 6.8, 营收增速: 12%",
    }
    return f"{ticker} 指标：{indicators.get(ticker, 'PE: 20, ROE: 15%, PB: 5.0, 营收增速: 10%')}"

@tool
def search_news(query: str) -> str:
    """搜索股票相关新闻。query 为搜索关键词"""
    news = {
        "AAPL": "1. iPhone 17 销量超预期 2. 苹果发布新 AI 战略 3. 伯克希尔增持苹果",
    }
    return f"关于 {query} 的新闻：{news.get(query, f'{query}相关新闻：无重大消息')}"

tools = [get_stock_price, get_financial_indicators, search_news]

# ============================================================
# 第2步：定义状态
# ============================================================

class StockAnalysisState(MessagesState):
    """股票分析状态 —— 继承 MessagesState 获得消息累加能力"""
    report: str       # 分析师报告
    decision: str     # 最终决策

# ============================================================
# 第3步：定义节点
# ============================================================

llm = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com/v1", temperature=0.1)
llm_with_tools = llm.bind_tools(tools)

def stock_analyst(state: StockAnalysisState) -> dict:
    """股票分析 Agent：调用工具获取数据并分析"""
    system_prompt = """你是一位专业的股票分析师。请根据用户的问题，使用可用工具获取数据并分析。
    如果需要多个工具，可以一次只调用一个，工具结果返回后继续分析。
    最终请用中文给出简洁的分析报告。"""

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    # 如果这次回复有文字内容（不是纯工具调用），保存为报告
    report = response.content if response.content and not response.tool_calls else ""
    return {"messages": [response], "report": state.get("report", "") + report}

def router(state: StockAnalysisState) -> str:
    """判断是否需要继续调工具"""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "done"

# ============================================================
# 第4步：建图
# ============================================================

workflow = StateGraph(StockAnalysisState)

workflow.add_node("analyst", stock_analyst)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "analyst")
workflow.add_conditional_edges("analyst", router, {
    "tools": "tools",
    "done": END,
})
workflow.add_edge("tools", "analyst")  # 工具执行完回到 analyst

graph = workflow.compile(checkpointer=MemorySaver())

# ============================================================
# 第5步：运行
# ============================================================

config = {"configurable": {"thread_id": "apple_analysis"}}

result = graph.invoke(
    {"messages": [HumanMessage("分析 AAPL 股票，查价格、指标和新闻，给我一个综合判断")]},
    config=config,
)

# 输出结果
print("=" * 50)
print("分析报告：")
print(result.get("report", "无"))
```

### 运行日志（模拟）

```
[analyst] LLM 收到问题 → 决定调 get_stock_price("AAPL")
[router]  检测到 tool_calls → 路由到 tools
[tools]   执行 get_stock_price → "AAPL 当前价格：$188.50"
[analyst] LLM 看到价格 → 决定调 get_financial_indicators("AAPL")
[router]  检测到 tool_calls → 路由到 tools
[tools]   执行 get_financial_indicators → "AAPL 指标：PE: 28.5..."
[analyst] LLM 看到指标 → 决定调 search_news("AAPL")
[router]  检测到 tool_calls → 路由到 tools
[tools]   执行 search_news → "关于 AAPL 的新闻：iPhone 17..."
[analyst] LLM 看到所有数据 → 生成分析报告
[router]  没有 tool_calls → 路由到 END
```

---

## 核心要点总结

| 概念 | 一句话解释 |
|------|----------|
| **LangChain** | LLM 开发的"标准库"，统一调用接口、工具绑定、结构化输出 |
| **LCEL** | `\|` 管道符，把 prompt / LLM / parser 串成流水线 |
| **RunnablePassthrough** | 透传数据不改动，`.assign()` 在 dict 上追加新字段 |
| **RunnableLambda** | 把普通函数包装成管道环节，融入 LCEL |
| **@tool** | 装饰器，把普通函数变成 LLM 可调用的工具 |
| **with_structured_output** | 让 LLM 输出 Pydantic 对象，不是纯文本 |
| **.stream()** | 逐 token 流式输出，LCEL chain 自动继承 |
| **RAG** | 检索增强生成：先查文档再回答，延伸 LLM 的知识边界 |
| **LangGraph** | 用"有向图"建模 Agent 工作流，自动管理状态 |
| **StateGraph** | 图 → 节点是步骤，边是流转方向 |
| **MessagesState** | 带消息累加器的状态基类，多轮对话自动追加 |
| **Annotated reducer** | `Annotated[类型, reducer]` 自定义每个字段的合并策略 |
| **条件边** | 路由函数 `(state) -> str` 决定下一步去哪 |
| **Send API** | 运行时动态生成 N 个并行节点实例，map-reduce 模式 |
| **ToolNode** | 自动解析 LLM 的 tool_calls 并执行对应的 `@tool` 函数 |
| **工具循环** | Agent ↔ ToolNode 循环，直到 LLM 不再要工具 |
| **with_fallbacks** | 自动切备用模型，LLM 挂了不崩 |
| **Checkpoint** | 自动保存状态，崩溃后从断点继续 |
| **stream_mode** | LangGraph 流式输出：values/updates/messages 三种视角 |

---

## 学习建议

1. **先跑起来**：复制上面的完整实战代码，换自己的 API Key 跑一遍，感受 Agent 工具循环
2. **改工具**：把 mock 工具换成真实 API，感受 LLM 如何自适应不同的工具
3. **加 Agent**：在图上多加一个节点（比如风险评估），感受多 Agent 协作
4. **加辩论**：实现 Bull/Bear 辩论循环，感受条件边的灵活性
5. **看源码**：LangGraph 源码很精简（核心不到 2000 行），值得一读

下一步可以阅读 [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/) 的 Tutorials 部分。
