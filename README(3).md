# AGNET — LangGraph Agent with Neo4j Memory and Self-Correction

An autonomous AI agent built with LangGraph and LangChain that answers multi-domain queries by selecting and executing the right tool at runtime, evaluating its own output with a separate LLM judge, and looping back to correct itself if the answer does not meet the quality threshold. All conversation state is persisted to a Neo4j graph database, enabling the agent to retrieve and reason over its own past conversations as a native tool call.

---

## What this project demonstrates

- Graph-based agent orchestration with LangGraph `StateGraph`
- LLM-as-judge self-correction loop (score, critique, reformulate, retry)
- Four-tool runtime dispatch: web search, weather API, symbolic math, and conversation memory retrieval
- Persistent conversation memory stored in Neo4j and queryable by the agent itself via Cypher
- Multi-model architecture distributing load across four independent Groq API keys
- A structured evaluation framework with 30 test cases measuring tool precision, content accuracy, and reformulation recovery rate

---

## Repository Structure

```
AGNET/
├── eval/
│   ├── test_cases.json      # 30 structured test cases
│   └── results.json         # generated output from eval.py
├── my_agent.py              # agent graph, tools, Neo4j checkpointer
├── eval.py                  # evaluation runner and metrics
├── requirements.txt
└── README.md
```

---

## Architecture

### The LangGraph State Graph

The agent is built as a cyclic `StateGraph` with three nodes and one conditional routing function. The graph is compiled once at startup with a Neo4j checkpointer attached, so every state transition is durably persisted to the database.

```
START
  │
  ▼
llm_call <──────────────────────────────────────────┐
  │                                                 │
  ▼                                                 │
validation_node                                     │
  │                                                 │
  ├── score >= 7.5 OR retry_count >= 2 ──► END      │
  │                                                 │
  └── score < 7.5 AND retry_count < 2               │
        │                                           │
        ▼                                           │
  reformulation_node ────────────────────────────── ┘
```

The typed state that flows through every node:

```python
class AgentState(TypedDict):
    query: str           # original user query, never mutated
    current_query: str   # reformulated query for the current attempt
    messages: str        # raw answer from the current attempt
    best_answer: str     # highest-scoring answer seen so far
    score: float         # validator score for the current attempt
    best_score: float    # highest score seen across all attempts
    critique: str        # validator's explanation of what is wrong
    tool_used: str       # tool name from the current attempt
    passed: bool         # whether validator marked answer as passing
    retry_count: int     # number of reformulation cycles completed
```

Every node returns a partial dict. LangGraph merges it into the existing state — only the returned keys are updated, the rest are preserved. This is why `best_answer` and `best_score` survive across retry cycles: the validation node only overwrites them when it finds a strictly better score.

### Why LangGraph and not a Python while loop

A plain `while` loop could implement the retry logic. The reason it was not used here comes down to three things:

**State typing and enforcement.** The `AgentState` `TypedDict` is declared once and enforced at the graph level. Every node is contractually limited to reading what is in state and returning a dict of known keys. A `while` loop has none of this — state lives in local variables with no structural boundary between stages.

**Durable checkpointing.** The graph is compiled with `checkpointer=Neo4jSaver(driver)`. Every time the graph transitions between nodes, LangGraph serializes the full `AgentState` to Neo4j under the `thread_id` of the current session. If the process crashes mid-loop, the next invocation with the same `thread_id` resumes from the last completed node. A `while` loop loses everything on crash.

**Inspectable routing.** The routing logic is isolated in `route_after_validation`, which returns a string key (`"to_end"` or `"to_reformulation"`). LangGraph maps those keys to concrete node transitions at compile time via `add_conditional_edges`. The control graph is statically defined and can be visualized. In a `while` loop, branching logic is scattered across the loop body.

### The Self-Correction Loop

**Attempt 1 — `llm_call`**

The primary LLM (`llama-3.3-70b-versatile`, key 1) receives `current_query` wrapped in a `HumanMessage`. The model decides whether to call a tool or answer from its own knowledge. If it emits tool calls, a `ToolNode` executes the requested tool and returns a `ToolMessage`. The full message chain — `HumanMessage`, `AIMessage` with tool_calls, `ToolMessage` with result — is then passed to a dedicated synthesizer LLM (key 2), which produces the plain-text final answer. The tool name and synthesized answer are written into state.

**Evaluation — `validation_node`**

A third LLM instance (key 4) receives the original query, the tool used, and the answer. It is prompted to respond only in JSON:

```json
{"score": 8.5, "passed": true, "critique": "Answer is complete and grounded in tool output."}
```

`passed` is `true` when `score >= 7.5`. The validator also tracks the best answer seen across attempts: if the current score exceeds `best_score`, it overwrites `best_answer`. This means even if the second attempt scores lower than the first, the best answer is preserved and returned at `END`.

**Reformulation — `reformulation_node`**

If `passed` is `false` and `retry_count < 2`, a fourth LLM instance (key 3) receives the original query, the failed answer, and the critique. It rewrites `current_query` to be more specific and correctable. `retry_count` is incremented. Control returns to `llm_call` with the new query.

The agent makes at most 3 attempts (initial + 2 reformulations). After that, `route_after_validation` routes to `END` regardless of score, returning `best_answer`.

### Four-Model Architecture

Four separate `ChatGroq` instances are initialized, each with its own API key:

| Role | Model | Key |
|---|---|---|
| Tool-bound primary LLM | llama-3.3-70b-versatile | groq_api_key |
| Answer synthesizer | llama-3.3-70b-versatile | groq_api_key_2 |
| Query reformulator | llama-3.3-70b-versatile | groq_api_key_3 |
| Answer validator / judge | llama-3.3-70b-versatile | groq_api_key_4 |

Each role has its own dedicated key because the self-correction loop calls all four models sequentially within a single user turn. Sharing one key would exhaust Groq's free-tier tokens-per-minute limit mid-run. Isolating each role to its own key keeps each bucket independent.

---

## Neo4j: Persistent Conversation Memory

Neo4j serves two distinct roles in this system.

### Role 1 — LangGraph Checkpointer

The graph is compiled with a `Neo4jSaver` checkpointer:

```python
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
checkpointer = Neo4jSaver(driver, database=NEO4J_DATABASE)
checkpointer.setup()
graph = builder.compile(checkpointer=checkpointer)
```

`Neo4jSaver` is LangGraph's native Neo4j persistence backend. On every node transition, it serializes the full `AgentState` into the graph database under the active `thread_id`. The data model it creates is:

```
(:Thread {thread_id}) -[:HAS_CHANNEL]-> (:ChannelState {channel, blob, version})
```

Each `ChannelState` node stores a serialized snapshot of one state channel (e.g., `messages`, `best_answer`, `tool_used`) at a specific version. This gives the agent a complete, versioned history of every state it has been in, per session.

Every conversation session is identified by a unique `thread_id` passed in `config = {"configurable": {"thread_id": "..."}}`. The agent can resume any prior session by reusing the same `thread_id`.

### Role 2 — Agent Memory Tool

The agent can actively query its own conversation history using the `query_conversation_graph` tool:

```python
@tool
def query_conversation_graph(entity: str) -> str:
    """
    Search past conversations stored in memory by a keyword or topic.
    Use this when the user asks about previous questions, past searches, or conversation history.
    """
    with GraphDatabase.driver(...).session(...) as session:
        result = session.run("""
            MATCH (t:Thread)-[:HAS_CHANNEL]->(cs:ChannelState)
            WHERE cs.channel = 'messages'
            AND toLower(cs.blob) CONTAINS toLower($entity)
            RETURN t.thread_id, cs.channel, cs.blob
            ORDER BY cs.version DESC
            LIMIT 5
        """, entity=entity)
```

The Cypher query traverses the `Thread → HAS_CHANNEL → ChannelState` graph, filters `ChannelState` nodes on the `messages` channel where the serialized blob contains the search term, and returns the five most recent matches ordered by version.

This means the agent can answer questions like "what did I ask about Paris last time?" or "what weather queries have I made?" by querying the same database that LangGraph uses for checkpointing. The memory system is not a separate vector store — it is the checkpoint store itself, queried directly via Cypher.

---

## Tools

The LLM selects one tool per turn based on the content of `current_query`. Tool selection is handled by the model's built-in function-calling capability (`llm.bind_tools(my_tools)`). The tool call is detected via `response.tool_calls` on the returned `AIMessage`.

| Tool | Source | What it does |
|---|---|---|
| `TavilySearch` | LangChain / Tavily API | Real-time web search, top 3 results |
| `weather_check` | OpenWeatherMap REST API | Current temperature and conditions by city |
| `universal_math_solver` | Wolfram\|Alpha XML API | Symbolic math, calculus, equation solving, LaTeX input |
| `query_conversation_graph` | Direct Cypher on Neo4j | Retrieves past conversations matching a keyword |

**`universal_math_solver` — extraction logic**

Wolfram returns a structured `pods` response. The tool iterates over pods and returns the primary-flagged pod immediately if one exists. If no pod is flagged as primary (which happens on some query types), it collects all subpods with plaintext content and returns the first three. This handles the variability in Wolfram's response structure without hardcoding expected pod titles.

---

## Evaluation Framework

`eval.py` runs all 30 test cases sequentially, measuring the agent on three dimensions:

**Tool-calling precision** — whether the agent called the tool specified in `expected_tool` for each case.

**Content correctness** — whether all strings in `expected_contains` appear in the lowercased final answer.

**Pass condition** — `tool_correct AND content_correct`.

### Test case schema

```json
  {
    "id": 101,
    "query": "What is the weather in London right now?",
    "expected_tool": "weather_check",
    "expected_contains": ["temperature", "London"],
    "category": "weather"
  }
```

### Memory test threading

Cases with `"category": "memory"` all share a single persistent `thread_id` (`eval_memory_thread`). This is intentional — memory tests need to run on a session that already has prior conversation history in Neo4j, so they chain sequentially on the same thread rather than starting from an empty state.

All other categories receive isolated `thread_id`s per test case to prevent cross-contamination.

### Summary metrics reported

```
By category:
  weather               6/6 (100%)
  math                  7/8 (88%)
  search                6/6 (100%)
  memory                3/3 (100%)
  general_knowledge     3/3 (100%)
  edge_case             1/2 (50%)
  multi_intent          0/1 (0%)
  math_simple           2/2 (100%)

Reformulation recovery rate: 4/6 cases that retried ended up passing
Average best_score: 8.3/10
```

### Rate limiting

The eval runner applies a 30-second sleep before each test and an additional 60-second sleep after non-memory tests. This is a hard requirement of the Groq free tier, which enforces strict tokens-per-minute limits. Memory tests are exempt from the post-test sleep because they chain sequentially and the inter-test sleep is sufficient.

Run the full evaluation suite:

```bash
python eval.py
```

Results are written to `eval/results.json`.

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent framework | LangGraph `StateGraph` |
| LLM inference | Groq (llama-3.3-70b-versatile) |
| LLM orchestration | LangChain Core |
| Conversation memory | Neo4j + `langchain_neo4j.Neo4jSaver` |
| Web search | Tavily (`langchain_tavily`) |
| Weather | OpenWeatherMap REST API |
| Symbolic math | Wolfram\|Alpha Python client |
| Environment config | `python-dotenv` |


# AGNET Setup Guide

## Prerequisites

- Python 3.10 or higher
- A running Neo4j instance (local Desktop or AuraDB cloud)
- API accounts for: Groq (x4 keys), Tavily, OpenWeatherMap, Wolfram|Alpha

---

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/AGNET.git
cd AGNET
```

---

## 2. Create and activate a virtual environment

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure environment variables

Create a `.env` file at the root of the project and populate it with your credentials:

```dotenv
# Groq — four separate keys, one per model role
groq_api_key=<primary-llm-key>
groq_api_key_2=<synthesizer-key>
groq_api_key_3=<reformulator-key>
groq_api_key_4=<validator-key>

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your-password>
NEO4J_DATABASE=neo4j

# External APIs
tavily_api_key=<your-tavily-key>
wheather_api_key=<your-openweathermap-key>
appid=<your-wolframalpha-appid>
```



### Where to get each key

| Variable | Provider | URL |
|---|---|---|
| `groq_api_key` x4 | Groq Console | https://console.groq.com |
| `tavily_api_key` | Tavily | https://app.tavily.com |
| `wheather_api_key` | OpenWeatherMap | https://openweathermap.org/api |
| `appid` | Wolfram\|Alpha Developer | https://developer.wolframalpha.com |
| Neo4j credentials | Neo4j AuraDB or local Desktop | https://neo4j.com |

> Four separate Groq keys are required because the self-correction loop calls four model instances sequentially within a single turn. One key would exhaust the free-tier tokens-per-minute limit mid-run.

---

## 5. Neo4j — database setup

The `Neo4jSaver` checkpointer calls `checkpointer.setup()` automatically on first run, which creates the required constraints and indexes in your Neo4j database. No manual schema setup is needed.

If using **Neo4j Desktop**, ensure the database is running before launching the agent.

If using **Neo4j AuraDB**, set `NEO4J_URI` to your instance's connection URI (format: `neo4j+s://<instance-id>.databases.neo4j.io`).

---

## 6. Run the agent

```bash
python my_agent.py
```

---

## 7. Run the evaluation suite

```bash
python eval.py
```

Results are written to `eval/results.json` on completion. The full run takes approximately 45–60 minutes due to rate-limiting sleeps between test cases required by the Groq free tier.
