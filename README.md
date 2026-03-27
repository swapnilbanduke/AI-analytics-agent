# AI Data Analyst Assistant

A conversational AI agent that answers any question about your data by choosing the right approach — query a database, search documents, fetch live info from the web — and deliver a polished answer with charts.

Built incrementally across **10 layers**, each adding a new capability while keeping everything before it working.

---

## Architecture Overview

```
User Question
     │
     ▼
┌─────────────┐
│  Classifier  │  ← LLM decides the best route
└──────┬──────┘
       │
       ├── calculation ──→ 🔢 Calculator (safe math eval)
       ├── web_search ───→ 🌐 Tavily Web Search
       ├── sql ──────────→ 🗄️ SQL Agent (NL → SQL → Execute → Summarize)
       ├── document ─────→ 📄 RAG Agent (ChromaDB vector search → LLM answer)
       └── direct ───────→ 💬 Direct LLM response
              │
              ▼
       ┌────────────┐
       │ Synthesizer │  ← Combines multi-agent outputs
       └──────┬─────┘
              │
              ▼
       ┌────────────┐
       │  Reflexion  │  ← Scores answer quality, retries if poor
       └──────┬─────┘
              │
              ▼
         Final Answer
```

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| **Agent Framework** | LangChain + LangGraph (StateGraph) |
| **LLM Providers** | OpenAI (GPT-4.1, GPT-4o) + Anthropic (Claude Sonnet 4, Opus 4) |
| **Frontend** | Streamlit |
| **Database** | SQLite via SQLAlchemy |
| **Vector Store** | ChromaDB (persistent, local) |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **Web Search** | Tavily Search API |
| **MCP** | FastMCP (Model Context Protocol) |
| **API** | FastAPI + Uvicorn |
| **Deployment** | Docker + Docker Compose |
| **Observability** | LangSmith tracing |

---

## Project Structure

```
AI Data Analytics/
│
├── app.py                 # Streamlit frontend (chat UI, sidebar, uploads)
├── config.py              # Multi-provider model catalog, API key resolution, LangSmith config
├── state.py               # AgentState TypedDict — all fields for the graph
├── graph.py               # LangGraph StateGraph: classify → agent ↔ tools → END
├── tools.py               # @tool definitions: calculator, web_search, sql_query, document_search
├── prompts.py             # All prompt templates organized by layer/domain
├── styles.py              # Streamlit custom CSS (dark blue theme, route badges)
│
├── database.py            # SQLAlchemy engine, run_query(), list_tables()
├── ingestion.py           # CSV/Excel → SQLite ingestion with clean_name()
├── sql_agent.py           # NL2SQL pipeline: generate → validate → execute → summarize
│
├── rag.py                 # Document loading, chunking, ChromaDB vector store, retrieval
│
├── agents.py              # Multi-agent: planner, SQL/RAG/web sub-agents, synthesizer
├── memory.py              # ConversationMemory + Reflexion quality evaluation
│
├── mcp_server.py          # FastMCP server exposing tools via MCP protocol
├── api.py                 # FastAPI REST server (/query, /upload, /health)
├── auth.py                # Bearer token authentication
│
├── Dockerfile             # Python 3.11-slim container
├── docker-compose.yml     # API + Streamlit frontend services
│
├── requirements.txt       # All dependencies organized by layer
├── .env.example           # Environment variable template
├── .gitignore             # Python/DB/IDE ignores
│
├── data/                  # Uploaded data + SQLite database
│   ├── sample_sales.csv   # 500-row sample dataset for testing
│   └── app.db             # Auto-created SQLite database (gitignored)
│
├── documents/             # Uploaded PDFs/docs for RAG
│
└── tests/                 # Test files
```

---

## Layer-by-Layer Breakdown

### Layer 1 — LangChain Basics + Tool Calling

**What it does:** A simple conversational agent that can call tools (calculator, web search) to answer questions.

**How it works:**
- User asks a question in the Streamlit chat UI
- The LLM receives the question along with tool definitions
- If the LLM decides a tool is needed, it generates a tool call
- The tool executes and returns results to the LLM
- The LLM formulates a final answer incorporating tool results

**Key files:**
- `tools.py` — `calculator()` uses Python's AST for safe math evaluation (no `eval()`). `web_search()` uses Tavily's API.
- `config.py` — `PROVIDER_CATALOG` dict defines OpenAI and Anthropic models. `build_chat_model()` creates the right LangChain chat model based on provider.
- `app.py` — Streamlit chat interface with provider/model selection in the sidebar.

**Skills learned:** LangChain tool calling, `@tool` decorator, prompt engineering, ReAct pattern.

---

### Layer 2 — LangGraph Agent with Routing

**What it does:** Adds intelligent routing — the agent classifies each question and picks the best tool category before acting.

**How it works:**
```
User Question → [Classify Node] → route decision → [Agent Node] ↔ [Tool Node] → Answer
```

- The `classify_question()` node sends the question to an LLM with a classification prompt
- The LLM returns one of: `calculation`, `web_search`, `sql`, `document`, `direct`
- The graph routes to the `agent` node which has all relevant tools bound
- The agent ↔ tools loop continues until the LLM stops calling tools

**Key files:**
- `graph.py` — `build_agent_graph()` creates a `StateGraph` with nodes and conditional edges
- `state.py` — `AgentState(TypedDict)` carries all state through the graph (messages, route, provider, etc.)
- `prompts.py` — `CLASSIFICATION_PROMPT` tells the LLM how to classify questions

**Skills learned:** LangGraph `StateGraph`, `TypedDict` state, conditional edges, node functions, graph compilation.

---

### Layer 3 — SQL Agent Tool

**What it does:** Users upload CSV/Excel files, which get ingested into SQLite. The agent converts natural language questions into SQL, executes them, and summarizes the results.

**How it works:**
```
Upload CSV → SQLite table → User asks question →
  [Get Schema] → [Generate SQL] → [Validate] → [Execute] → [Summarize Answer]
```

1. **Ingestion** (`ingestion.py`): CSV/Excel files are read with pandas, column names are cleaned for SQL compatibility (`clean_name()` handles spaces, special chars, reserved words), and data is written to SQLite via `df.to_sql()`.

2. **SQL Pipeline** (`sql_agent.py`):
   - Retrieves database schema via LangChain's `SQLDatabase`
   - Sends schema + question to LLM with `SQL_GENERATION_PROMPT`
   - Validates generated SQL: only SELECT allowed, forbidden keywords blocked (`INSERT`, `DELETE`, `DROP`, etc.)
   - Enforces `LIMIT 100` on all queries
   - Executes against SQLite and captures results
   - Sends results back to LLM for natural language summarization
   - **Retry logic**: if validation or execution fails, retries once with the error message in the prompt

3. **Safety** (`is_sql_safe()`): Blocks any non-SELECT query, strips SQL comments, checks for 13 forbidden keywords.

**Key files:**
- `database.py` — `get_engine()`, `run_query()`, `list_tables()` (SQLAlchemy wrapper)
- `ingestion.py` — `ingest_csv()`, `ingest_excel()`, `clean_name()` with `SQL_RESERVED_WORDS`
- `sql_agent.py` — `run_sql_pipeline()` end-to-end function
- `data/sample_sales.csv` — 500 rows with: order_id, order_date, customer_name, customer_segment, product_name, category, region, city, state, sales, quantity, discount, profit

**Skills learned:** NL2SQL, SQLAlchemy, LangChain `SQLDatabase`, prompt engineering for SQL, error recovery loops.

---

### Layer 4 — RAG Tool for Documents

**What it does:** Users upload PDF/text documents. They are chunked, embedded, and stored in ChromaDB. The agent retrieves relevant chunks to answer document-based questions.

**How it works:**
```
Upload PDF → [PyPDF Loader] → [Text Splitter] → [Embeddings] → ChromaDB
                                                                    │
User Question → [Similarity Search] → Top 4 chunks → [LLM Answer]
```

1. **Document Loading**: `PyPDFLoader` (PDFs) or `TextLoader` (txt/md)
2. **Chunking**: `RecursiveCharacterTextSplitter` with 1000-char chunks, 200-char overlap
3. **Embedding**: OpenAI `text-embedding-3-small` model
4. **Storage**: ChromaDB with persistent local storage at `data/vectorstore/`
5. **Retrieval**: Similarity search returns top-4 most relevant chunks
6. **Answer Generation**: LLM answers using `RAG_ANSWER_PROMPT` which instructs it to base answers strictly on retrieved excerpts

**Key files:**
- `rag.py` — `add_documents()`, `search_documents()`, `load_document()`, `has_documents()`
- `prompts.py` — `RAG_ANSWER_PROMPT` with grounding instructions
- `tools.py` — `document_search()` tool wrapping the RAG pipeline

**Skills learned:** RAG pipeline, embeddings, vector stores, ChromaDB, document parsing, retrieval-augmented generation.

---

### Layer 5 — MCP Integration

**What it does:** Exposes all agent tools as an MCP (Model Context Protocol) server, so any MCP-compatible client (Claude Desktop, other agents) can connect and use them.

**How it works:**
```
MCP Client ──stdio──→ [FastMCP Server]
                          ├── calculate()
                          ├── query_database()
                          ├── search_documents()
                          ├── web_search()
                          ├── list_data_tables()
                          └── Resource: schema://database
```

- Built with `FastMCP` — a high-level MCP server builder
- Each tool is decorated with `@mcp.tool()` and delegates to the corresponding Python function
- Also exposes a `schema://database` resource for schema introspection
- Runs over `stdio` transport (standard for MCP)

**Key files:**
- `mcp_server.py` — Complete MCP server with 5 tools + 1 resource

**How to test:**
```bash
python mcp_server.py                                          # Run server
npx @modelcontextprotocol/inspector python mcp_server.py      # Test with inspector
```

**Skills learned:** MCP server/client, FastMCP, tool connectivity standard, stdio transport.

---

### Layer 6 — Multi-Agent Architecture

**What it does:** Instead of one agent doing everything, specialized sub-agents handle different tasks, coordinated by a supervisor (planner).

**How it works:**
```
User Question → [Planner]
                    │
                    ├── "sql" ────→ [SQL Sub-Agent] ──┐
                    ├── "rag" ────→ [RAG Sub-Agent] ──┼──→ [Synthesizer] → Answer
                    ├── "web" ────→ [Web Sub-Agent] ──┘
                    └── "direct" ─→ [Synthesizer] → Answer
```

1. **Planner** (`plan_action()`): LLM receives the question + context (what data/docs are available) and returns a JSON plan: `{"agents": ["sql", "rag"], "reasoning": "..."}`
2. **Sub-Agents**: Each runs independently — SQL agent queries the database, RAG agent searches documents, Web agent searches the internet
3. **Synthesizer** (`synthesize_results()`): If multiple agents ran, the LLM merges their outputs into one coherent answer. If only one ran, passes through directly.

**Key files:**
- `agents.py` — `plan_action()`, `run_sql_sub_agent()`, `run_rag_sub_agent()`, `run_web_sub_agent()`, `synthesize_results()`, `route_plan()`
- `prompts.py` — `PLANNING_PROMPT` and `SYNTHESIS_PROMPT`

**Skills learned:** Multi-agent orchestration, supervisor pattern, message passing, result synthesis.

---

### Layer 7 — Memory + Self-Correction

**What it does:** The agent remembers previous questions in a conversation and evaluates its own answer quality, retrying if the answer is poor (Reflexion pattern).

**How it works:**

**Memory:**
- `ConversationMemory` class stores question-answer turns (max 20)
- Recent turns (last 10) are injected into the LLM context as message history
- Older turns can be summarized for compression

**Reflexion (Self-Correction):**
```
Answer → [Quality Evaluator] → Score 1-10
                                   │
                                   ├── Score >= 6 → Done ✓
                                   └── Score < 6  → Retry (max 2 retries)
```

- The LLM scores its own answer on: relevance (0-3), accuracy (0-3), completeness (0-2), clarity (0-2)
- If score < 6 and retry_count < 2, the answer is cleared and the pipeline re-runs
- The reflection (what to improve) is passed to the next attempt

**Key files:**
- `memory.py` — `ConversationMemory` class + `evaluate_answer_quality()` + `should_retry()`
- `prompts.py` — `QUALITY_EVAL_PROMPT` with scoring rubric

**Skills learned:** Conversation memory, context window management, Reflexion pattern, self-evaluation, quality scoring.

---

### Layer 8 — Human-in-the-Loop

**What it does:** The agent pauses before executing potentially dangerous or complex SQL queries and waits for human approval.

**How it works:**
```
Generated SQL → [classify_sql_sensitivity()]
                        │
                        ├── "safe"    → Execute immediately
                        ├── "review"  → Pause, show SQL, wait for approval
                        └── "blocked" → Reject entirely
```

**Sensitivity levels:**
- **Safe**: Simple SELECT queries → auto-execute
- **Review**: Queries with JOINs, subqueries, UNION, HAVING, CASE, or nested SELECTs → pause for approval
- **Blocked**: Any DML (INSERT, UPDATE, DELETE, DROP, ALTER, etc.) → rejected with error

**Key files:**
- `sql_agent.py` — `classify_sql_sensitivity()` function with `REVIEW_KEYWORDS`
- `state.py` — `requires_approval`, `human_approved`, `pending_sql` fields

**Skills learned:** LangGraph interrupt/resume, workflow persistence, approval flows, SQL sensitivity classification.

---

### Layer 9 — Streaming + Observability

**What it does:** Adds LangSmith tracing for full pipeline observability — every LLM call, tool invocation, and routing decision is logged and viewable in the LangSmith dashboard.

**How it works:**
- When `LANGCHAIN_TRACING_V2=true` is set in `.env`, all LangChain/LangGraph operations are automatically traced
- Traces include: input/output of every node, token counts, latency, tool call details
- Viewable at [smith.langchain.com](https://smith.langchain.com)

**Configuration** (in `.env`):
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2-...
LANGCHAIN_PROJECT=ai-data-analyst
```

**Key files:**
- `config.py` — Auto-configures LangSmith environment variables on startup

**Skills learned:** LangSmith tracing, debugging LLM chains, latency monitoring, token usage tracking.

---

### Layer 10 — Deployment

**What it does:** Wraps everything in a production-ready FastAPI backend with authentication, containerized with Docker.

**Components:**

1. **FastAPI Server** (`api.py`):
   | Endpoint | Method | Description |
   |----------|--------|-------------|
   | `/health` | GET | Health check + list tables + document status |
   | `/query` | POST | Ask a question (returns answer, route, SQL) |
   | `/upload/data` | POST | Upload CSV/Excel → ingest to SQLite |
   | `/upload/document` | POST | Upload PDF/text → index in ChromaDB |
   | `/tables` | GET | List all database tables |
   | `/schema` | GET | Get full database schema |

2. **Authentication** (`auth.py`):
   - Bearer token auth via `API_AUTH_TOKEN` env var
   - Disabled in dev mode (when token is not set)
   - Returns 401 for invalid/missing tokens

3. **Docker**:
   - `Dockerfile`: Python 3.11-slim, installs deps, runs uvicorn
   - `docker-compose.yml`: Two services — `api` (port 8000) + `frontend` (port 8501), shared volumes for data persistence

**How to run:**
```bash
# API only
uvicorn api:app --reload

# Full stack with Docker
docker-compose up --build
```

**Key files:**
- `api.py` — FastAPI app with Pydantic request/response models
- `auth.py` — `verify_token()` dependency for protected endpoints
- `Dockerfile`, `docker-compose.yml`

**Skills learned:** FastAPI, REST API design, Pydantic models, Docker, Docker Compose, production deployment, API authentication.

---

## Quick Start

### 1. Install dependencies
```bash
cd "AI Data Analytics"
pip install -r requirements.txt
```

### 2. Configure API keys
```bash
cp .env.example .env
# Edit .env and add your keys:
#   OPENAI_API_KEY=sk-...
#   TAVILY_API_KEY=tvly-...       (optional, for web search)
#   ANTHROPIC_API_KEY=sk-ant-...  (optional, for Claude models)
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

### 4. Try it out
- **Calculator**: "What is 15% of 2847?"
- **Web Search**: "What's the latest news about AI?"
- **SQL**: Upload `data/sample_sales.csv` → "Top 5 products by total sales?"
- **Documents**: Upload a PDF → "Summarize the key findings"

---

## Running Individual Components

### MCP Server
```bash
python mcp_server.py
# Test with inspector:
npx @modelcontextprotocol/inspector python mcp_server.py
```

### FastAPI Server
```bash
uvicorn api:app --reload
# Test:
curl http://localhost:8000/health
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is 2 + 2?"}'
```

### Docker
```bash
docker-compose up --build
# API: http://localhost:8000
# Frontend: http://localhost:8501
```

---

## Sample Dataset

`data/sample_sales.csv` contains 500 rows of synthetic sales data:

| Column | Type | Example |
|--------|------|---------|
| order_id | string | ORD-0001 |
| order_date | date | 2024-03-15 |
| customer_name | string | Alice Johnson |
| customer_segment | category | Consumer, Corporate, Home Office |
| product_name | string | Laptop, Chair, Paper |
| category | category | Technology, Furniture, Office Supplies |
| region | category | West, East, Central, South |
| city | string | Los Angeles, New York, Chicago |
| state | string | California, New York, Illinois |
| sales | float | 1,234.56 |
| quantity | int | 1-15 |
| discount | float | 0.00 - 0.30 |
| profit | float | -200 to 800 (can be negative) |

**Example questions to ask:**
- "How many orders are there?"
- "What are the top 5 products by total sales?"
- "Average profit by region?"
- "Which customer segment generates the most revenue?"
- "Show monthly sales trends for 2024"
- "What's the most profitable category?"

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT models + embeddings |
| `ANTHROPIC_API_KEY` | No | Anthropic API key for Claude models |
| `TAVILY_API_KEY` | No | Tavily API key for web search |
| `LANGCHAIN_TRACING_V2` | No | Set to `true` for LangSmith tracing |
| `LANGCHAIN_API_KEY` | No | LangSmith API key |
| `LANGCHAIN_PROJECT` | No | LangSmith project name (default: `ai-data-analyst`) |
| `API_AUTH_TOKEN` | No | Bearer token for FastAPI auth (empty = no auth) |
