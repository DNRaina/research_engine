# JANE — Production-Grade AI Research Agent

An enterprise-ready, stateful **AI Research Agent** (Just A Nuanced Engine) built with **LangGraph**, **FastMCP (Model Context Protocol)**, **Gemini Models & Embeddings**, **HTTPX**, and **Streamlit**.

JANE routes user queries through heuristic orchestration, invokes specialized academic and web research tools in parallel via a resilient FastMCP server, incrementally indexes conversation memory in **FAISS**, and synthesizes rigorous research dossiers with single-pass token streaming.

---

## Key Production Engineering Features

- **Single-Pass Real-Time Streaming**:
  - Eliminates duplicate LLM calls by streaming tokens directly into the UI in one unified pass.
  - Slashes end-to-end turnaround latency and Gemini API token costs by **~50%**.
- **Resilient FastMCP Server & Concurrency Architecture**:
  - Executes tools via a shared, bounded daemon worker pool (`ThreadPoolExecutor(max_workers=16)`).
  - True non-blocking timeouts that do not freeze calling threads upon expiry.
  - Exponential backoff with random jitter to prevent API stampedes and gracefully handle HTTP 429/rate limits.
- **Modern Resilient HTTP Networking Layer**:
  - Upgraded from raw `urllib` to `httpx.Client` featuring HTTP keep-alive connection pooling.
  - Enforced HTTPS on arXiv (`https://export.arxiv.org/api/query`) with robot-policy-compliant User-Agent headers (`JANEResearchAgent/2.0`), eliminating 403 blocks from Wikipedia and redirect timeouts.
- **Incremental FAISS Memory Indexing**:
  - Tracks conversation manifests (`memory_manifest.json`) measuring session modification times and turn counts.
  - Skips re-indexing entirely if no chats have changed (**0ms overhead, 0 API calls**).
  - On new turns, only embeds and appends the delta (`add_documents`), preventing quadratic latency degradation and quota exhaustion.
- **Execution Telemetry & Observability**:
  - Structured, timestamped logging across all modules via `logger.py`.
  - Live per-turn telemetry in the UI (`⏱️ Total: 2.1s | Orchestration: 80ms | Retrieval: 1.1s | Synthesis: 0.9s`).
  - Per-source response duration and retrieval status badges.
- **Production Healthcheck & Diagnostics CLI**:
  - Built-in `health.py` diagnostic suite probing storage paths, Gemini API authentication and response latency, and connectivity to all 4 external research tools.
  - Available via CLI (`python run_me.py --check`) or interactively in Streamlit's sidebar.
- **Centralized Typed Configuration**:
  - Managed via `config.py` using Pydantic Settings, providing a single source of truth for paths, models, timeouts, retry limits, and credentials.
- **Automated Test Suite**:
  - 100% passing automated unit tests (`tests/`) covering configuration, resilience wrappers, tool sanitizers, and session lifecycles.

---

## Research Tools Registered

| Tool | Source | Details |
| :--- | :--- | :--- |
| `arxiv_search` | arXiv Atom API (HTTPS) | Academic preprints in CS, AI, physics, and math with direct PDF links. |
| `wikipedia_search` | Wikipedia REST API | Encyclopedic summaries, entity definitions, and reference URLs. |
| `pubmed_search` | NCBI PubMed eUtils API | Biomedical literature, clinical trials, and life sciences citations. |
| `web_search` | DuckDuckGo Search | Real-time web results and news snippets. |
| `knowledge_base_search` | Local FAISS Index | Semantic search over local documents (`books/` `.pdf`, `.txt`, `.md`). |
| `memory_search` | Past Chat FAISS Index | Cross-session semantic recall over historical user research conversations. |

---

## Project Structure

```
research_agent/
├── config.py           # Centralized typed settings & path configurations (Pydantic)
├── logger.py           # Unified structured logging with timestamps and log levels
├── health.py           # Production health check & diagnostics CLI
├── run_me.py           # Production bootstrap launcher & preflight checks
├── app.py              # Streamlit interface with single-pass streaming & telemetry
├── mcp_server.py       # FastMCP Server with non-blocking timeouts & backoff retry
├── mcp_tools.py        # HTTPX-powered research tool clients (arXiv, Wiki, PubMed, Web)
├── knowledge_base.py   # Persistent FAISS Knowledge Base with fingerprint caching (books/)
├── retriever.py        # Incremental conversational FAISS memory & dossier exports
├── books/              # Source PDFs / reference documents for local Knowledge Base
├── scans/              # Persisted FAISS vector stores & manifest fingerprints
├── past_chats/         # Session persistence directory (JSON files)
├── tests/              # Automated unit test suite
│   ├── test_config.py
│   ├── test_resilience.py
│   ├── test_tools.py
│   └── test_retriever.py
└── README.md           # System documentation
```

---

## Quick Start

### 1. Environment Configuration

Create or update your `.env` file with your Gemini API key:

```env
GEMINI_API_KEY="AIzaSy..."
```

*(Note: `GOOGLE_API_KEY` is also supported as an alias.)*

### 2. Pre-Flight Diagnostics Check

Run the system diagnostics probe to ensure all APIs and storage permissions are operational:

```bash
python run_me.py --check
```

### 3. Launch the Application

Start the application with pre-flight index verification:

```bash
python run_me.py
```

Or launch Streamlit directly:

```bash
streamlit run app.py
```

### 4. Run Automated Tests

Execute the unit test suite:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### 5. Standalone FastMCP Server

To expose the research tools to external MCP-compatible agents via stdio:

```bash
python mcp_server.py
```