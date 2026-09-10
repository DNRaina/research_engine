# AI Research Agent with FastMCP Server & Gemini RAG Memory

An intelligent, stateful **AI Research Agent** powered by **LangGraph**, **FastMCP (Model Context Protocol)**, **Gemini Embeddings**, and **Streamlit**.

The agent executes research tools directly through the **FastMCP Server** (`arxiv_search`, `wikipedia_search`, `pubmed_search`, `web_search`) and maintains a cross-session RAG (Retrieval-Augmented Generation) memory store using **FAISS** and free **Gemini Embeddings**.

---

## Key Features

- **Stateful LangGraph Workflow**: Three-node pipeline (`orchestrator -> retriever -> synthesiser`) built on `StateGraph`.
- **FastMCP Server & Academic Tools Integration**:
  - **FastMCP Server Tool Execution**: `app.py` routes research requests through `mcp.call_tool(...)` on the `FastMCP` server instance.
  - **arXiv Tool**: Searches peer-reviewed computer science, AI, and math papers with author lists, abstracts, and direct PDF download links.
  - **Wikipedia Tool**: Retrieves encyclopedia definitions, summaries, and article links.
  - **PubMed Tool**: Queries NCBI PubMed for biomedical literature and clinical research.
  - **DuckDuckGo Tool**: General web search context.
- **Cross-Session Gemini RAG Memory**:
  - Automatically exports chat turns as JSON files into `past_chats/`.
  - Indexes past research interactions using `GoogleGenerativeAIEmbeddings` (`models/text-embedding-004`) and `FAISS`.
  - Performs similarity searches over historical chats to provide cross-functional context awareness.
- **Query Rewriting** *(new)*:
  - Before routing to any search tool, the orchestrator node runs a lightweight LLM pass to rewrite the user's natural-language prompt into a concise, keyword-rich search query (max 12 words).
  - Significantly improves retrieval quality for conversational, vague, or verbose inputs.
  - Falls back transparently to the raw prompt if the API key is absent or the rewrite call fails.
  - The rewritten query is shown next to the routing status line in the UI.
- **Streaming LLM Responses** *(new)*:
  - After retrieval completes, the synthesiser response is streamed token-by-token directly into the Streamlit UI using `ChatGoogleGenerativeAI.stream()`.
  - The answer appears progressively, reducing perceived latency on long research reports.
  - Gracefully falls back to a full-batch response if streaming is unavailable.
- **Interactive Streamlit Interface**:
  - Real-time chat interface in `app.py`.
  - Sidebar model selection (`gemini-2.5-flash-lite` default, plus 2.0-flash, 1.5-flash, 1.5-pro) and toggles for individual FastMCP research tools.

---

## Project Architecture

```
research_agent/
├── run_me.py           # Bootstrap launcher: verifies dirs, syncs indices, launches UI
├── app.py              # Main Streamlit Application invoking tools via FastMCP Server
├── knowledge_base.py   # Persistent FAISS Knowledge Base with fingerprinting (books/)
├── retriever.py        # Past Chat JSON Persistence & Gemini FAISS Vector RAG Engine
├── mcp_server.py       # FastMCP Server exposing research tool endpoints
├── mcp_tools.py        # Academic API Tool implementations (arXiv, Wikipedia, PubMed)
├── books/              # Source PDFs / reference documents for local Knowledge Base
├── scans/              # Persisted FAISS vector index files and fingerprint caches
├── past_chats/         # Directory storing individual chat turns as JSON
├── past_sessions/      # Directory storing full conversation sessions
└── README.md           # Documentation
```

---

## Getting Started

### Prerequisites

Ensure you have Python 3.10+ installed.

### Installation

Install the required Python dependencies:

```bash
pip install streamlit langgraph langchain langchain-google-genai google-genai mcp ddgs faiss-cpu python-dotenv
```

### Running the Web Application

To automatically verify directories, pre-build or sync any pending FAISS knowledge base indices, and launch the application:

```bash
python run_me.py
```

Alternatively, you can launch Streamlit directly:

```bash
streamlit run app.py
```

### Running the Standalone FastMCP Server

You can also run the FastMCP server independently for external MCP clients via stdio:

```bash
python mcp_server.py
```

---

## Configuration & API Keys

Configure your API keys via a `.env` file in the project root:

```env
GEMINI_API_KEY="AIzaSy..."
```

- **`GOOGLE_API_KEY` / `GEMINI_API_KEY`**: Required for Gemini LLM synthesis, query rewriting, and Gemini Embeddings (`models/text-embedding-004`) for knowledge base and memory search.

If no API key is set, the application functions as a raw research collector — retrieving and displaying paper summaries, Wikipedia facts, and web results directly, without LLM synthesis.