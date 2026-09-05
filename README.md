# AI Research Agent with FastMCP Server & Gemini RAG Memory

An intelligent, stateful **AI Research Agent** powered by **LangGraph**, **FastMCP (Model Context Protocol)**, **Gemini Embeddings**, and **Streamlit**.

The agent executes research tools directly through the **FastMCP Server** (`arxiv_search`, `wikipedia_search`, `pubmed_search`) and maintains a cross-session RAG (Retrieval-Augmented Generation) memory store using **FAISS** and free **Gemini Embeddings**.

---

## Key Features

- **Stateful LangGraph Workflow**: Built on `StateGraph` using a `CurSession` TypedDict reducer (`prompts` & `replies`).
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
- **Interactive Streamlit Interface**:
  - Real-time chat interface in `app.py`.
  - Sidebar configuration for OpenAI API keys, Google/Gemini API keys, model selection (`gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`), and toggles for individual FastMCP research tools.

---

## Project Architecture

```
research_agent/
├── app.py              # Main Streamlit Application invoking tools via FastMCP Server
├── mcp_server.py       # FastMCP Server exposing research tool endpoints
├── mcp_tools.py        # Academic API Tool implementations (arXiv, Wikipedia, PubMed)
├── retriever.py        # Past Chat JSON Persistence & Gemini FAISS Vector RAG Engine
├── past_chats/         # Directory storing session JSON files
└── README.md           # Documentation
```

---

## Getting Started

### Prerequisites

Ensure you have Python 3.10+ installed.

### Installation

Install the required Python dependencies:

```bash
pip install streamlit langgraph langchain langchain-openai langchain-google-genai google-genai mcp langchain-mcp-adapters ddgs faiss-cpu python-dotenv
```

### Running the Web Application

Launch the Streamlit interface:

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

Configure your API keys via `.env` file or directly inside the Streamlit sidebar:

- **`OPENAI_API_KEY`**: Enables OpenAI model responses (`gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`).
- **`GOOGLE_API_KEY` / `GEMINI_API_KEY`**: Enables free **Gemini Embeddings** (`models/text-embedding-004`) for past chat vector retrieval.

If no API keys are entered, the application automatically functions as a raw research collector, retrieving and displaying paper summaries, Wikipedia facts, and web results directly.