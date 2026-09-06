"""
app.py
------
Streamlit front-end with a three-node LangGraph multi-agent pipeline:

  [orchestrator]  Heuristic router — scores and orders the enabled MCP tools
                  based on the query type. No LLM call; instant decision.

  [retriever]     Fires all selected MCP tools concurrently via
                  ThreadPoolExecutor. Each tool has its own retry + timeout
                  logic encapsulated in mcp_server.py.

  [synthesiser]   Single Gemini call that reasons over the collected context
                  and produces a structured Markdown research report.

Model provider: Google Gemini only.
"""

import operator
import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from typing_extensions import Annotated, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from retriever import save_chat_session, build_knowledge_base_if_needed, build_memory_index
from mcp_server import mcp

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

.sidebar-section {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #484f58 !important;
    margin: 1.1rem 0 0.35rem 0;
}

[data-testid="stToggle"] label  { font-size: 0.82rem !important; }
[data-testid="stSelectbox"] label,
[data-testid="stTextInput"] label {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #484f58 !important;
}

/* Status pill */
.pill {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    margin-top: 4px;
}
.pill-green  { background:#0d2a1f; color:#3fb950; border:1px solid #238636; }
.pill-amber  { background:#271d0a; color:#d29922; border:1px solid #9e6a03; }
.pill-red    { background:#2d0d0d; color:#f85149; border:1px solid #da3633; }
.pill-blue   { background:#0c1a2e; color:#58a6ff; border:1px solid #1f6feb; }

/* Agent pipeline banner shown during response generation */
.pipeline-step {
    display: inline-block;
    padding: 2px 8px;
    margin: 2px 3px;
    border-radius: 4px;
    font-size: 0.72rem;
    background: #161b22;
    border: 1px solid #30363d;
    color: #8b949e;
}
.pipeline-step.active { border-color: #388bfd; color: #58a6ff; }
.pipeline-step.done   { border-color: #238636; color: #3fb950; }

/* Main header */
.app-header { padding: 1.2rem 0 1rem 0; border-bottom: 1px solid #21262d; margin-bottom: 1.2rem; }
.app-header h1 { font-size: 1.45rem; font-weight: 600; color: #e6edf3; margin: 0; }
.app-header p  { font-size: 0.8rem; color: #8b949e; margin: 0.15rem 0 0 0; }

/* Chat input */
[data-testid="stChatInputTextArea"] { font-family: 'Inter', sans-serif !important; }

/* Response headings */
.stMarkdown h3 {
    font-size: 0.85rem;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-top: 1px solid #21262d;
    padding-top: 0.8rem;
    margin-top: 1.2rem;
}

/* Hide Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ─── State ────────────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    # Conversation history — accumulated across turns
    prompts:   Annotated[List[str], operator.add]
    replies:   Annotated[List[str], operator.add]
    # Set by orchestrator, consumed by retriever
    tool_plan: List[str]
    # Filled by retriever, consumed by synthesiser
    contexts:  Annotated[List[str], operator.add]


# ─── API key persistence ─────────────────────────────────────────────────────

def _persist_key_to_env(key: str, var: str = "GOOGLE_API_KEY") -> None:
    """
    Write the API key to .env so it survives restarts.
    Updates an existing line if the variable is already present, appends otherwise.
    Also sets the variable in the current process so load_dotenv() isn't needed again.
    """
    env_path = ".env"
    lines: List[str] = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{var}="):
            lines[i] = f"{var}={key}\n"
            updated   = True
            break
    if not updated:
        lines.append(f"{var}={key}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    os.environ[var] = key


@st.cache_resource(show_spinner=False)
def _init_knowledge_base() -> str:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    try:
        return build_knowledge_base_if_needed(
            books_dir="books", scans_dir="scans", google_api_key=key
        )
    except Exception as exc:
        logger.error("KB init: %s", exc)
        return f"error: {exc}"

kb_status = _init_knowledge_base()


# ─── MCP tool invocation (runs inside threads) ────────────────────────────────

def _invoke_tool(tool_name: str, args: Dict[str, Any]) -> tuple[str, str]:
    """
    Synchronous wrapper around async mcp.call_tool().
    Each thread creates its own event loop via asyncio.run().
    Retry + timeout are already handled by mcp_server._tool_wrapper.
    """
    try:
        blocks, _ = asyncio.run(mcp.call_tool(tool_name, args))
        text = blocks[0].text if blocks else ""
        return tool_name, text
    except Exception as exc:
        logger.warning("Tool %s invocation error: %s", tool_name, exc)
        return tool_name, ""


# ─── Tool metadata ────────────────────────────────────────────────────────────

# Human-readable section titles for each MCP tool
TOOL_LABELS: Dict[str, str] = {
    "knowledge_base_search": "Knowledge Base (Local Books)",
    "memory_search":         "Conversation Memory",
    "arxiv_search":          "arXiv Academic Papers",
    "wikipedia_search":      "Wikipedia",
    "pubmed_search":         "PubMed Literature",
    "web_search":            "Web Search",
}

# Sidebar toggle keys → tool name
TOGGLE_MAP: Dict[str, str] = {
    "enable_kb":     "knowledge_base_search",
    "enable_memory": "memory_search",
    "enable_arxiv":  "arxiv_search",
    "enable_wiki":   "wikipedia_search",
    "enable_pubmed": "pubmed_search",
    "enable_web":    "web_search",
}

# Keywords used by the heuristic orchestrator
_ACADEMIC  = {"research", "paper", "study", "journal", "arxiv", "model", "algorithm",
              "method", "theorem", "proof", "experiment", "dataset", "neural", "ml", "ai"}
_BIOMEDICAL = {"drug", "disease", "clinical", "patient", "gene", "protein", "therapy",
               "medicine", "health", "symptom", "treatment", "diagnosis", "pubmed"}
_FACTUAL   = {"what is", "what are", "define", "definition", "explain", "meaning",
              "who is", "when was", "where is"}
_NEWS      = {"latest", "recent", "current", "new", "update", "2024", "2025", "2026",
              "news", "today", "announcement"}


# ─── Node 1: Orchestrator ─────────────────────────────────────────────────────

def orchestrator_node(state: ResearchState) -> Dict[str, Any]:
    """
    Heuristic router. Scores each enabled tool against the query and returns
    an ordered tool_plan. No LLM call — instant decision, zero API cost.

    Tools that require a Google/Gemini API key (knowledge_base_search,
    memory_search) are automatically excluded from the plan when no key
    is configured, avoiding wasted parallel threads.
    """
    query   = state["prompts"][-1]
    q_lower = query.lower()
    words   = set(q_lower.split())

    google_key = (
        st.session_state.get("google_api_key")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or ""
    )

    # Collect enabled tools from session state
    enabled: List[str] = [
        tool for key, tool in TOGGLE_MAP.items()
        if st.session_state.get(key, key not in ("enable_pubmed",))
    ]

    # kb_search and memory_search require Gemini embeddings — skip without a key
    if not google_key:
        enabled = [t for t in enabled if t not in ("knowledge_base_search", "memory_search")]

    # Score each enabled tool
    scores: Dict[str, int] = {t: 10 for t in enabled}

    # Academic signals
    if words & _ACADEMIC:
        for t in ("knowledge_base_search", "arxiv_search"):
            if t in scores: scores[t] += 8
    # Biomedical signals
    if words & _BIOMEDICAL:
        for t in ("pubmed_search", "arxiv_search"):
            if t in scores: scores[t] += 8
    # Factual / definition signals
    if any(k in q_lower for k in _FACTUAL):
        for t in ("wikipedia_search", "knowledge_base_search"):
            if t in scores: scores[t] += 6
    # News / recency signals
    if words & _NEWS:
        if "web_search" in scores: scores["web_search"] += 8
    # Memory is always valuable; boost it slightly
    if "memory_search" in scores: scores["memory_search"] += 4
    # Local KB is free to query; slight boost
    if "knowledge_base_search" in scores: scores["knowledge_base_search"] += 3

    tool_plan = sorted(enabled, key=lambda t: scores.get(t, 0), reverse=True)
    logger.info("Orchestrator tool_plan: %s", tool_plan)
    return {"tool_plan": tool_plan}


# ─── Node 2: Parallel Retriever ───────────────────────────────────────────────

def retriever_node(state: ResearchState) -> Dict[str, Any]:
    """
    Fires all tools in tool_plan concurrently using ThreadPoolExecutor.
    Each thread calls asyncio.run(mcp.call_tool(...)) independently.
    Results arrive as they complete (as_completed); order is non-deterministic.
    """
    query      = state["prompts"][-1]
    tool_plan  = state.get("tool_plan", [])
    google_key = (
        st.session_state.get("google_api_key")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or ""
    )

    # Build (tool_name, args) pairs
    tool_calls: List[tuple[str, Dict[str, Any]]] = []
    for tool in tool_plan:
        if tool in ("knowledge_base_search", "memory_search"):
            tool_calls.append((tool, {"query": query, "api_key": google_key,
                                      "top_k": 4 if tool == "knowledge_base_search" else 3}))
        elif tool == "arxiv_search":
            tool_calls.append((tool, {"query": query, "max_results": 3}))
        elif tool == "wikipedia_search":
            tool_calls.append((tool, {"query": query, "max_results": 2}))
        elif tool == "pubmed_search":
            tool_calls.append((tool, {"query": query, "max_results": 3}))
        elif tool == "web_search":
            tool_calls.append((tool, {"query": query, "max_results": 3}))

    if not tool_calls:
        return {"contexts": []}

    contexts: List[str] = []
    # Run all tools in parallel; collect as they finish
    with ThreadPoolExecutor(max_workers=min(len(tool_calls), 8)) as executor:
        futures = {
            executor.submit(_invoke_tool, name, args): name
            for name, args in tool_calls
        }
        for future in as_completed(futures, timeout=60):
            tool_name = futures[future]
            try:
                _, result = future.result()
                if result and result.strip() and len(result) > 30:
                    label = TOOL_LABELS.get(tool_name, tool_name)
                    contexts.append(f"### {label}\n{result}")
            except Exception as exc:
                logger.warning("Retriever: %s future error — %s", tool_name, exc)

    return {"contexts": contexts}


# ─── Node 3: Synthesiser ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert AI Research Assistant with access to multiple retrieved knowledge sources.

## Reasoning Protocol

Before writing, work through these steps silently:

1. Understand — identify exactly what is being asked.
2. Evaluate — critically assess each context block. Prefer peer-reviewed and authoritative sources. \
Discard snippets that are irrelevant or contradictory.
3. Synthesise — do NOT copy-paste raw context. Extract key concepts, reconcile perspectives, \
apply your own domain knowledge to fill gaps.
4. Respond — produce a well-structured Markdown report in the format below.

## Output Format

### Overview
2–4 sentences directly answering the core question.

### Key Findings
Bullet-point synthesis. Each point must be a genuine insight, not a verbatim quote.

### Sources
Cite sources with URLs. Format: [Name](URL) — one-line description.

### Conclusions
Brief synthesis paragraph. Note context limitations and suggest follow-up directions.

## Rules
- Be technically rigorous and precise.
- If context is insufficient, say so explicitly — do not speculate.
- Professional, academic tone.
- Use `inline code` for model names, formulas, technical identifiers.
"""


@st.cache_resource(show_spinner=False)
def _get_llm(model_name: str, google_key: str):
    """Cached Gemini LLM instance. Re-created only when model or key changes."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=google_key,
        temperature=0.35,
        max_retries=3,
    )


def synthesiser_node(state: ResearchState) -> Dict[str, Any]:
    """
    Assembles the context block collected by the retriever and calls Gemini
    to produce a structured research report.
    """
    query      = state["prompts"][-1]
    contexts   = state.get("contexts", [])
    google_key = (
        st.session_state.get("google_api_key")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )
    model_name = st.session_state.get("model_name", "gemini-2.0-flash")

    # Always build the context block — it is shown regardless of API key
    if contexts:
        ctx_block = (
            f"## Retrieved Context ({len(contexts)} source(s))\n\n"
            + "\n\n---\n\n".join(contexts)
        )
    else:
        ctx_block = "No external research context was retrieved."

    # No API key → show a clear notice and the raw context; do not block
    if not google_key:
        return {"replies": [
            "> **No Gemini API key configured.** "
            "LLM synthesis is disabled — showing raw research context below. "
            "Add your key in the sidebar to enable AI-powered synthesis.\n\n"
            "---\n\n" + ctx_block
        ]}

    try:
        llm = _get_llm(model_name, google_key)

        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        messages.append(HumanMessage(content=ctx_block))

        # Inject prior conversation turns
        for p, r in zip(state["prompts"][:-1], state["replies"]):
            messages.append(HumanMessage(content=p))
            messages.append(AIMessage(content=r))

        messages.append(HumanMessage(content=query))

        response = llm.invoke(messages)
        return {"replies": [response.content]}

    except Exception as exc:
        err = str(exc)
        err_l = err.lower()
        if "api key" in err_l or "invalid" in err_l or "unauthorized" in err_l:
            msg = "**Authentication error:** Your Gemini API key is invalid or expired."
        elif "quota" in err_l or "429" in err_l or "rate" in err_l:
            msg = "**Rate limit:** Gemini API quota exceeded. Please wait a moment and try again."
        elif "network" in err_l or "connection" in err_l or "connect" in err_l:
            msg = "**Network error:** Could not reach the Gemini API. Check your connection."
        else:
            msg = f"**Synthesis error:** `{err}`"
        return {"replies": [f"{msg}\n\n---\n\n{ctx_block}"]}


# ─── LangGraph pipeline ───────────────────────────────────────────────────────

def _build_graph() -> Any:
    g = StateGraph(ResearchState)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("retriever",    retriever_node)
    g.add_node("synthesiser",  synthesiser_node)
    g.add_edge(START,          "orchestrator")
    g.add_edge("orchestrator", "retriever")
    g.add_edge("retriever",    "synthesiser")
    g.add_edge("synthesiser",  END)
    return g.compile()

research_graph = _build_graph()


# ─── Session state defaults ───────────────────────────────────────────────────
for _k in ("prompts", "replies"):
    if _k not in st.session_state:
        st.session_state[_k] = []


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Research Agent")
    st.markdown("---")

    st.markdown('<p class="sidebar-section">API Key</p>', unsafe_allow_html=True)
    google_key_input = st.text_input(
        "Google / Gemini API Key",
        value=os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", "")),
        type="password",
        placeholder="AIzaSy...",
        help="Used for both the Gemini LLM and knowledge base embeddings.",
    )
    if google_key_input:
        # Persist to .env so restarts don't ask again
        if google_key_input != os.getenv("GOOGLE_API_KEY", ""):
            _persist_key_to_env(google_key_input, "GOOGLE_API_KEY")
        st.session_state["google_api_key"] = google_key_input

    st.markdown('<p class="sidebar-section">Model</p>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Gemini model",
        options=[
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ],
        index=0,
        help="gemini-2.0-flash: best speed/quality balance. gemini-1.5-pro: most capable.",
    )
    st.session_state["model_name"] = model_choice

    st.markdown('<p class="sidebar-section">Research Sources</p>', unsafe_allow_html=True)
    st.session_state["enable_kb"]     = st.toggle("Knowledge Base",       value=True)
    st.session_state["enable_memory"] = st.toggle("Conversation Memory",  value=True)
    st.session_state["enable_arxiv"]  = st.toggle("arXiv Papers",         value=True)
    st.session_state["enable_wiki"]   = st.toggle("Wikipedia",            value=True)
    st.session_state["enable_pubmed"] = st.toggle("PubMed Literature",    value=False)
    st.session_state["enable_web"]    = st.toggle("Web Search",           value=True)

    st.markdown("---")
    st.markdown('<p class="sidebar-section">Knowledge Base</p>', unsafe_allow_html=True)
    if kb_status == "built":
        st.markdown('<span class="pill pill-green">Index built from books/</span>', unsafe_allow_html=True)
    elif kb_status == "up_to_date":
        st.markdown('<span class="pill pill-blue">Index up to date</span>', unsafe_allow_html=True)
    elif kb_status == "no_books":
        st.markdown('<span class="pill pill-amber">No books — add .pdf/.txt/.md to books/</span>', unsafe_allow_html=True)
    elif kb_status == "no_api_key":
        st.markdown('<span class="pill pill-amber">Set a Google API key to enable indexing</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="pill pill-red">Error: {kb_status}</span>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Clear conversation", type="secondary", use_container_width=True):
        st.session_state["prompts"] = []
        st.session_state["replies"] = []
        st.session_state.pop("session_id", None)
        st.rerun()


# ─── Main area ────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-header">'
    '<h1>Research Agent</h1>'
    '<p>Multi-agent pipeline · Parallel retrieval · Gemini synthesis</p>'
    '</div>',
    unsafe_allow_html=True,
)

if not st.session_state["prompts"]:
    with st.chat_message("assistant"):
        st.markdown(
            "Ready. I route your query through a heuristic orchestrator, then fire all relevant "
            "sources in parallel — knowledge base, arXiv, Wikipedia, PubMed, web — before "
            "synthesising everything into a structured report. What would you like to explore?"
        )

for u, r in zip(st.session_state["prompts"], st.session_state["replies"]):
    with st.chat_message("user"):
        st.write(u)
    with st.chat_message("assistant"):
        st.markdown(r)

prompt = st.chat_input("Ask a research question...")

if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    inputs: ResearchState = {
        "prompts":   [prompt],
        "replies":   [],
        "tool_plan": [],
        "contexts":  [],
    }

    reply        = ""
    plan_label   = ""

    with st.chat_message("assistant"):
        status_box = st.empty()

        for chunk in research_graph.stream(inputs, stream_mode="updates"):
            node = next(iter(chunk))
            data = chunk[node]

            if node == "orchestrator":
                plan = data.get("tool_plan", [])
                plan_label = ", ".join(TOOL_LABELS.get(t, t) for t in plan)
                status_box.markdown(
                    f"**Routing** — {len(plan)} source(s) selected: "
                    f"<span style='color:#8b949e;font-size:0.8rem'>{plan_label}</span>",
                    unsafe_allow_html=True,
                )

            elif node == "retriever":
                n = len(data.get("contexts", []))
                status_box.markdown(
                    f"**Gathering context** — {n} source(s) returned results. Synthesising...",
                )

            elif node == "synthesiser":
                replies = data.get("replies", [])
                if replies:
                    reply = replies[-1]

        status_box.empty()
        if reply:
            st.markdown(reply)
        else:
            reply = "No response was generated. Check your API key and connection."
            st.warning(reply)

    st.session_state["prompts"].append(prompt)
    st.session_state["replies"].append(reply)

    # Persist chat to past_chats/ JSON
    session_id = st.session_state.get("session_id")
    saved_id   = save_chat_session(
        prompts=st.session_state["prompts"],
        replies=st.session_state["replies"],
        folder_path="past_chats",
        session_id=session_id,
    )
    st.session_state["session_id"] = saved_id

    # Refresh the persistent memory index in the background so the next
    # query can search this turn without rebuilding from scratch.
    google_key = (
        st.session_state.get("google_api_key")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )
    if google_key:
        try:
            build_memory_index(
                google_api_key=google_key,
                folder_path="past_chats",
                scans_dir="scans",
            )
        except Exception as _mem_exc:
            logger.warning("Memory index refresh failed: %s", _mem_exc)