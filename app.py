"""
app.py — JANE (Just A Nuanced Engine)
--------------------------------------
Production-grade multi-agent research engine built with LangGraph and Streamlit.
Features:
  - Single-pass real-time token streaming (zero duplicate LLM calls)
  - Parallel resilient retrieval via FastMCP tools
  - Comprehensive telemetry & per-source latency tracking
  - Incremental FAISS conversational memory
  - Enterprise dark theme UI
"""

import time
import operator
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from typing_extensions import Annotated, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from config import settings
from logger import app_logger
from retriever import (
    save_chat_session,
    delete_chat_session,
    load_all_chat_sessions,
    export_session_to_markdown,
    build_knowledge_base_if_needed,
    build_memory_index,
)
from mcp_server import mcp

load_dotenv()
settings.ensure_directories()

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JANE — Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Design System ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

.sidebar-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #484f58 !important;
    margin: 1.1rem 0 0.3rem 0;
}

/* Session load button */
.sess-btn [data-testid="stButton"] button {
    width: 100%;
    text-align: left !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 5px !important;
    padding: 5px 8px !important;
    font-size: 0.77rem !important;
    color: #8b949e !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    transition: all 0.12s ease;
}
.sess-btn [data-testid="stButton"] button:hover {
    background: #161b22 !important;
    border-color: #30363d !important;
    color: #c9d1d9 !important;
}
.sess-active [data-testid="stButton"] button {
    background: #161b22 !important;
    border-color: #388bfd !important;
    color: #e6edf3 !important;
}

/* Delete (×) button */
.del-btn [data-testid="stButton"] button {
    background: transparent !important;
    border: none !important;
    color: #484f58 !important;
    font-size: 0.78rem !important;
    padding: 4px 6px !important;
    border-radius: 4px !important;
    min-height: unset !important;
    height: 28px !important;
    transition: color 0.12s ease;
}
.del-btn [data-testid="stButton"] button:hover {
    color: #f85149 !important;
    background: #2d0d0d !important;
}

/* New chat / sidebar generic buttons */
.new-chat-btn [data-testid="stButton"] button {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    border-radius: 6px !important;
    width: 100%;
}
.new-chat-btn [data-testid="stButton"] button:hover {
    border-color: #388bfd !important;
}

/* Search box */
[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 5px !important;
    font-size: 0.78rem !important;
    color: #c9d1d9 !important;
    padding: 5px 10px !important;
}

/* Toggles + selectbox */
[data-testid="stToggle"] label  { font-size: 0.8rem !important; }
[data-testid="stSelectbox"] label {
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #484f58 !important;
}

/* Status pills */
.pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 500;
    margin: 2px 3px 2px 0;
}
.pill-green  { background:#0d2a1f; color:#3fb950; border:1px solid #238636; }
.pill-amber  { background:#271d0a; color:#d29922; border:1px solid #9e6a03; }
.pill-blue   { background:#0c1a2e; color:#58a6ff; border:1px solid #1f6feb; }
.pill-red    { background:#2d0d0d; color:#f85149; border:1px solid #da3633; }
.pill-gray   { background:#161b22; color:#8b949e; border:1px solid #30363d; }

/* Main header */
.app-header {
    padding: 0.8rem 0 0.6rem 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
}
.app-header h1 {
    font-size: 1.4rem;
    font-weight: 600;
    color: #e6edf3;
    margin: 0;
    letter-spacing: -0.01em;
}
.app-header span {
    font-size: 0.75rem;
    color: #484f58;
}

/* Telemetry Badge */
.telemetry-tag {
    font-size: 0.71rem;
    color: #6e7681;
    margin-top: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-family: monospace;
}

/* No-key banner */
.no-key-banner {
    background: #271d0a;
    border: 1px solid #9e6a03;
    border-radius: 6px;
    padding: 7px 14px;
    font-size: 0.78rem;
    color: #d29922;
    margin-bottom: 0.8rem;
}

/* Response headings */
.stMarkdown h3 {
    font-size: 0.82rem;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-top: 1px solid #21262d;
    padding-top: 0.75rem;
    margin-top: 1.1rem;
}

#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)


# ─── State Schema ─────────────────────────────────────────────────────────────
class ResearchState(TypedDict):
    prompts: Annotated[List[str], operator.add]
    replies: Annotated[List[str], operator.add]
    tool_plan: List[str]
    contexts: Annotated[List[str], operator.add]
    refined_query: str
    telemetry: Dict[str, Any]


# ─── Startup Index Builds ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _init_indices() -> Dict[str, str]:
    key = settings.api_key
    out: Dict[str, str] = {}
    try:
        out["kb"] = build_knowledge_base_if_needed(
            books_dir=settings.books_dir,
            scans_dir=settings.scans_dir,
            google_api_key=key,
        )
    except Exception as exc:
        out["kb"] = f"error:{exc}"
    try:
        out["memory"] = build_memory_index(
            google_api_key=key,
            folder_path=settings.past_chats_dir,
            scans_dir=settings.scans_dir,
        )
    except Exception as exc:
        out["memory"] = f"error:{exc}"
    return out


idx_status = _init_indices()


# ─── Session Helpers ──────────────────────────────────────────────────────────
def _all_sessions() -> List[Dict[str, Any]]:
    sessions = load_all_chat_sessions(settings.past_chats_dir)
    sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
    return sessions


def _session_title(s: Dict[str, Any]) -> str:
    chats = s.get("chats", [])
    if chats:
        first = chats[0].get("prompt", "")
        return (first[:50] + "…") if len(first) > 50 else first
    return s.get("session_id", "Untitled")


def _session_date(s: Dict[str, Any]) -> str:
    ts = s.get("updated_at", "")
    if ts:
        try:
            from datetime import datetime
            return datetime.fromisoformat(ts).strftime("%d %b, %H:%M")
        except Exception:
            pass
    return ""


def _load_session(s: Dict[str, Any]) -> None:
    chats = s.get("chats", [])
    st.session_state["prompts"] = [t.get("prompt", "") for t in chats]
    st.session_state["replies"] = [t.get("reply", "") for t in chats]
    st.session_state["contexts_list"] = [t.get("contexts", []) for t in chats]
    st.session_state["session_id"] = s.get("session_id")
    st.session_state["turn_metrics"] = []


def _new_chat() -> None:
    st.session_state["prompts"] = []
    st.session_state["replies"] = []
    st.session_state["contexts_list"] = []
    st.session_state["turn_metrics"] = []
    st.session_state.pop("session_id", None)


# ─── Source Toggle Defaults ───────────────────────────────────────────────────
SOURCE_DEFAULTS: Dict[str, bool] = {
    "enable_kb": True,
    "enable_memory": True,
    "enable_arxiv": True,
    "enable_wiki": True,
    "enable_pubmed": False,
    "enable_web": True,
}
for _k, _v in SOURCE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─── MCP Tool Invocation ──────────────────────────────────────────────────────
def _invoke_tool(tool_name: str, args: Dict[str, Any]) -> tuple[str, str, float]:
    t0 = time.perf_counter()
    try:
        blocks, _ = asyncio.run(mcp.call_tool(tool_name, args))
        res = blocks[0].text if blocks else ""
        return tool_name, res, time.perf_counter() - t0
    except Exception as exc:
        app_logger.warning(f"Tool {tool_name} error: {exc}")
        return tool_name, "", time.perf_counter() - t0


# ─── Tool Metadata & Heuristics ───────────────────────────────────────────────
TOOL_LABELS: Dict[str, str] = {
    "knowledge_base_search": "Knowledge Base",
    "memory_search": "Memory",
    "arxiv_search": "arXiv",
    "wikipedia_search": "Wikipedia",
    "pubmed_search": "PubMed",
    "web_search": "Web",
}
TOGGLE_MAP: Dict[str, str] = {
    "enable_kb": "knowledge_base_search",
    "enable_memory": "memory_search",
    "enable_arxiv": "arxiv_search",
    "enable_wiki": "wikipedia_search",
    "enable_pubmed": "pubmed_search",
    "enable_web": "web_search",
}

_ACADEMIC = {
    "research", "paper", "study", "journal", "arxiv", "model", "algorithm",
    "method", "theorem", "proof", "experiment", "dataset", "neural", "ml", "ai"
}
_BIOMEDICAL = {
    "drug", "disease", "clinical", "patient", "gene", "protein", "therapy",
    "medicine", "health", "symptom", "treatment", "diagnosis"
}
_FACTUAL = {"what is", "what are", "define", "definition", "explain", "meaning", "who is", "when was"}
_NEWS = {"latest", "recent", "current", "new", "update", "news", "today"}

_REWRITE_PROMPT = (
    "You are a search-query specialist. Given a user's research question, "
    "produce one concise, keyword-rich search query (max 12 words, no filler "
    "words, no punctuation) that will maximise retrieval from academic databases "
    "and web search engines. Output ONLY the query string, nothing else."
)


def _rewrite_query(raw_query: str, key: str, model: str) -> str:
    """Use a lightweight LLM call to rewrite the prompt into a terse retrieval query."""
    if not key:
        return raw_query
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=key,
            temperature=0.0,
            max_retries=1,
            timeout=8.0,
        )
        resp = llm.invoke([
            SystemMessage(content=_REWRITE_PROMPT),
            HumanMessage(content=raw_query),
        ])
        refined = resp.content.strip().strip('"').strip("'")
        return refined if refined else raw_query
    except Exception as exc:
        app_logger.warning(f"Query rewrite failed: {exc}")
        return raw_query


# ─── Node 1: Orchestrator ─────────────────────────────────────────────────────
def orchestrator_node(state: ResearchState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    query = state["prompts"][-1]
    ql = query.lower()
    words = set(ql.split())
    key = settings.api_key
    model = st.session_state.get("model_name", settings.default_model)

    refined_query = _rewrite_query(query, key, model)

    enabled = [
        tool for toggle, tool in TOGGLE_MAP.items()
        if st.session_state.get(toggle, SOURCE_DEFAULTS.get(toggle, True))
    ]
    if not key:
        enabled = [t for t in enabled if t not in ("knowledge_base_search", "memory_search")]

    scores: Dict[str, int] = {t: 10 for t in enabled}
    if words & _ACADEMIC:
        for t in ("knowledge_base_search", "arxiv_search"):
            if t in scores:
                scores[t] += 8
    if words & _BIOMEDICAL:
        for t in ("pubmed_search", "arxiv_search"):
            if t in scores:
                scores[t] += 8
    if any(k in ql for k in _FACTUAL):
        for t in ("wikipedia_search", "knowledge_base_search"):
            if t in scores:
                scores[t] += 6
    if words & _NEWS:
        if "web_search" in scores:
            scores["web_search"] += 8
    if "memory_search" in scores:
        scores["memory_search"] += 4
    if "knowledge_base_search" in scores:
        scores["knowledge_base_search"] += 3

    tool_plan = sorted(enabled, key=lambda t: scores.get(t, 0), reverse=True)
    latency = time.perf_counter() - t0

    return {
        "tool_plan": tool_plan,
        "refined_query": refined_query,
        "telemetry": {"orchestrator_latency": latency},
    }


# ─── Node 2: Parallel Retriever ───────────────────────────────────────────────
def retriever_node(state: ResearchState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    query = state.get("refined_query") or state["prompts"][-1]
    tool_plan = state.get("tool_plan", [])
    key = settings.api_key

    calls: List[tuple[str, Dict]] = []
    for tool in tool_plan:
        if tool in ("knowledge_base_search", "memory_search"):
            calls.append((tool, {"query": query, "api_key": key, "top_k": 4 if tool == "knowledge_base_search" else 3}))
        elif tool == "arxiv_search":
            calls.append((tool, {"query": query, "max_results": 3}))
        elif tool == "wikipedia_search":
            calls.append((tool, {"query": query, "max_results": 2}))
        elif tool == "pubmed_search":
            calls.append((tool, {"query": query, "max_results": 3}))
        elif tool == "web_search":
            calls.append((tool, {"query": query, "max_results": 3}))

    if not calls:
        return {"contexts": [], "telemetry": {"retriever_latency": 0.0, "sources": []}}

    contexts: List[str] = []
    source_stats = []

    with ThreadPoolExecutor(max_workers=min(len(calls), 8)) as ex:
        futures = {ex.submit(_invoke_tool, n, a): n for n, a in calls}
        for future in as_completed(futures, timeout=settings.http_timeout_s + 5.0):
            tool_name = futures[future]
            try:
                _, result, duration = future.result()
                has_content = bool(result and result.strip() and len(result) > 30)
                if has_content:
                    label = TOOL_LABELS.get(tool_name, tool_name)
                    contexts.append(f"### {label}\n{result}")
                source_stats.append({
                    "tool": tool_name,
                    "label": TOOL_LABELS.get(tool_name, tool_name),
                    "duration": duration,
                    "success": has_content,
                })
            except Exception as exc:
                app_logger.warning(f"Future error for {tool_name}: {exc}")

    retriever_latency = time.perf_counter() - t0
    return {
        "contexts": contexts,
        "telemetry": {
            "retriever_latency": retriever_latency,
            "sources": source_stats,
        },
    }


# ─── Graph Compilation ────────────────────────────────────────────────────────
def _build_graph() -> Any:
    g = StateGraph(ResearchState)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("retriever", retriever_node)
    g.add_edge(START, "orchestrator")
    g.add_edge("orchestrator", "retriever")
    g.add_edge("retriever", END)
    return g.compile()


research_graph = _build_graph()

# ─── Synthesiser Prompts & Helpers ────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are JANE — Just A Nuanced Engine — an expert AI Research Assistant.

## Reasoning Protocol
1. Understand — identify exactly what is being asked.
2. Evaluate — critically assess each context block. Prefer peer-reviewed sources.
3. Synthesise — do NOT copy-paste raw context. Extract key concepts, reconcile perspectives.
4. Respond — produce a structured Markdown report.

## Output Format

### Overview
2–4 sentences directly answering the core question.

### Key Findings
Bullet-point synthesis. Each point must be a genuine insight, not a verbatim quote.

### Sources
[Name](URL) — one-line description.

### Conclusions
Brief synthesis. Note limitations and suggest follow-up directions.

## Rules
- Technically rigorous. If context is insufficient, say so explicitly.
- Professional, academic tone.
- Use `inline code` for model names, formulas, technical identifiers.
"""


@st.cache_resource(show_spinner=False)
def _get_llm(model_name: str, google_key: str):
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=google_key,
        temperature=0.35,
        max_retries=2,
    )


def _build_messages(prompts: List[str], replies: List[str], ctx_block: str, current_prompt: str) -> list:
    """Assemble the conversation and context message history."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.append(HumanMessage(content=ctx_block))
    for p, r in zip(prompts, replies):
        messages.append(HumanMessage(content=p))
        messages.append(AIMessage(content=r))
    messages.append(HumanMessage(content=current_prompt))
    return messages


# ─── Session State Defaults ───────────────────────────────────────────────────
for _k in ("prompts", "replies", "contexts_list", "turn_metrics"):
    if _k not in st.session_state:
        st.session_state[_k] = []


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## JANE")
    st.markdown("---")

    st.markdown('<p class="sidebar-label">Conversations</p>', unsafe_allow_html=True)

    search_q = st.text_input(
        "search", placeholder="Search…",
        label_visibility="collapsed", key="sess_search"
    )

    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("+ New Chat", key="new_chat_btn", use_container_width=True):
        _new_chat()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.get("prompts"):
        curr_id = st.session_state.get("session_id", "active")
        report_md = export_session_to_markdown(
            prompts=st.session_state["prompts"],
            replies=st.session_state["replies"],
            contexts_list=st.session_state.get("contexts_list"),
            session_id=curr_id,
            model_name=st.session_state.get("model_name"),
        )
        st.download_button(
            label="Export Dossier (.md)",
            data=report_md,
            file_name=f"research_dossier_{curr_id}.md",
            mime="text/markdown",
            key="export_report_btn",
            use_container_width=True,
        )

    sessions = _all_sessions()
    current_sid = st.session_state.get("session_id")
    sq = (search_q or "").strip().lower()
    filtered = [s for s in sessions if not sq or sq in _session_title(s).lower()]

    for sess in filtered:
        sid = sess.get("session_id", "")
        title = _session_title(sess)
        date = _session_date(sess)
        is_active = sid == current_sid

        col_title, col_del = st.columns([0.86, 0.14])

        with col_title:
            css_class = "sess-active" if is_active else "sess-btn"
            st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
            if st.button(title, key=f"sess_{sid}", help=date, use_container_width=True):
                _load_session(sess)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with col_del:
            st.markdown('<div class="del-btn">', unsafe_allow_html=True)
            if st.button("×", key=f"del_{sid}", help="Delete"):
                delete_chat_session(sid)
                if is_active:
                    _new_chat()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Model Selection ───────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-label">Model</p>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "model", label_visibility="collapsed",
        options=settings.available_models,
        index=0,
    )
    st.session_state["model_name"] = model_choice

    # ── System Health & Index Badges ──────────────────────────────────────────
    st.markdown('<p class="sidebar-label">System Health</p>', unsafe_allow_html=True)
    has_key = settings.has_api_key

    if has_key:
        st.markdown('<span class="pill pill-green">Gemini API Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill pill-amber">API Key Required</span>', unsafe_allow_html=True)

    kb_s = idx_status.get("kb", "")
    if kb_s in ("built", "up_to_date"):
        st.markdown('<span class="pill pill-blue">KB Ready</span>', unsafe_allow_html=True)
    elif kb_s == "no_books":
        st.markdown('<span class="pill pill-gray">No books</span>', unsafe_allow_html=True)

    mem_s = idx_status.get("memory", "")
    if mem_s in ("built", "up_to_date"):
        st.markdown('<span class="pill pill-green">Memory Synced</span>', unsafe_allow_html=True)

    with st.expander("Diagnostics", expanded=False):
        if st.button("Run System Diagnostics", key="diag_btn", use_container_width=True):
            with st.spinner("Probing tools and APIs..."):
                from health import check_directories, check_gemini_api, check_tool_connectivity
                d_ok, d_msg = check_directories()
                g_ok, g_msg = check_gemini_api()
                t_res = check_tool_connectivity()

                st.write(f"**Storage:** {'PASS' if d_ok else 'FAIL'}")
                st.write(f"**Gemini API:** {'PASS' if g_ok else 'WARN'} ({g_msg})")
                for name, (ok, msg) in t_res.items():
                    st.write(f"**{name}:** {'PASS' if ok else 'WARN'} ({msg})")


# ─── Main Area ────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-header">'
    f'<h1>{settings.app_name}</h1>'
    f'<span>v{settings.app_version} &nbsp;·&nbsp; Parallel Academic & Web Intelligence</span>'
    '</div>',
    unsafe_allow_html=True,
)

if not has_key:
    st.markdown(
        '<div class="no-key-banner">'
        'No Gemini API key detected. Set <code>GEMINI_API_KEY=...</code> in <code>.env</code>. '
        'arXiv, Wikipedia, PubMed, and Web search remain active without an API key.'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Source Toggles Strip ───────────────────────────────────────────────────────
SOURCE_LABELS = [
    ("enable_kb", "Knowledge Base"),
    ("enable_memory", "Memory"),
    ("enable_arxiv", "arXiv"),
    ("enable_wiki", "Wikipedia"),
    ("enable_pubmed", "PubMed"),
    ("enable_web", "Web"),
]
src_cols = st.columns(len(SOURCE_LABELS))
for i, (toggle_key, label) in enumerate(SOURCE_LABELS):
    with src_cols[i]:
        current_val = st.session_state.get(toggle_key, SOURCE_DEFAULTS[toggle_key])
        new_val = st.checkbox(
            label,
            value=current_val,
            key=f"src_chk_{toggle_key}",
        )
        st.session_state[toggle_key] = new_val

st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)


# ── Render Conversation History ───────────────────────────────────────────────
if not st.session_state["prompts"]:
    with st.chat_message("assistant"):
        st.markdown(
            "Hello. I am **JANE** — a production research assistant. I analyze queries, "
            "trigger specialized academic and web tools in parallel, and synthesize findings "
            "into structured dossiers with full citations. What are we investigating today?"
        )

ctx_list = st.session_state.get("contexts_list", [])
turn_metrics = st.session_state.get("turn_metrics", [])

for idx, (u, r) in enumerate(zip(st.session_state["prompts"], st.session_state["replies"])):
    with st.chat_message("user"):
        st.write(u)
    with st.chat_message("assistant"):
        st.markdown(r)
        if idx < len(ctx_list) and ctx_list[idx]:
            with st.expander(f"Retrieved Evidence ({len(ctx_list[idx])} sources)", expanded=False):
                for c in ctx_list[idx]:
                    st.markdown(c)
        if idx < len(turn_metrics) and turn_metrics[idx]:
            m = turn_metrics[idx]
            st.markdown(
                f'<div class="telemetry-tag">'
                f'⏱️ Total: {m.get("total_s", 0):.2f}s | '
                f'Orchestration: {m.get("orchestrator_s", 0):.2f}s | '
                f'Retrieval: {m.get("retriever_s", 0):.2f}s | '
                f'Synthesis: {m.get("synthesis_s", 0):.2f}s'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── User Input & Execution Pipeline ───────────────────────────────────────────
prompt = st.chat_input("Ask JANE a research question…")

if prompt:
    t_start = time.perf_counter()

    with st.chat_message("user"):
        st.write(prompt)

    inputs: ResearchState = {
        "prompts": [prompt],
        "replies": [],
        "tool_plan": [],
        "contexts": [],
        "refined_query": "",
        "telemetry": {},
    }

    captured_contexts: List[str] = []
    orchestrator_s = 0.0
    retriever_s = 0.0
    synthesis_s = 0.0
    reply = ""

    with st.chat_message("assistant"):
        status_box = st.empty()
        stream_box = st.empty()

        # Step 1 & 2: Run Orchestrator and Parallel Retriever via Graph
        for chunk in research_graph.stream(inputs, stream_mode="updates"):
            node = next(iter(chunk))
            data = chunk[node]

            if node == "orchestrator":
                plan = data.get("tool_plan", [])
                refined = data.get("refined_query", "")
                labels = ", ".join(TOOL_LABELS.get(t, t) for t in plan)
                orchestrator_s = data.get("telemetry", {}).get("orchestrator_latency", 0.0)
                rewrite_hint = (
                    f" <span style='color:#6e7681;font-size:0.73rem'>(query: {refined})</span>"
                    if refined and refined != prompt else ""
                )
                status_box.markdown(
                    f"**Routing** — {len(plan)} source(s): "
                    f"<span style='color:#58a6ff;font-size:0.76rem'>{labels}</span>"
                    f"{rewrite_hint}",
                    unsafe_allow_html=True,
                )
            elif node == "retriever":
                captured_contexts = data.get("contexts", [])
                retriever_s = data.get("telemetry", {}).get("retriever_latency", 0.0)
                n = len(captured_contexts)
                status_box.markdown(f"**Gathered** — {n} source(s) responded in {retriever_s:.2f}s. Synthesising...")

        status_box.empty()

        # Step 3: Single-Pass Streaming Synthesis (Eliminating redundant double LLM invocation)
        t_synth_start = time.perf_counter()
        ctx_block = (
            f"## Retrieved Context ({len(captured_contexts)} source(s))\n\n"
            + "\n\n---\n\n".join(captured_contexts)
            if captured_contexts else "No external research context was retrieved."
        )

        key = settings.api_key
        model = st.session_state.get("model_name", settings.default_model)

        if not key:
            reply = (
                "> **No Gemini API Key Configured.** Set `GEMINI_API_KEY=...` in `.env` to enable AI synthesis.\n\n"
                "---\n\n" + ctx_block
            )
            stream_box.markdown(reply)
            synthesis_s = time.perf_counter() - t_synth_start
        else:
            try:
                stream_llm = _get_llm(model, key)
                stream_msgs = _build_messages(
                    prompts=st.session_state["prompts"],
                    replies=st.session_state["replies"],
                    ctx_block=ctx_block,
                    current_prompt=prompt,
                )

                streamed_tokens: List[str] = []
                with stream_box:
                    placeholder = st.empty()
                    for tok in stream_llm.stream(stream_msgs):
                        streamed_tokens.append(tok.content)
                        placeholder.markdown("".join(streamed_tokens) + " ▌")
                    placeholder.markdown("".join(streamed_tokens))

                reply = "".join(streamed_tokens)
                synthesis_s = time.perf_counter() - t_synth_start
            except Exception as stream_exc:
                app_logger.error(f"Streaming error: {stream_exc}")
                reply = f"**Synthesis Error:** `{stream_exc}`\n\n---\n\n{ctx_block}"
                stream_box.markdown(reply)
                synthesis_s = time.perf_counter() - t_synth_start

    total_s = time.perf_counter() - t_start

    # Record turn metrics
    metrics = {
        "total_s": total_s,
        "orchestrator_s": orchestrator_s,
        "retriever_s": retriever_s,
        "synthesis_s": synthesis_s,
    }

    st.session_state["prompts"].append(prompt)
    st.session_state["replies"].append(reply)
    st.session_state.setdefault("contexts_list", []).append(captured_contexts)
    st.session_state.setdefault("turn_metrics", []).append(metrics)

    # Save session
    saved_id = save_chat_session(
        prompts=st.session_state["prompts"],
        replies=st.session_state["replies"],
        folder_path=settings.past_chats_dir,
        session_id=st.session_state.get("session_id"),
        latest_contexts=captured_contexts,
    )
    st.session_state["session_id"] = saved_id

    # Incremental memory index update (only embeds new turn)
    if has_key:
        try:
            build_memory_index(
                google_api_key=key,
                folder_path=settings.past_chats_dir,
                scans_dir=settings.scans_dir,
            )
        except Exception as exc:
            app_logger.warning(f"Memory index refresh error: {exc}")

    st.rerun()