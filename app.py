"""
app.py — JANE (Just A Nuanced Engine)
--------------------------------------
Three-node LangGraph multi-agent pipeline.
API key read from .env only. Source toggles live in the main chat area.
Conversation history (with retrieved context) shown at the top of the sidebar.
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

from retriever import (
    save_chat_session,
    delete_chat_session,
    load_all_chat_sessions,
    build_knowledge_base_if_needed,
    build_memory_index,
)
from mcp_server import mcp

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JANE",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

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
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 0.69rem;
    font-weight: 500;
    margin-top: 4px;
}
.pill-green  { background:#0d2a1f; color:#3fb950; border:1px solid #238636; }
.pill-amber  { background:#271d0a; color:#d29922; border:1px solid #9e6a03; }
.pill-blue   { background:#0c1a2e; color:#58a6ff; border:1px solid #1f6feb; }
.pill-red    { background:#2d0d0d; color:#f85149; border:1px solid #da3633; }

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

/* Source toggles strip */
.source-strip {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 0.8rem;
    padding: 8px 12px;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
}
.src-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
    cursor: pointer;
    border: 1px solid #30363d;
    background: transparent;
    color: #8b949e;
    transition: all 0.12s ease;
    user-select: none;
}
.src-chip.on {
    background: #0c1a2e;
    border-color: #1f6feb;
    color: #58a6ff;
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

/* Hide Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ─── State schema ─────────────────────────────────────────────────────────────
class ResearchState(TypedDict):
    prompts:   Annotated[List[str], operator.add]
    replies:   Annotated[List[str], operator.add]
    tool_plan: List[str]
    contexts:  Annotated[List[str], operator.add]


# ─── Startup index builds ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _init_indices() -> Dict[str, str]:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    out: Dict[str, str] = {}
    try:
        out["kb"] = build_knowledge_base_if_needed(
            books_dir="books", scans_dir="scans", google_api_key=key
        )
    except Exception as exc:
        out["kb"] = f"error:{exc}"
    try:
        out["memory"] = build_memory_index(
            google_api_key=key, folder_path="past_chats", scans_dir="scans"
        )
    except Exception as exc:
        out["memory"] = f"error:{exc}"
    return out

idx_status = _init_indices()


# ─── Session helpers ──────────────────────────────────────────────────────────
def _all_sessions() -> List[Dict[str, Any]]:
    sessions = load_all_chat_sessions("past_chats")
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
    st.session_state["prompts"]    = [t.get("prompt", "") for t in chats]
    st.session_state["replies"]    = [t.get("reply",  "") for t in chats]
    st.session_state["session_id"] = s.get("session_id")

def _new_chat() -> None:
    st.session_state["prompts"] = []
    st.session_state["replies"] = []
    st.session_state.pop("session_id", None)


# ─── Source toggle defaults ───────────────────────────────────────────────────
SOURCE_DEFAULTS: Dict[str, bool] = {
    "enable_kb":     True,
    "enable_memory": True,
    "enable_arxiv":  True,
    "enable_wiki":   True,
    "enable_pubmed": False,
    "enable_web":    True,
}
for _k, _v in SOURCE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─── MCP tool invocation ──────────────────────────────────────────────────────
def _invoke_tool(tool_name: str, args: Dict[str, Any]) -> tuple[str, str]:
    try:
        blocks, _ = asyncio.run(mcp.call_tool(tool_name, args))
        return tool_name, (blocks[0].text if blocks else "")
    except Exception as exc:
        logger.warning("Tool %s: %s", tool_name, exc)
        return tool_name, ""


# ─── Tool metadata ────────────────────────────────────────────────────────────
TOOL_LABELS: Dict[str, str] = {
    "knowledge_base_search": "Knowledge Base",
    "memory_search":         "Memory",
    "arxiv_search":          "arXiv",
    "wikipedia_search":      "Wikipedia",
    "pubmed_search":         "PubMed",
    "web_search":            "Web",
}
TOGGLE_MAP: Dict[str, str] = {
    "enable_kb":     "knowledge_base_search",
    "enable_memory": "memory_search",
    "enable_arxiv":  "arxiv_search",
    "enable_wiki":   "wikipedia_search",
    "enable_pubmed": "pubmed_search",
    "enable_web":    "web_search",
}

_ACADEMIC   = {"research","paper","study","journal","arxiv","model","algorithm",
               "method","theorem","proof","experiment","dataset","neural","ml","ai"}
_BIOMEDICAL = {"drug","disease","clinical","patient","gene","protein","therapy",
               "medicine","health","symptom","treatment","diagnosis"}
_FACTUAL    = {"what is","what are","define","definition","explain","meaning","who is","when was"}
_NEWS       = {"latest","recent","current","new","update","2024","2025","2026","news","today"}


# ─── Node 1: Orchestrator ─────────────────────────────────────────────────────
def orchestrator_node(state: ResearchState) -> Dict[str, Any]:
    query  = state["prompts"][-1]
    ql     = query.lower()
    words  = set(ql.split())
    key    = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""

    enabled = [tool for toggle, tool in TOGGLE_MAP.items()
               if st.session_state.get(toggle, SOURCE_DEFAULTS.get(toggle, True))]
    if not key:
        enabled = [t for t in enabled if t not in ("knowledge_base_search","memory_search")]

    scores: Dict[str, int] = {t: 10 for t in enabled}
    if words & _ACADEMIC:
        for t in ("knowledge_base_search","arxiv_search"):
            if t in scores: scores[t] += 8
    if words & _BIOMEDICAL:
        for t in ("pubmed_search","arxiv_search"):
            if t in scores: scores[t] += 8
    if any(k in ql for k in _FACTUAL):
        for t in ("wikipedia_search","knowledge_base_search"):
            if t in scores: scores[t] += 6
    if words & _NEWS:
        if "web_search" in scores: scores["web_search"] += 8
    if "memory_search"         in scores: scores["memory_search"]         += 4
    if "knowledge_base_search" in scores: scores["knowledge_base_search"] += 3

    return {"tool_plan": sorted(enabled, key=lambda t: scores.get(t,0), reverse=True)}


# ─── Node 2: Parallel Retriever ───────────────────────────────────────────────
def retriever_node(state: ResearchState) -> Dict[str, Any]:
    query     = state["prompts"][-1]
    tool_plan = state.get("tool_plan", [])
    key       = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""

    calls: List[tuple[str, Dict]] = []
    for tool in tool_plan:
        if tool in ("knowledge_base_search","memory_search"):
            calls.append((tool, {"query": query, "api_key": key,
                                 "top_k": 4 if tool == "knowledge_base_search" else 3}))
        elif tool == "arxiv_search":
            calls.append((tool, {"query": query, "max_results": 3}))
        elif tool == "wikipedia_search":
            calls.append((tool, {"query": query, "max_results": 2}))
        elif tool == "pubmed_search":
            calls.append((tool, {"query": query, "max_results": 3}))
        elif tool == "web_search":
            calls.append((tool, {"query": query, "max_results": 3}))

    if not calls:
        return {"contexts": []}

    contexts: List[str] = []
    with ThreadPoolExecutor(max_workers=min(len(calls), 8)) as ex:
        futures = {ex.submit(_invoke_tool, n, a): n for n, a in calls}
        for future in as_completed(futures, timeout=60):
            tool_name = futures[future]
            try:
                _, result = future.result()
                if result and result.strip() and len(result) > 30:
                    contexts.append(f"### {TOOL_LABELS.get(tool_name, tool_name)}\n{result}")
            except Exception as exc:
                logger.warning("Future %s: %s", tool_name, exc)

    return {"contexts": contexts}


# ─── Node 3: Synthesiser ──────────────────────────────────────────────────────
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
        model=model_name, google_api_key=google_key,
        temperature=0.35, max_retries=3,
    )


def synthesiser_node(state: ResearchState) -> Dict[str, Any]:
    query    = state["prompts"][-1]
    contexts = state.get("contexts", [])
    key      = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    model    = st.session_state.get("model_name", "gemini-2.0-flash")

    ctx_block = (
        f"## Retrieved Context ({len(contexts)} source(s))\n\n"
        + "\n\n---\n\n".join(contexts)
        if contexts else "No external research context was retrieved."
    )

    if not key:
        return {"replies": [
            "> **No Gemini API key.** Add `GOOGLE_API_KEY=...` to `.env` and restart.\n\n"
            "---\n\n" + ctx_block
        ]}
    try:
        llm = _get_llm(model, key)
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        messages.append(HumanMessage(content=ctx_block))
        for p, r in zip(state["prompts"][:-1], state["replies"]):
            messages.append(HumanMessage(content=p))
            messages.append(AIMessage(content=r))
        messages.append(HumanMessage(content=query))
        response = llm.invoke(messages)
        return {"replies": [response.content]}
    except Exception as exc:
        err = str(exc).lower()
        if "api key" in err or "invalid" in err or "unauthorized" in err:
            msg = "**Auth error:** Gemini API key invalid or expired."
        elif "quota" in err or "429" in err or "rate" in err:
            msg = "**Rate limit:** Gemini quota exceeded — wait and retry."
        elif "network" in err or "connection" in err:
            msg = "**Network error:** Could not reach Gemini API."
        else:
            msg = f"**Error:** `{str(exc)}`"
        return {"replies": [f"{msg}\n\n---\n\n{ctx_block}"]}


# ─── Graph ────────────────────────────────────────────────────────────────────
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
    st.markdown("## JANE")
    st.markdown("---")

    # ── Conversations (TOP) ───────────────────────────────────────────────────
    st.markdown('<p class="sidebar-label">Conversations</p>', unsafe_allow_html=True)

    # Search filter
    search_q = st.text_input(
        "search", placeholder="Search…",
        label_visibility="collapsed", key="sess_search"
    )

    # New chat button
    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("+ New Chat", key="new_chat_btn", use_container_width=True):
        _new_chat()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Session list
    sessions     = _all_sessions()
    current_sid  = st.session_state.get("session_id")
    sq           = (search_q or "").strip().lower()
    filtered     = [s for s in sessions if not sq or sq in _session_title(s).lower()]

    for sess in filtered:
        sid      = sess.get("session_id", "")
        title    = _session_title(sess)
        date     = _session_date(sess)
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

    # ── Model ─────────────────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-label">Model</p>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "model", label_visibility="collapsed",
        options=["gemini-2.0-flash","gemini-2.0-flash-lite","gemini-1.5-flash","gemini-1.5-pro"],
        index=0,
    )
    st.session_state["model_name"] = model_choice

    # ── KB status ─────────────────────────────────────────────────────────────
    kb_s = idx_status.get("kb","")
    if   kb_s == "built":      st.markdown('<span class="pill pill-green">KB built</span>',   unsafe_allow_html=True)
    elif kb_s == "up_to_date": st.markdown('<span class="pill pill-blue">KB up to date</span>', unsafe_allow_html=True)
    elif kb_s == "no_books":   st.markdown('<span class="pill pill-amber">No books found</span>', unsafe_allow_html=True)
    elif kb_s == "no_api_key": st.markdown('<span class="pill pill-amber">KB needs API key</span>', unsafe_allow_html=True)


# ─── Main area ────────────────────────────────────────────────────────────────
has_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))

st.markdown(
    '<div class="app-header">'
    '<h1>JANE</h1>'
    '<span>Just A Nuanced Engine &nbsp;·&nbsp; Multi-agent parallel research</span>'
    '</div>',
    unsafe_allow_html=True,
)

if not has_key:
    st.markdown(
        '<div class="no-key-banner">'
        'No API key found. Add <code>GOOGLE_API_KEY=your_key</code> to <code>.env</code> and restart. '
        'arXiv, Wikipedia, and Web Search still work without it.'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Source toggles strip (in main chat area) ───────────────────────────────────
SOURCE_LABELS = [
    ("enable_kb",     "Knowledge Base"),
    ("enable_memory", "Memory"),
    ("enable_arxiv",  "arXiv"),
    ("enable_wiki",   "Wikipedia"),
    ("enable_pubmed", "PubMed"),
    ("enable_web",    "Web"),
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

# ── Conversation render ────────────────────────────────────────────────────────
if not st.session_state["prompts"]:
    with st.chat_message("assistant"):
        st.markdown(
            "Hello. I'm JANE — I route your query through a heuristic orchestrator, "
            "fire all relevant sources in parallel, then synthesise everything into a "
            "structured research report. What would you like to explore?"
        )

for u, r in zip(st.session_state["prompts"], st.session_state["replies"]):
    with st.chat_message("user"):
        st.write(u)
    with st.chat_message("assistant"):
        st.markdown(r)

prompt = st.chat_input("Ask JANE a research question…")

if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    inputs: ResearchState = {
        "prompts":   [prompt],
        "replies":   [],
        "tool_plan": [],
        "contexts":  [],
    }

    reply              = ""
    captured_contexts: List[str] = []

    with st.chat_message("assistant"):
        status_box = st.empty()

        for chunk in research_graph.stream(inputs, stream_mode="updates"):
            node = next(iter(chunk))
            data = chunk[node]

            if node == "orchestrator":
                plan   = data.get("tool_plan", [])
                labels = ", ".join(TOOL_LABELS.get(t, t) for t in plan)
                status_box.markdown(
                    f"**Routing** — {len(plan)} source(s): "
                    f"<span style='color:#484f58;font-size:0.76rem'>{labels}</span>",
                    unsafe_allow_html=True,
                )
            elif node == "retriever":
                captured_contexts = data.get("contexts", [])
                n = len(captured_contexts)
                status_box.markdown(f"**Gathering** — {n} source(s) responded. Synthesising…")
            elif node == "synthesiser":
                replies = data.get("replies", [])
                if replies:
                    reply = replies[-1]

        status_box.empty()
        if reply:
            st.markdown(reply)
        else:
            reply = "No response generated. Check your API key and connection."
            st.warning(reply)

    st.session_state["prompts"].append(prompt)
    st.session_state["replies"].append(reply)

    # Save session with retrieved context attached to this turn
    saved_id = save_chat_session(
        prompts=st.session_state["prompts"],
        replies=st.session_state["replies"],
        folder_path="past_chats",
        session_id=st.session_state.get("session_id"),
        latest_contexts=captured_contexts,
    )
    st.session_state["session_id"] = saved_id

    # Refresh memory index so next query sees this turn
    if has_key:
        try:
            build_memory_index(
                google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
                folder_path="past_chats",
                scans_dir="scans",
            )
        except Exception as _e:
            logger.warning("Memory refresh: %s", _e)

    st.rerun()