import operator
import os
import asyncio
from typing import List, Dict, Any, Optional
from typing_extensions import Annotated, TypedDict
import streamlit as st
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from retriever import (
    save_chat_session,
    search_past_chats,
    build_knowledge_base_if_needed,
    search_knowledge_base,
)
from mcp_server import mcp

# Load environment variables
load_dotenv()

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── State Schema ─────────────────────────────────────────────────────────────
class CurSession(TypedDict):
    prompts: Annotated[List[str], operator.add]
    replies: Annotated[List[str], operator.add]

# ─── Knowledge Base — build once at startup ───────────────────────────────────
@st.cache_resource(show_spinner=False)
def _init_knowledge_base():
    """
    Runs once per Streamlit process. Checks whether books/ has changed and
    rebuilds the FAISS index in scans/ only when needed.
    """
    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    try:
        status = build_knowledge_base_if_needed(
            books_dir="books",
            scans_dir="scans",
            google_api_key=google_api_key,
        )
        return status
    except Exception as e:
        return f"error: {e}"

kb_init_status = _init_knowledge_base()

# ─── MCP Tool Caller ──────────────────────────────────────────────────────────
def call_mcp_server_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Execute a registered FastMCP tool and return its text output."""
    try:
        content_blocks, _ = asyncio.run(mcp.call_tool(tool_name, arguments))
        if content_blocks:
            return content_blocks[0].text
        return f"No output returned from MCP tool {tool_name}."
    except Exception as e:
        return f"MCP server tool error ({tool_name}): {str(e)}"

# ─── Web Search ───────────────────────────────────────────────────────────────
def perform_web_search(query: str, max_results: int = 3) -> str:
    """DuckDuckGo web search via the ddgs package."""
    try:
        from ddgs import DDGS
        results = list(DDGS().text(query, max_results=max_results))
        if not results:
            return "No web results found."
        snippets = []
        for i, res in enumerate(results, 1):
            title   = res.get("title", "No Title")
            snippet = res.get("body", res.get("snippet", ""))
            href    = res.get("href", res.get("link", ""))
            snippets.append(f"[{i}] {title}\nURL: {href}\nSummary: {snippet}")
        return "\n\n".join(snippets)
    except Exception as e:
        return f"Web search unavailable ({str(e)})."

# ─── LLM Factory ──────────────────────────────────────────────────────────────
def _get_llm(model_name: str, openai_key: Optional[str], google_key: Optional[str]):
    """
    Return the appropriate LangChain chat model based on selected model name.
    Supports OpenAI (gpt-*) and Google Gemini (gemini-*).
    """
    if model_name.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not google_key:
            raise ValueError("Google / Gemini API key is required for Gemini models.")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_key,
            temperature=0.4,
            convert_system_message_to_human=False,
        )
    else:
        from langchain_openai import ChatOpenAI
        if not openai_key:
            raise ValueError("OpenAI API key is required for GPT models.")
        return ChatOpenAI(
            model=model_name,
            openai_api_key=openai_key,
            temperature=0.4,
        )

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert AI Research Assistant with access to multiple knowledge sources:
  • Academic papers (arXiv, PubMed)
  • Encyclopaedic knowledge (Wikipedia)
  • General web search (DuckDuckGo)
  • A local knowledge base of curated books and documents
  • Memory of past conversations

## Your Reasoning Protocol
Before writing your answer, silently work through these steps:
  1. **Understand** — Identify exactly what the user is asking. Distinguish factual questions, conceptual explanations, comparisons, and open-ended research queries.
  2. **Evaluate sources** — Critically assess each provided context block. Note which sources are authoritative (peer-reviewed papers, textbooks) vs. secondary (web snippets). Discard irrelevant or contradictory snippets.
  3. **Synthesise** — Do NOT just copy-paste context. Extract key concepts, compare perspectives, identify consensus and gaps. Apply your own domain knowledge to fill in missing context.
  4. **Structure** — Write a well-organised Markdown report using the output format below.

## Output Format
Always structure responses as follows:

### 🔍 Overview
A concise 2–4 sentence summary answering the core question directly.

### 📌 Key Findings
Use bullet points or numbered lists for distinct insights. Each point should be a synthesis, not a copy of the raw context.

### 📚 Sources & Evidence
Cite specific sources with URLs where available. Format: `[Source Name](URL) — one-line description`.

### 💡 Conclusions & Further Research
A brief synthesis paragraph. Suggest follow-up angles or open questions where appropriate.

## Rules
- Be precise and technically rigorous. Prefer specificity over vagueness.
- If the context is insufficient to answer confidently, say so explicitly rather than hallucinating.
- Keep tone professional but accessible.
- Use inline code formatting for technical terms, model names, formulas, etc.
"""

# ─── Research Agent Node ──────────────────────────────────────────────────────
def research_agent_node(state: CurSession) -> Dict[str, Any]:
    """
    LangGraph node: gathers context from all enabled sources, then calls the
    LLM with a structured system prompt to synthesise a research report.
    """
    if not state["prompts"]:
        return {"replies": ["No user prompt found to process."]}

    latest_prompt = state["prompts"][-1]

    # ── Retrieve config ──────────────────────────────────────────────────────
    openai_key  = st.session_state.get("openai_api_key")  or os.getenv("OPENAI_API_KEY")
    google_key  = st.session_state.get("google_api_key")  or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    model_name  = st.session_state.get("model_name", "gpt-4o-mini")

    enable_search    = st.session_state.get("enable_search", True)
    enable_arxiv     = st.session_state.get("enable_arxiv", True)
    enable_wiki      = st.session_state.get("enable_wiki", True)
    enable_pubmed    = st.session_state.get("enable_pubmed", False)
    enable_past_chats = st.session_state.get("enable_past_chats", True)
    enable_kb        = st.session_state.get("enable_kb", True)

    collected_contexts = []

    # ── 1. Local Knowledge Base (books → scans FAISS) ────────────────────────
    if enable_kb:
        with st.spinner("🔎 Searching local knowledge base..."):
            kb_context = search_knowledge_base(
                query=latest_prompt,
                google_api_key=google_key,
                scans_dir="scans",
                top_k=4,
            )
            if kb_context:
                collected_contexts.append(f"### 📚 Local Knowledge Base (Books):\n{kb_context}")

    # ── 2. Past Chat Memory ──────────────────────────────────────────────────
    if enable_past_chats:
        with st.spinner("🧠 Searching past chat memory..."):
            past_context = search_past_chats(
                query=latest_prompt,
                google_api_key=google_key,
                folder_path="past_chats",
                top_k=3,
            )
            if past_context and "No relevant past" not in past_context and "No past chat" not in past_context:
                collected_contexts.append(f"### 🗂️ Relevant Past Conversations:\n{past_context}")

    # ── 3. arXiv Papers ──────────────────────────────────────────────────────
    if enable_arxiv:
        with st.spinner("📄 Fetching arXiv papers..."):
            arxiv_res = call_mcp_server_tool("arxiv_search", {"query": latest_prompt, "max_results": 3})
            if arxiv_res and "No arXiv" not in arxiv_res and "error" not in arxiv_res.lower():
                collected_contexts.append(f"### 🎓 arXiv Academic Papers:\n{arxiv_res}")

    # ── 4. Wikipedia ─────────────────────────────────────────────────────────
    if enable_wiki:
        with st.spinner("🌐 Querying Wikipedia..."):
            wiki_res = call_mcp_server_tool("wikipedia_search", {"query": latest_prompt, "max_results": 2})
            if wiki_res and "No Wikipedia" not in wiki_res:
                collected_contexts.append(f"### 📖 Wikipedia:\n{wiki_res}")

    # ── 5. PubMed Literature ─────────────────────────────────────────────────
    if enable_pubmed:
        with st.spinner("🧬 Searching PubMed..."):
            pubmed_res = call_mcp_server_tool("pubmed_search", {"query": latest_prompt, "max_results": 3})
            if pubmed_res and "No PubMed" not in pubmed_res:
                collected_contexts.append(f"### 🧬 PubMed Literature:\n{pubmed_res}")

    # ── 6. General Web Search ────────────────────────────────────────────────
    if enable_search:
        with st.spinner("🔍 Running web search..."):
            web_res = perform_web_search(latest_prompt, max_results=3)
            if web_res and "unavailable" not in web_res and "No web results" not in web_res:
                collected_contexts.append(f"### 🌍 Web Search:\n{web_res}")

    # ── Assemble context block ───────────────────────────────────────────────
    if collected_contexts:
        combined_context = "\n\n---\n\n".join(collected_contexts)
        context_header = (
            f"## Research Context\n\n"
            f"The following information was retrieved from {len(collected_contexts)} source(s). "
            f"Use this to inform your response:\n\n{combined_context}"
        )
    else:
        context_header = "No external research context was collected."

    # ── Build LLM messages ───────────────────────────────────────────────────
    has_openai_key  = bool(openai_key)
    has_gemini_key  = bool(google_key)
    can_use_llm     = (model_name.startswith("gemini") and has_gemini_key) or \
                      (not model_name.startswith("gemini") and has_openai_key)

    if can_use_llm:
        try:
            llm = _get_llm(model_name, openai_key, google_key)

            messages = [SystemMessage(content=SYSTEM_PROMPT)]

            # Inject context as a human message so the LLM can reason over it
            messages.append(HumanMessage(content=context_header))

            # Re-inject conversation history
            for p, r in zip(state["prompts"][:-1], state["replies"]):
                messages.append(HumanMessage(content=p))
                messages.append(AIMessage(content=r))

            messages.append(HumanMessage(content=latest_prompt))

            response = llm.invoke(messages)
            reply_text = response.content

        except Exception as e:
            reply_text = (
                f"**⚠️ LLM Error:** `{str(e)}`\n\n"
                f"---\n\n{context_header}"
            )
    else:
        missing = []
        if not model_name.startswith("gemini") and not has_openai_key:
            missing.append("OpenAI API key")
        if model_name.startswith("gemini") and not has_gemini_key:
            missing.append("Google / Gemini API key")
        reply_text = (
            f"**🔑 Missing API Key(s): {', '.join(missing)}**\n\n"
            f"Please enter the required key(s) in the sidebar.\n\n"
            f"---\n\n{context_header}"
        )

    return {"replies": [reply_text]}

# ─── LangGraph Workflow ───────────────────────────────────────────────────────
def build_research_graph():
    builder = StateGraph(CurSession)
    builder.add_node("researcher", research_agent_node)
    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", END)
    return builder.compile()

research_app_graph = build_research_graph()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown("### 🔑 API Keys")
    openai_input = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Required for GPT-4o / GPT-3.5 models.",
    )
    if openai_input:
        st.session_state["openai_api_key"] = openai_input

    google_input = st.text_input(
        "Google / Gemini API Key",
        value=os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", "")),
        type="password",
        help="Required for Gemini models and knowledge base embeddings.",
    )
    if google_input:
        st.session_state["google_api_key"] = google_input

    st.markdown("### 🤖 Model")
    model_choice = st.selectbox(
        "Select Model",
        options=[
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ],
        index=0,
        help="GPT models require an OpenAI key. Gemini models require a Google key.",
    )
    st.session_state["model_name"] = model_choice

    st.markdown("### 🔧 Research Sources")
    st.session_state["enable_kb"]         = st.toggle("📚 Local Knowledge Base (Books)", value=True)
    st.session_state["enable_past_chats"] = st.toggle("🧠 Past Chat Memory (RAG)",       value=True)
    st.session_state["enable_arxiv"]      = st.toggle("🎓 arXiv Papers (MCP)",            value=True)
    st.session_state["enable_wiki"]       = st.toggle("📖 Wikipedia (MCP)",               value=True)
    st.session_state["enable_pubmed"]     = st.toggle("🧬 PubMed Literature (MCP)",       value=False)
    st.session_state["enable_search"]     = st.toggle("🌍 Web Search (DuckDuckGo)",       value=True)

    # Knowledge base status indicator
    st.markdown("---")
    st.markdown("### 📦 Knowledge Base")
    if kb_init_status == "built":
        st.success("✅ Index built from books/")
    elif kb_init_status == "up_to_date":
        st.info("✅ Index up to date")
    elif kb_init_status == "no_books":
        st.warning("📂 No books found in `books/` — add `.pdf`, `.txt`, or `.md` files.")
    elif kb_init_status == "no_api_key":
        st.warning("🔑 Set a Google API key to enable the knowledge base.")
    else:
        st.error(f"KB Error: {kb_init_status}")

    st.divider()
    if st.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state["prompts"] = []
        st.session_state["replies"] = []
        if "session_id" in st.session_state:
            del st.session_state["session_id"]
        st.rerun()

# ─── Main UI ──────────────────────────────────────────────────────────────────
st.title("🔬 AI Research Agent")
st.markdown(
    "Powered by **LangGraph** · **FastMCP** · **Gemini Embeddings** · **FAISS Knowledge Base**"
)

# Init session state
if "prompts" not in st.session_state:
    st.session_state["prompts"] = []
if "replies" not in st.session_state:
    st.session_state["replies"] = []

# Welcome message
if not st.session_state["prompts"]:
    with st.chat_message("assistant"):
        st.write(
            "Hello! I'm your AI Research Assistant. I can search arXiv, Wikipedia, PubMed, "
            "the web, your personal book collection, and our past conversations to give you "
            "a structured, well-reasoned research report. What would you like to explore?"
        )

# Render conversation history
for user_p, bot_r in zip(st.session_state["prompts"], st.session_state["replies"]):
    with st.chat_message("user"):
        st.write(user_p)
    with st.chat_message("assistant"):
        st.markdown(bot_r)

# Chat input
prompt = st.chat_input("Ask a research question...")

if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    inputs: CurSession = {"prompts": [prompt], "replies": []}

    with st.chat_message("assistant"):
        with st.spinner("Researching and synthesising..."):
            result = research_app_graph.invoke(inputs)
            latest_reply = result["replies"][-1]
            st.markdown(latest_reply)

    st.session_state["prompts"].append(prompt)
    st.session_state["replies"].append(latest_reply)

    # Persist session
    session_id = st.session_state.get("session_id")
    saved_id = save_chat_session(
        prompts=st.session_state["prompts"],
        replies=st.session_state["replies"],
        folder_path="past_chats",
        session_id=session_id,
    )
    st.session_state["session_id"] = saved_id