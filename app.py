import operator
import os
import asyncio
from typing import List, Dict, Any, Optional
from typing_extensions import Annotated, TypedDict
import streamlit as st
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from retriever import save_chat_session, search_past_chats
from mcp_server import mcp

# Load environment variables if available
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="AI Research Agent",
    layout="wide"
)

# 1. State Definition matching user's original schema
class CurSession(TypedDict):
    prompts: Annotated[List[str], operator.add]
    replies: Annotated[List[str], operator.add]

# Helper function to invoke tools via FastMCP server
def call_mcp_server_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Executes research tools through the FastMCP server instance.
    """
    try:
        content_blocks, _ = asyncio.run(mcp.call_tool(tool_name, arguments))
        if content_blocks:
            return content_blocks[0].text
        return f"No output returned from MCP tool {tool_name}."
    except Exception as e:
        return f"MCP server tool error ({tool_name}): {str(e)}"

# Helper function for web research using ddgs
def perform_web_search(query: str, max_results: int = 3) -> str:
    """Performs web search using ddgs package to retrieve relevant content."""
    try:
        from ddgs import DDGS
        results = list(DDGS().text(query, max_results=max_results))
        if not results:
            return "No web results found."
        formatted_snippets = []
        for i, res in enumerate(results, 1):
            title = res.get("title", "No Title")
            snippet = res.get("body", res.get("snippet", ""))
            href = res.get("href", res.get("link", ""))
            formatted_snippets.append(f"[{i}] {title}\nURL: {href}\nSummary: {snippet}")
        return "\n\n".join(formatted_snippets)
    except Exception as e:
        return f"Web search unavailable ({str(e)})."

# 2. Graph Node Function: takes user message, runs enabled MCP server tools & LLM response
def research_agent_node(state: CurSession) -> Dict[str, Any]:
    """
    LangGraph node function that takes the current state,
    invokes tools strictly through the FastMCP server (arxiv_search, wikipedia_search, pubmed_search),
    and synthesizes a comprehensive research report.
    """
    if not state["prompts"]:
        return {"replies": ["No user prompt found to process."]}

    latest_prompt = state["prompts"][-1]
    
    # Retrieve configuration from session state or env
    api_key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    google_api_key = st.session_state.get("google_api_key") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    model_name = st.session_state.get("model_name", "gpt-4o-mini")
    
    enable_search = st.session_state.get("enable_search", True)
    enable_arxiv = st.session_state.get("enable_arxiv", True)
    enable_wiki = st.session_state.get("enable_wiki", True)
    enable_pubmed = st.session_state.get("enable_pubmed", False)
    enable_past_chats = st.session_state.get("enable_past_chats", True)

    collected_contexts = []

    # 1. Past Chat History Context (Gemini Embeddings RAG)
    if enable_past_chats:
        with st.spinner("Searching past chat memory (Gemini Embeddings)..."):
            past_chats_context = search_past_chats(
                query=latest_prompt,
                google_api_key=google_api_key,
                folder_path="past_chats",
                top_k=3
            )
            if past_chats_context and "No relevant past" not in past_chats_context and "No past chat" not in past_chats_context:
                collected_contexts.append(f"### Relevant Past Chat History:\n{past_chats_context}")

    # 2. arXiv Papers via FastMCP Server
    if enable_arxiv:
        with st.spinner("Invoking arxiv_search via FastMCP Server..."):
            arxiv_res = call_mcp_server_tool("arxiv_search", {"query": latest_prompt, "max_results": 3})
            if arxiv_res and "No arXiv" not in arxiv_res:
                collected_contexts.append(f"### arXiv Academic Papers (via MCP Server):\n{arxiv_res}")

    # 3. Wikipedia Summary via FastMCP Server
    if enable_wiki:
        with st.spinner("Invoking wikipedia_search via FastMCP Server..."):
            wiki_res = call_mcp_server_tool("wikipedia_search", {"query": latest_prompt, "max_results": 2})
            if wiki_res and "No Wikipedia" not in wiki_res:
                collected_contexts.append(f"### Wikipedia Encyclopedia Summary (via MCP Server):\n{wiki_res}")

    # 4. PubMed Literature via FastMCP Server
    if enable_pubmed:
        with st.spinner("Invoking pubmed_search via FastMCP Server..."):
            pubmed_res = call_mcp_server_tool("pubmed_search", {"query": latest_prompt, "max_results": 3})
            if pubmed_res and "No PubMed" not in pubmed_res:
                collected_contexts.append(f"### PubMed Research Literature (via MCP Server):\n{pubmed_res}")

    # 5. Web Search Context
    if enable_search:
        with st.spinner("Searching general web..."):
            search_context = perform_web_search(latest_prompt, max_results=3)
            if search_context and "unavailable" not in search_context and "No web results" not in search_context:
                collected_contexts.append(f"### Web Search Context:\n{search_context}")

    combined_context_text = "\n\n---\n\n".join(collected_contexts) if collected_contexts else "No external research context collected."

    if api_key:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                temperature=0.7
            )

            system_instruction = (
                "You are an expert AI Research Assistant equipped with tools exposed via FastMCP Server (arXiv, PubMed, Wikipedia, Web Search, and Past Chat Memory).\n"
                "Provide detailed, well-structured, and rigorous research reports based on the user prompt and provided research context.\n"
                "Use clear headings, bullet points, citations of arXiv/PubMed URLs, and summarize key technical insights."
            )
            
            messages = [SystemMessage(content=system_instruction)]

            if collected_contexts:
                messages.append(HumanMessage(content=f"Research Context Collected from MCP Server Tools:\n\n{combined_context_text}"))

            # Add previous conversation turn history
            for p, r in zip(state["prompts"][:-1], state["replies"]):
                messages.append(HumanMessage(content=p))
                messages.append(AIMessage(content=r))

            messages.append(HumanMessage(content=latest_prompt))

            response = llm.invoke(messages)
            reply_text = response.content
        except Exception as e:
            reply_text = (
                f"**Error invoking LLM Model:** {str(e)}\n\n"
                f"--- \n### Research Context Collected from MCP Server:\n\n{combined_context_text}"
            )
    else:
        # Informative response when no OpenAI API key is configured yet
        reply_text = (
            "**OpenAI API Key is missing.** Please enter your API key in the sidebar to generate AI research synthesis.\n\n"
            f"### Research Context Collected from MCP Server:\n\n{combined_context_text}"
        )

    return {"replies": [reply_text]}

# 3. LangGraph Workflow Construction
def build_research_graph():
    builder = StateGraph(CurSession)
    builder.add_node("researcher", research_agent_node)
    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", END)
    return builder.compile()

research_app_graph = build_research_graph()

# 4. Streamlit UI Interface
st.title("AI Research Agent with FastMCP Server")
st.markdown("Powered by **LangGraph**, **FastMCP Server**, and **Gemini Embeddings**.")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration & Keys")
    api_key_input = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Enter your OpenAI API key to enable LLM research synthesis."
    )
    if api_key_input:
        st.session_state["openai_api_key"] = api_key_input

    google_api_key_input = st.text_input(
        "Google / Gemini API Key",
        value=os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", "")),
        type="password",
        help="Enter your Google/Gemini API key for free Gemini embeddings & vector retrieval."
    )
    if google_api_key_input:
        st.session_state["google_api_key"] = google_api_key_input
        
    model_choice = st.selectbox(
        "Select Model",
        options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0
    )
    st.session_state["model_name"] = model_choice

    st.subheader("FastMCP Server Tools")
    st.session_state["enable_search"] = st.toggle("General Web Search (DuckDuckGo)", value=True)
    st.session_state["enable_arxiv"] = st.toggle("arXiv Academic Papers (MCP)", value=True)
    st.session_state["enable_wiki"] = st.toggle("Wikipedia Summary (MCP)", value=True)
    st.session_state["enable_pubmed"] = st.toggle("PubMed Literature (MCP)", value=False)
    st.session_state["enable_past_chats"] = st.toggle("Past Chat Memory (Gemini RAG)", value=True)
    
    st.divider()
    if st.button("Clear Chat History", type="secondary"):
        st.session_state["prompts"] = []
        st.session_state["replies"] = []
        if "session_id" in st.session_state:
            del st.session_state["session_id"]
        st.rerun()

# Initialize Streamlit Session State for CurSession state tracking
if "prompts" not in st.session_state:
    st.session_state["prompts"] = []
if "replies" not in st.session_state:
    st.session_state["replies"] = []

# Display welcome message from assistant if conversation is empty
if not st.session_state["prompts"]:
    with st.chat_message("assistant"):
        st.write("Hello! What do you want to research today?")

# Render past chat history
for user_p, bot_r in zip(st.session_state["prompts"], st.session_state["replies"]):
    with st.chat_message("user"):
        st.write(user_p)
    with st.chat_message("assistant"):
        st.markdown(bot_r)

# Chat input
prompt = st.chat_input("Go ahead and type here...")

if prompt:
    # Render user prompt immediately
    with st.chat_message("user"):
        st.write(prompt)
        
    # Prepare state for LangGraph execution
    inputs: CurSession = {
        "prompts": [prompt],
        "replies": []
    }
    
    # Run graph execution
    with st.chat_message("assistant"):
        with st.spinner("Researching across MCP server tools and synthesizing output..."):
            result = research_app_graph.invoke(inputs)
            latest_reply = result["replies"][-1]
            st.markdown(latest_reply)
            
    # Update Streamlit session state history
    st.session_state["prompts"].append(prompt)
    st.session_state["replies"].append(latest_reply)

    # Save chat session state to past_chats folder as JSON
    session_id = st.session_state.get("session_id")
    saved_session_id = save_chat_session(
        prompts=st.session_state["prompts"],
        replies=st.session_state["replies"],
        folder_path="past_chats",
        session_id=session_id
    )
    st.session_state["session_id"] = saved_session_id