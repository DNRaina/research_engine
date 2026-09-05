import operator
import os
from typing import List, Dict, Any, Optional
from typing_extensions import Annotated, TypedDict
import streamlit as st
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from retriever import save_chat_session, search_past_chats

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

# 2. Graph Node Function: takes user message, conducts web search research & attempts LLM response
def research_agent_node(state: CurSession) -> Dict[str, Any]:
    """
    LangGraph node function that takes the current state (user messages in prompts),
    conducts web search and searches past chat history using Gemini Embeddings.
    """
    if not state["prompts"]:
        return {"replies": ["No user prompt found to process."]}

    latest_prompt = state["prompts"][-1]
    
    # Retrieve configuration from session state or env
    api_key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    google_api_key = st.session_state.get("google_api_key") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    model_name = st.session_state.get("model_name", "gpt-4o-mini")
    enable_search = st.session_state.get("enable_search", True)
    enable_past_chats = st.session_state.get("enable_past_chats", True)

    search_context = ""
    if enable_search:
        with st.spinner("Searching the web for research context..."):
            search_context = perform_web_search(latest_prompt, max_results=3)

    past_chats_context = ""
    if enable_past_chats:
        with st.spinner("Searching past chat history using Gemini Embeddings..."):
            past_chats_context = search_past_chats(
                query=latest_prompt,
                google_api_key=google_api_key,
                folder_path="past_chats",
                top_k=3
            )

    if api_key:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                temperature=0.7
            )

            system_instruction = (
                "You are an expert AI Research Assistant. Provide detailed, well-structured, "
                "and informative research reports based on the user prompt, web context, and relevant past chat history.\n"
                "Structure your response with clear headings, bullet points, and cite source URLs if provided."
            )
            
            messages = [SystemMessage(content=system_instruction)]

            # Add past chat context if relevant
            if past_chats_context and "No relevant past" not in past_chats_context and "No past chat" not in past_chats_context:
                messages.append(HumanMessage(content=f"Relevant Past Chat History (Cross-session Memory):\n{past_chats_context}"))

            # Add web search context if available
            if search_context and "unavailable" not in search_context and "No web results" not in search_context:
                messages.append(HumanMessage(content=f"Web Search Results for research context:\n{search_context}"))

            # Add previous prompt/reply conversation history if present in state
            for p, r in zip(state["prompts"][:-1], state["replies"]):
                messages.append(HumanMessage(content=p))
                messages.append(AIMessage(content=r))

            messages.append(HumanMessage(content=latest_prompt))

            response = llm.invoke(messages)
            reply_text = response.content
        except Exception as e:
            reply_text = (
                f"**Error invoking LLM Model:** {str(e)}\n\n"
                f"--- \n### Web Research Results Collected:\n\n{search_context if search_context else 'No research context available.'}"
            )
    else:
        # Informative response when no OpenAI API key is configured yet
        reply_text = (
            "**OpenAI API Key is missing.** Please enter your API key in the sidebar to generate AI research synthesis.\n\n"
            f"### Raw Web Research Context Collected:\n\n{search_context if search_context else 'Web search did not return results.'}\n\n"
            f"### Relevant Past Chat History:\n\n{past_chats_context if past_chats_context else 'No past chat history found.'}"
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
st.title("AI Research Agent")
st.markdown("Powered by **LangGraph**, **LangChain**, **Gemini Embeddings**, and **Streamlit**.")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
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
    
    enable_search = st.toggle("Enable Web Search Context", value=True)
    st.session_state["enable_search"] = enable_search

    enable_past_chats = st.toggle("Enable Past Chat Context (Gemini Memory)", value=True)
    st.session_state["enable_past_chats"] = enable_past_chats
    
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
        with st.spinner("Researching and synthesizing output..."):
            result = research_app_graph.invoke(inputs)
            latest_reply = result["replies"][-1]
            st.markdown(latest_reply)
            
    # Update Streamlit session state history
    st.session_state["prompts"].append(prompt)
    st.session_state["replies"].append(latest_reply)

    # Continuously save chat session state to past_chats folder as JSON
    session_id = st.session_state.get("session_id")
    saved_session_id = save_chat_session(
        prompts=st.session_state["prompts"],
        replies=st.session_state["replies"],
        folder_path="past_chats",
        session_id=session_id
    )
    st.session_state["session_id"] = saved_session_id