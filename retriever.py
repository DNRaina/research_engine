import os
import json
import glob
from datetime import datetime
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Re-export knowledge base helpers so app.py has a single import point
from knowledge_base import build_knowledge_base_if_needed, search_knowledge_base

__all__ = [
    "save_chat_session",
    "load_all_chat_sessions",
    "prepare_chat_documents",
    "search_past_chats",
    "build_knowledge_base_if_needed",
    "search_knowledge_base",
]

# 1. Save state continuously as JSON in past_chats directory
def save_chat_session(
    prompts: List[str],
    replies: List[str],
    folder_path: str = "past_chats",
    session_id: Optional[str] = None
) -> str:
    """
    Saves or updates the current chat session to a JSON file in the specified folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    if not session_id:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"session_{timestamp_str}"

    file_path = os.path.join(folder_path, f"{session_id}.json")

    chats = []
    for p, r in zip(prompts, replies):
        chats.append({"prompt": p, "reply": r})

    session_data = {
        "session_id": session_id,
        "updated_at": datetime.now().isoformat(),
        "chats": chats
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)

    return session_id

# 2. Load all past chat JSON files
def load_all_chat_sessions(folder_path: str = "past_chats") -> List[Dict[str, Any]]:
    """
    Loops through all JSON files in past_chats folder and returns session dicts.
    """
    if not os.path.exists(folder_path):
        return []

    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    sessions = []

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                sessions.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return sessions

# 3. Create Chunked Documents from Past Chats
def prepare_chat_documents(folder_path: str = "past_chats") -> List[Document]:
    """
    Extracts past chat prompt-reply interactions and converts them into chunked Document objects.
    """
    sessions = load_all_chat_sessions(folder_path)
    raw_documents = []

    for session in sessions:
        session_id = session.get("session_id", "unknown_session")
        chats = session.get("chats", [])
        for i, turn in enumerate(chats, 1):
            prompt_text = turn.get("prompt", "")
            reply_text = turn.get("reply", "")
            
            content = f"User Prompt: {prompt_text}\nAssistant Reply: {reply_text}"
            metadata = {
                "session_id": session_id,
                "turn_index": i,
                "updated_at": session.get("updated_at", "")
            }
            raw_documents.append(Document(page_content=content, metadata=metadata))

    if not raw_documents:
        return []

    # Chunk text for search indexing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunked_docs = text_splitter.split_documents(raw_documents)
    return chunked_docs

# 4. Gemini Embeddings & Vector Search
def search_past_chats(
    query: str,
    google_api_key: Optional[str] = None,
    folder_path: str = "past_chats",
    top_k: int = 3
) -> str:
    """
    Searches past chat history using Gemini Embeddings and FAISS vector store.
    Falls back to keyword matching if Google API Key is unavailable.
    """
    api_key = google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    docs = prepare_chat_documents(folder_path)

    if not docs:
        return "No past chat context found."

    if api_key:
        try:
            
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            from langchain_community.vectorstores import FAISS

            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=api_key
            )

            vector_store = FAISS.from_documents(docs, embeddings)
            matched_docs = vector_store.similarity_search(query, k=top_k)

            formatted_results = []
            for i, doc in enumerate(matched_docs, 1):
                session_info = doc.metadata.get("session_id", "Session")
                formatted_results.append(
                    f"[{i}] (Source: {session_info})\n{doc.page_content}"
                )
            return "\n\n".join(formatted_results)
        except Exception as e:
            print(f"Gemini embedding search error: {e}")
            # Fall through to keyword search fallback

    # Fallback keyword match if API key missing or embedding fails
    query_words = set(query.lower().split())
    scored_docs = []
    for doc in docs:
        content_lower = doc.page_content.lower()
        score = sum(1 for word in query_words if word in content_lower)
        if score > 0:
            scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    top_matches = [doc for _, doc in scored_docs[:top_k]]

    if not top_matches:
        return "No relevant past chat context matched."

    formatted_results = []
    for i, doc in enumerate(top_matches, 1):
        session_info = doc.metadata.get("session_id", "Session")
        formatted_results.append(
            f"[{i}] (Source: {session_info})\n{doc.page_content}"
        )
    return "\n\n".join(formatted_results)
