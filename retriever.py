"""
retriever.py
------------
Chat session persistence, document preparation, and memory search.

Memory index is persisted to scans/memory_index/ using FAISS so the
vector store is not rebuilt from scratch on every query.
After each new chat turn is saved, the index is refreshed incrementally.
"""

import os
import json
import glob
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Re-export KB helpers so app.py has one import point
from knowledge_base import build_knowledge_base_if_needed, search_knowledge_base

logger = logging.getLogger(__name__)

MEMORY_INDEX_SUBDIR = "memory_index"

__all__ = [
    "save_chat_session",
    "delete_chat_session",
    "load_all_chat_sessions",
    "export_session_to_markdown",
    "prepare_chat_documents",
    "build_memory_index",
    "search_memory_index",
    "search_past_chats",          # kept for backwards compat / fallback
    "build_knowledge_base_if_needed",
    "search_knowledge_base",
]


# ─── 1. Persistence ───────────────────────────────────────────────────────────

def save_chat_session(
    prompts: List[str],
    replies: List[str],
    folder_path: str = "past_chats",
    session_id: Optional[str] = None,
    latest_contexts: Optional[List[str]] = None,
) -> str:
    """
    Save or update the current chat session as a JSON file.
    latest_contexts: the raw retrieved context blocks for the most recent turn.
    Returns the session_id.
    """
    os.makedirs(folder_path, exist_ok=True)

    if not session_id:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    file_path = os.path.join(folder_path, f"{session_id}.json")

    # Load existing chats so we preserve contexts from prior turns
    existing_chats: List[Dict[str, Any]] = []
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                existing_chats = json.load(f).get("chats", [])
        except Exception:
            pass

    chats: List[Dict[str, Any]] = []
    for i, (p, r) in enumerate(zip(prompts, replies)):
        turn: Dict[str, Any] = {"prompt": p, "reply": r}
        # Preserve contexts from previously saved turns
        if i < len(existing_chats) and "contexts" in existing_chats[i]:
            turn["contexts"] = existing_chats[i]["contexts"]
        # Attach fresh contexts to the latest (last) turn
        if i == len(prompts) - 1 and latest_contexts:
            turn["contexts"] = latest_contexts
        chats.append(turn)

    session_data = {
        "session_id": session_id,
        "updated_at": datetime.now().isoformat(),
        "chats": chats,
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)

    return session_id


def delete_chat_session(session_id: str, folder_path: str = "past_chats") -> bool:
    """Delete the JSON file for the given session_id. Returns True if deleted."""
    file_path = os.path.join(folder_path, f"{session_id}.json")
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False


def export_session_to_markdown(
    prompts: List[str],
    replies: List[str],
    contexts_list: Optional[List[List[str]]] = None,
    session_id: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    Format a research conversation into a publication-ready Markdown dossier.
    Includes timestamp, session metadata, questions, synthesized analysis, and evidence sources.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sid = session_id or "adhoc_session"
    lines = [
        f"# JANE Research Dossier - Session `{sid}`",
        f"**Generated:** {now_str}  ",
        f"**Model:** {model_name or 'Gemini'}  ",
        f"**Total Research Queries:** {len(prompts)}",
        "\n---",
    ]

    for idx, (p, r) in enumerate(zip(prompts, replies), 1):
        lines.append(f"\n## Research Query {idx}")
        lines.append(f"> **Query:** {p}\n")
        lines.append("### Synthesised Report\n")
        lines.append(r.strip())

        if contexts_list and idx - 1 < len(contexts_list) and contexts_list[idx - 1]:
            ctxs = contexts_list[idx - 1]
            lines.append(f"\n<details><summary><b>📚 Retrieved Evidence & Citations ({len(ctxs)} sources)</b></summary>\n")
            for c in ctxs:
                lines.append(f"{c}\n\n---\n")
            lines.append("</details>\n")

        lines.append("\n---")

    lines.append("\n*Generated automatically by JANE (Just A Nuanced Engine).*")
    return "\n".join(lines)


# ─── 2. Document preparation ──────────────────────────────────────────────────

def load_all_chat_sessions(folder_path: str = "past_chats") -> List[Dict[str, Any]]:
    """Load all JSON files from past_chats/ and return their dicts."""
    if not os.path.exists(folder_path):
        return []
    sessions = []
    for fp in glob.glob(os.path.join(folder_path, "*.json")):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                sessions.append(json.load(f))
        except Exception as exc:
            logger.warning("Could not load %s: %s", fp, exc)
    return sessions


def prepare_chat_documents(folder_path: str = "past_chats") -> List[Document]:
    """
    Convert all past chat turns into chunked LangChain Document objects
    suitable for embedding and vector search.
    """
    sessions     = load_all_chat_sessions(folder_path)
    raw_documents: List[Document] = []

    for session in sessions:
        sid   = session.get("session_id", "unknown")
        chats = session.get("chats", [])
        for i, turn in enumerate(chats, 1):
            content = (
                f"User Prompt: {turn.get('prompt', '')}\n"
                f"Assistant Reply: {turn.get('reply', '')}"
            )
            raw_documents.append(Document(
                page_content=content,
                metadata={
                    "session_id": sid,
                    "turn_index": i,
                    "updated_at": session.get("updated_at", ""),
                },
            ))

    if not raw_documents:
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    return splitter.split_documents(raw_documents)


# ─── 3. Persistent memory index ───────────────────────────────────────────────

def _memory_index_path(scans_dir: str) -> str:
    return os.path.join(scans_dir, MEMORY_INDEX_SUBDIR)


def _memory_index_exists(scans_dir: str) -> bool:
    d = _memory_index_path(scans_dir)
    return (
        os.path.exists(os.path.join(d, "index.faiss"))
        and os.path.exists(os.path.join(d, "index.pkl"))
    )


def build_memory_index(
    google_api_key: Optional[str] = None,
    folder_path: str = "past_chats",
    scans_dir: str = "scans",
) -> str:
    """
    Build or refresh the FAISS memory index from all past_chats/ JSON files.
    Persists the index to scans/memory_index/.

    Returns: "built" | "no_chats" | "no_api_key" | "error:<msg>"
    """
    api_key = google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "no_api_key"

    docs = prepare_chat_documents(folder_path)
    if not docs:
        return "no_chats"

    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_community.vectorstores import FAISS

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key,
        )
        index_dir = _memory_index_path(scans_dir)
        os.makedirs(index_dir, exist_ok=True)

        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(index_dir)
        logger.info("Memory index built: %d chunks → %s", len(docs), index_dir)
        return "built"

    except Exception as exc:
        logger.error("Memory index build failed: %s", exc)
        return f"error:{exc}"


def search_memory_index(
    query: str,
    google_api_key: Optional[str] = None,
    scans_dir: str = "scans",
    top_k: int = 3,
) -> str:
    """
    Search the persistent memory FAISS index.
    Falls back to on-the-fly keyword search if the index is not available.
    Returns formatted string of top-k matching passages, or "".
    """
    api_key = google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if api_key and _memory_index_exists(scans_dir):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            from langchain_community.vectorstores import FAISS

            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=api_key,
            )
            vs      = FAISS.load_local(
                _memory_index_path(scans_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            matched = vs.similarity_search(query, k=top_k)
            if matched:
                return "\n\n".join(
                    f"[{i}] (Session: {doc.metadata.get('session_id', '?')})\n{doc.page_content}"
                    for i, doc in enumerate(matched, 1)
                )
        except Exception as exc:
            logger.warning("Memory index search failed, falling back: %s", exc)

    # Keyword fallback (no API key or index read error)
    return search_past_chats(query, google_api_key=api_key, top_k=top_k)


# ─── 4. Legacy on-the-fly search (kept as fallback) ──────────────────────────

def search_past_chats(
    query: str,
    google_api_key: Optional[str] = None,
    folder_path: str = "past_chats",
    top_k: int = 3,
) -> str:
    """
    Searches past chat history.
    Builds a transient FAISS store when a key is available;
    falls back to keyword scoring otherwise.
    """
    api_key = google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    docs    = prepare_chat_documents(folder_path)

    if not docs:
        return ""

    if api_key:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            from langchain_community.vectorstores import FAISS

            embeddings   = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=api_key,
            )
            vector_store = FAISS.from_documents(docs, embeddings)
            matched      = vector_store.similarity_search(query, k=top_k)
            return "\n\n".join(
                f"[{i}] (Session: {doc.metadata.get('session_id', '?')})\n{doc.page_content}"
                for i, doc in enumerate(matched, 1)
            )
        except Exception as exc:
            logger.warning("Embedding search failed: %s", exc)

    # Keyword fallback
    query_words = set(query.lower().split())
    scored      = []
    for doc in docs:
        score = sum(1 for w in query_words if w in doc.page_content.lower())
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [d for _, d in scored[:top_k]]
    if not top:
        return ""
    return "\n\n".join(
        f"[{i}] (Session: {doc.metadata.get('session_id', '?')})\n{doc.page_content}"
        for i, doc in enumerate(top, 1)
    )
