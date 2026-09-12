"""
retriever.py
------------
Chat session persistence, document preparation, and high-performance
incremental memory search.

Features:
  - Atomic, durable session persistence to past_chats/
  - Incremental FAISS vector store updating via memory_manifest.json
    (prevents redundant re-embedding of past conversation history)
  - Publication-ready Markdown dossier exports
"""

import os
import json
import glob
from datetime import datetime
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from logger import app_logger

# Re-export KB helpers so callers have one clean import point
from knowledge_base import build_knowledge_base_if_needed, search_knowledge_base

__all__ = [
    "save_chat_session",
    "delete_chat_session",
    "load_all_chat_sessions",
    "export_session_to_markdown",
    "prepare_chat_documents",
    "build_memory_index",
    "search_memory_index",
    "search_past_chats",
    "build_knowledge_base_if_needed",
    "search_knowledge_base",
]


# ─── 1. Persistence ───────────────────────────────────────────────────────────

def save_chat_session(
    prompts: List[str],
    replies: List[str],
    folder_path: str = settings.past_chats_dir,
    session_id: Optional[str] = None,
    latest_contexts: Optional[List[str]] = None,
) -> str:
    """
    Save or update the current chat session as a JSON file.
    Preserves contexts from prior turns and attaches latest_contexts to the current turn.
    Returns the session_id.
    """
    os.makedirs(folder_path, exist_ok=True)

    if not session_id:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    file_path = os.path.join(folder_path, f"{session_id}.json")

    # Load existing chats to preserve contexts from prior turns
    existing_chats: List[Dict[str, Any]] = []
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                existing_chats = json.load(f).get("chats", [])
        except Exception as exc:
            app_logger.warning(f"Failed to read existing session {file_path}: {exc}")

    chats: List[Dict[str, Any]] = []
    for i, (p, r) in enumerate(zip(prompts, replies)):
        turn: Dict[str, Any] = {"prompt": p, "reply": r}
        if i < len(existing_chats) and "contexts" in existing_chats[i]:
            turn["contexts"] = existing_chats[i]["contexts"]
        if i == len(prompts) - 1 and latest_contexts:
            turn["contexts"] = latest_contexts
        chats.append(turn)

    session_data = {
        "session_id": session_id,
        "updated_at": datetime.now().isoformat(),
        "chats": chats,
    }

    temp_path = f"{file_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)
    os.replace(temp_path, file_path)

    app_logger.info(f"Saved session {session_id} ({len(chats)} turns)")
    return session_id


def delete_chat_session(session_id: str, folder_path: str = settings.past_chats_dir) -> bool:
    """Delete the JSON file for the given session_id. Returns True if deleted."""
    file_path = os.path.join(folder_path, f"{session_id}.json")
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            app_logger.info(f"Deleted chat session: {session_id}")
            return True
        except Exception as exc:
            app_logger.error(f"Failed to delete session {session_id}: {exc}")
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
        f"# JANE Research Dossier — Session `{sid}`",
        f"**Generated:** {now_str}  ",
        f"**Model:** {model_name or settings.default_model}  ",
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
            lines.append(f"\n<details><summary><b>Retrieved Evidence & Citations ({len(ctxs)} sources)</b></summary>\n")
            for c in ctxs:
                lines.append(f"{c}\n\n---\n")
            lines.append("</details>\n")

        lines.append("\n---")

    lines.append("\n*Generated automatically by JANE (Just A Nuanced Engine).*")
    return "\n".join(lines)


# ─── 2. Document Preparation ──────────────────────────────────────────────────

def load_all_chat_sessions(folder_path: str = settings.past_chats_dir) -> List[Dict[str, Any]]:
    """Load all JSON files from past_chats/ and return their dictionaries."""
    if not os.path.exists(folder_path):
        return []
    sessions = []
    for fp in sorted(glob.glob(os.path.join(folder_path, "*.json"))):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "chats" in data:
                    sessions.append(data)
        except Exception as exc:
            app_logger.warning(f"Could not load chat file {fp}: {exc}")
    return sessions


def prepare_chat_documents(
    sessions: Optional[List[Dict[str, Any]]] = None,
    folder_path: str = settings.past_chats_dir,
) -> List[Document]:
    """
    Convert chat sessions into chunked LangChain Document objects.
    If sessions list is not provided, loads all from folder_path.
    """
    chat_sessions = sessions if sessions is not None else load_all_chat_sessions(folder_path)
    raw_documents: List[Document] = []

    for session in chat_sessions:
        sid = session.get("session_id", "unknown")
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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.memory_chunk_size,
        chunk_overlap=settings.memory_chunk_overlap,
    )
    return splitter.split_documents(raw_documents)


# ─── 3. Persistent & Incremental Memory Index ─────────────────────────────────

def _memory_index_path(scans_dir: str = settings.scans_dir) -> str:
    return os.path.join(scans_dir, settings.memory_index_subdir)


def _memory_index_exists(scans_dir: str = settings.scans_dir) -> bool:
    d = _memory_index_path(scans_dir)
    return (
        os.path.exists(os.path.join(d, "index.faiss"))
        and os.path.exists(os.path.join(d, "index.pkl"))
    )


def _load_memory_manifest(scans_dir: str) -> Dict[str, Any]:
    manifest_path = os.path.join(scans_dir, settings.memory_manifest_filename)
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            app_logger.warning(f"Failed to read memory manifest: {exc}")
    return {}


def _save_memory_manifest(scans_dir: str, manifest: Dict[str, Any]) -> None:
    manifest_path = os.path.join(scans_dir, settings.memory_manifest_filename)
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except Exception as exc:
        app_logger.warning(f"Failed to save memory manifest: {exc}")


def build_memory_index(
    google_api_key: Optional[str] = None,
    folder_path: str = settings.past_chats_dir,
    scans_dir: str = settings.scans_dir,
) -> str:
    """
    Build or incrementally refresh the FAISS memory index from past_chats/.
    Utilizes manifest tracking so only new/modified chat sessions are embedded.

    Returns: "built" | "up_to_date" | "no_chats" | "no_api_key" | "error:<msg>"
    """
    api_key = google_api_key or settings.api_key
    if not api_key:
        return "no_api_key"

    all_sessions = load_all_chat_sessions(folder_path)
    if not all_sessions:
        return "no_chats"

    index_dir = _memory_index_path(scans_dir)
    os.makedirs(index_dir, exist_ok=True)
    manifest = _load_memory_manifest(scans_dir)

    # Current fingerprint of sessions
    current_fingerprint = {}
    for s in all_sessions:
        sid = s.get("session_id")
        if sid:
            current_fingerprint[sid] = {
                "updated_at": s.get("updated_at", ""),
                "turn_count": len(s.get("chats", [])),
            }

    index_exists = _memory_index_exists(scans_dir)

    # If index exists and all sessions match manifest, skip entirely
    if index_exists and manifest.get("sessions") == current_fingerprint:
        app_logger.info("[Memory] Memory index is up-to-date — skipping rebuild.")
        return "up_to_date"

    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_community.vectorstores import FAISS

        embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=api_key,
        )

        old_sessions_meta = manifest.get("sessions", {})

        # If index exists, attempt incremental update for only new or modified sessions
        if index_exists and old_sessions_meta:
            changed_sessions = []
            for s in all_sessions:
                sid = s.get("session_id")
                curr_meta = current_fingerprint.get(sid)
                if sid not in old_sessions_meta or old_sessions_meta[sid] != curr_meta:
                    changed_sessions.append(s)

            if changed_sessions:
                app_logger.info(
                    f"[Memory] Incrementally indexing {len(changed_sessions)} modified session(s)..."
                )
                new_docs = prepare_chat_documents(changed_sessions)
                if new_docs:
                    vector_store = FAISS.load_local(
                        index_dir,
                        embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    vector_store.add_documents(new_docs)
                    vector_store.save_local(index_dir)
                    _save_memory_manifest(scans_dir, {"sessions": current_fingerprint})
                    app_logger.info(f"[Memory] Appended {len(new_docs)} new chunks to memory index.")
                    return "built"

        # Full rebuild if index doesn't exist or clean state needed
        app_logger.info(f"[Memory] Building complete memory index from {len(all_sessions)} session(s)...")
        all_docs = prepare_chat_documents(all_sessions)
        if not all_docs:
            return "no_chats"

        vector_store = FAISS.from_documents(all_docs, embeddings)
        vector_store.save_local(index_dir)
        _save_memory_manifest(scans_dir, {"sessions": current_fingerprint})
        app_logger.info(f"[Memory] Memory index successfully built: {len(all_docs)} chunks → {index_dir}")
        return "built"

    except Exception as exc:
        app_logger.error(f"[Memory] Memory index build failed: {exc}")
        return f"error:{exc}"


def search_memory_index(
    query: str,
    google_api_key: Optional[str] = None,
    scans_dir: str = settings.scans_dir,
    top_k: int = 3,
) -> str:
    """
    Search the persistent memory FAISS index.
    Falls back to on-the-fly keyword search if the index is not available.
    """
    api_key = google_api_key or settings.api_key

    if api_key and _memory_index_exists(scans_dir):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            from langchain_community.vectorstores import FAISS

            embeddings = GoogleGenerativeAIEmbeddings(
                model=settings.embedding_model,
                google_api_key=api_key,
            )
            vs = FAISS.load_local(
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
            app_logger.warning(f"Memory index search failed, falling back: {exc}")

    # Keyword fallback
    return search_past_chats(query, google_api_key=api_key, top_k=top_k)


# ─── 4. Keyword Fallback Search ───────────────────────────────────────────────

def search_past_chats(
    query: str,
    google_api_key: Optional[str] = None,
    folder_path: str = settings.past_chats_dir,
    top_k: int = 3,
) -> str:
    """
    Fallback keyword search over past chat history when vector index is unavailable.
    """
    docs = prepare_chat_documents(folder_path=folder_path)
    if not docs:
        return ""

    query_words = set(query.lower().split())
    scored = []
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
