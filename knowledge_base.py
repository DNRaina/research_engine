"""
knowledge_base.py
-----------------
Indexes all supported documents in books/ into a persistent FAISS vector store
in scans/faiss_index/. Supports .pdf, .txt, .md.

Uses manifest-based fingerprinting (path + size + mtime) to ensure zero-overhead
startup when documents have not changed.
"""

import os
import json
import glob
import time
from typing import List, Optional, Dict, Any
from datetime import datetime

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from logger import app_logger


# ─── Fingerprinting ───────────────────────────────────────────────────────────

def _file_fingerprint(path: str) -> Dict[str, Any]:
    s = os.stat(path)
    return {"path": path, "size": s.st_size, "mtime": s.st_mtime}


def _books_fingerprint(books_dir: str) -> List[Dict[str, Any]]:
    paths: List[str] = []
    for pat in ("*.pdf", "*.txt", "*.md"):
        paths.extend(glob.glob(os.path.join(books_dir, "**", pat), recursive=True))
    return [_file_fingerprint(p) for p in sorted(set(paths))]


def _load_manifest(scans_dir: str) -> Optional[List[Dict[str, Any]]]:
    path = os.path.join(scans_dir, settings.manifest_filename)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("files")
    except Exception as exc:
        app_logger.warning(f"Could not load manifest from {path}: {exc}")
        return None


def _save_manifest(scans_dir: str, fingerprint: List[Dict[str, Any]]) -> None:
    os.makedirs(scans_dir, exist_ok=True)
    path = os.path.join(scans_dir, settings.manifest_filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"files": fingerprint, "built_at": datetime.now().isoformat()}, f, indent=2)


def _index_exists(scans_dir: str) -> bool:
    d = os.path.join(scans_dir, settings.kb_index_subdir)
    return (
        os.path.exists(os.path.join(d, "index.faiss"))
        and os.path.exists(os.path.join(d, "index.pkl"))
    )


# ─── Document loaders ────────────────────────────────────────────────────────

def _load_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(p for p in pages if p.strip())
    except ImportError:
        app_logger.warning(f"PDF extraction requires pypdf. File skipped: {path}")
        return ""
    except Exception as exc:
        app_logger.warning(f"Could not read PDF {os.path.basename(path)}: {exc}")
        return ""


def _load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _load_all_documents(books_dir: str) -> List[Document]:
    paths: List[str] = []
    for pat in ("*.pdf", "*.txt", "*.md"):
        paths.extend(glob.glob(os.path.join(books_dir, "**", pat), recursive=True))
    paths = sorted(set(paths))

    docs: List[Document] = []
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        fname = os.path.basename(path)
        try:
            text = _load_pdf(path) if ext == ".pdf" else _load_txt(path)
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": fname, "path": path, "type": ext.lstrip(".")},
                ))
                app_logger.info(f"Loaded KB document: {fname} ({len(text):,} chars)")
        except Exception as exc:
            app_logger.warning(f"Error loading {fname}: {exc}")
    return docs


# ─── Build ────────────────────────────────────────────────────────────────────

def build_knowledge_base_if_needed(
    books_dir: str = settings.books_dir,
    scans_dir: str = settings.scans_dir,
    google_api_key: Optional[str] = None,
) -> str:
    """
    Build or refresh the FAISS index from books/.

    Returns
    -------
    "built"       Index was (re)built successfully.
    "up_to_date"  No changes detected; existing index is current.
    "no_books"    No supported files found in books_dir.
    "no_api_key"  Google API key missing; cannot generate embeddings.
    """
    api_key = google_api_key or settings.api_key
    if not api_key:
        app_logger.info("[KB] No Google/Gemini API key — knowledge base indexing skipped.")
        return "no_api_key"

    os.makedirs(books_dir, exist_ok=True)
    os.makedirs(scans_dir, exist_ok=True)

    current_fp = _books_fingerprint(books_dir)
    if not current_fp:
        app_logger.info("[KB] No supported files found in books/ — nothing to index.")
        return "no_books"

    saved_fp = _load_manifest(scans_dir)
    if saved_fp == current_fp and _index_exists(scans_dir):
        app_logger.info("[KB] Index is current — skipping rebuild.")
        return "up_to_date"

    app_logger.info(f"[KB] Building index from {len(current_fp)} file(s) in '{books_dir}'...")

    raw_docs = _load_all_documents(books_dir)
    if not raw_docs:
        app_logger.info("[KB] All files returned empty content — aborting.")
        return "no_books"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.kb_chunk_size,
        chunk_overlap=settings.kb_chunk_overlap,
    )
    chunks = splitter.split_documents(raw_docs)
    app_logger.info(f"[KB] Created {len(chunks)} chunks from {len(raw_docs)} document(s).")

    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_community.vectorstores import FAISS

        embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=api_key,
        )

        batch_size = settings.embedding_batch_size
        app_logger.info(f"[KB] Embedding {len(chunks)} chunks in batches of {batch_size}...")
        all_embeddings = []
        total_batches = -(-len(chunks) // batch_size)

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [doc.page_content for doc in batch]
            try:
                batch_vecs = embeddings.embed_documents(texts)
                all_embeddings.extend(batch_vecs)
                app_logger.info(f"[KB] Batch {i // batch_size + 1}/{total_batches} processed.")
                time.sleep(0.5)
            except Exception as exc:
                err = str(exc).lower()
                if "429" in err or "quota" in err or "rate" in err:
                    app_logger.warning(f"[KB] Rate limit on batch {i}; waiting 10s...")
                    time.sleep(10)
                    batch_vecs = embeddings.embed_documents(texts)
                    all_embeddings.extend(batch_vecs)
                else:
                    raise

        vector_store = FAISS.from_embeddings(
            text_embeddings=list(zip([d.page_content for d in chunks], all_embeddings)),
            embedding=embeddings,
            metadatas=[d.metadata for d in chunks],
        )

        index_dir = os.path.join(scans_dir, settings.kb_index_subdir)
        os.makedirs(index_dir, exist_ok=True)
        vector_store.save_local(index_dir)
        app_logger.info(f"[KB] Index saved to '{index_dir}'.")

        _save_manifest(scans_dir, current_fp)
        app_logger.info("[KB] Manifest updated — build complete.")
        return "built"

    except Exception as exc:
        app_logger.error(f"[KB] Build failed: {exc}")
        raise


# ─── Search ───────────────────────────────────────────────────────────────────

def search_knowledge_base(
    query: str,
    google_api_key: Optional[str] = None,
    scans_dir: str = settings.scans_dir,
    top_k: int = 4,
) -> str:
    """
    Search the persisted FAISS index and return formatted matching passages.
    Returns an empty string if the index is unavailable or on error.
    """
    api_key = google_api_key or settings.api_key
    if not api_key or not _index_exists(scans_dir):
        return ""

    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_community.vectorstores import FAISS

        embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=api_key,
        )
        vector_store = FAISS.load_local(
            os.path.join(scans_dir, settings.kb_index_subdir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        matched = vector_store.similarity_search(query, k=top_k)
        if not matched:
            return ""

        results = []
        for i, doc in enumerate(matched, 1):
            source = doc.metadata.get("source", "Unknown")
            results.append(f"[{i}] Source: {source}\n{doc.page_content.strip()}")
        return "\n\n".join(results)

    except Exception as exc:
        app_logger.warning(f"Knowledge base search error: {exc}")
        return ""


if __name__ == "__main__":
    import sys
    key = settings.api_key
    if not key:
        print("ERROR: Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment or .env file.")
        sys.exit(1)
    status = build_knowledge_base_if_needed(google_api_key=key)
    print(f"\nResult: {status}")
