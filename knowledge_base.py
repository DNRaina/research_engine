"""
knowledge_base.py
-----------------
Indexes all supported documents in books/ into a persistent FAISS vector
store stored in scans/faiss_index/.

Supported formats: .pdf, .txt, .md

A scans/manifest.json fingerprints every indexed file (name + size + mtime).
On subsequent runs, if the fingerprint matches the current books/ contents,
indexing is skipped entirely — making startup near-instant after the first build.

Public API
----------
    build_knowledge_base_if_needed(books_dir, scans_dir, google_api_key)
        Returns: "built" | "up_to_date" | "no_books" | "no_api_key"

    search_knowledge_base(query, google_api_key, scans_dir, top_k)
        Returns: formatted string of top-k matching passages, or "".
"""

import os
import json
import glob
import time
import logging
from typing import List, Optional
from datetime import datetime

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ─── Directory constants ──────────────────────────────────────────────────────
BOOKS_DIR    = "books"
SCANS_DIR    = "scans"
INDEX_SUBDIR = "faiss_index"
MANIFEST_FILE = "manifest.json"

# ─── Fingerprinting ───────────────────────────────────────────────────────────

def _file_fingerprint(path: str) -> dict:
    s = os.stat(path)
    return {"path": path, "size": s.st_size, "mtime": s.st_mtime}

def _books_fingerprint(books_dir: str) -> List[dict]:
    paths: List[str] = []
    for pat in ("*.pdf", "*.txt", "*.md"):
        paths.extend(glob.glob(os.path.join(books_dir, "**", pat), recursive=True))
    return [_file_fingerprint(p) for p in sorted(set(paths))]

def _load_manifest(scans_dir: str) -> Optional[List[dict]]:
    path = os.path.join(scans_dir, MANIFEST_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("files")
    except Exception:
        return None

def _save_manifest(scans_dir: str, fingerprint: List[dict]) -> None:
    os.makedirs(scans_dir, exist_ok=True)
    path = os.path.join(scans_dir, MANIFEST_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"files": fingerprint, "built_at": datetime.now().isoformat()}, f, indent=2)

def _index_exists(scans_dir: str) -> bool:
    d = os.path.join(scans_dir, INDEX_SUBDIR)
    return (
        os.path.exists(os.path.join(d, "index.faiss"))
        and os.path.exists(os.path.join(d, "index.pkl"))
    )

# ─── Document loaders ────────────────────────────────────────────────────────

def _load_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages  = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(p for p in pages if p.strip())
    except ImportError:
        return f"[PDF support requires pypdf — install with: pip install pypdf]\nFile: {path}"
    except Exception as exc:
        logger.warning("Could not read PDF %s: %s", os.path.basename(path), exc)
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
        ext   = os.path.splitext(path)[1].lower()
        fname = os.path.basename(path)
        try:
            text = _load_pdf(path) if ext == ".pdf" else _load_txt(path)
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": fname, "path": path, "type": ext.lstrip(".")},
                ))
                logger.info("Loaded: %s (%d chars)", fname, len(text))
                print(f"  Loaded: {fname} ({len(text):,} chars)")
        except Exception as exc:
            logger.warning("Error loading %s: %s", fname, exc)
    return docs

# ─── Build ────────────────────────────────────────────────────────────────────

def build_knowledge_base_if_needed(
    books_dir: str = BOOKS_DIR,
    scans_dir: str = SCANS_DIR,
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
    api_key = google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[KB] No Google API key — knowledge base indexing disabled.")
        return "no_api_key"

    os.makedirs(books_dir, exist_ok=True)
    os.makedirs(scans_dir, exist_ok=True)

    current_fp = _books_fingerprint(books_dir)
    if not current_fp:
        print("[KB] No supported files found in books/ — nothing to index.")
        return "no_books"

    saved_fp = _load_manifest(scans_dir)
    if saved_fp == current_fp and _index_exists(scans_dir):
        print("[KB] Index is current — skipping rebuild.")
        return "up_to_date"

    print(f"[KB] Building index from {len(current_fp)} file(s) in '{books_dir}'...")

    raw_docs = _load_all_documents(books_dir)
    if not raw_docs:
        print("[KB] All files returned empty content — aborting.")
        return "no_books"

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks   = splitter.split_documents(raw_docs)
    print(f"  Created {len(chunks)} chunks from {len(raw_docs)} document(s).")

    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_community.vectorstores import FAISS

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key,
        )

        # Embed in small batches to respect rate limits
        BATCH_SIZE = 50
        print(f"  Embedding {len(chunks)} chunks in batches of {BATCH_SIZE}...")
        all_embeddings = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            texts = [doc.page_content for doc in batch]
            try:
                batch_vecs = embeddings.embed_documents(texts)
                all_embeddings.extend(batch_vecs)
                print(f"  Batch {i // BATCH_SIZE + 1}/{-(-len(chunks) // BATCH_SIZE)} done.")
                time.sleep(0.5)   # respect Gemini embedding rate limits
            except Exception as exc:
                err = str(exc).lower()
                if "429" in err or "quota" in err or "rate" in err:
                    print(f"  Rate limit on batch {i}; waiting 10s...")
                    time.sleep(10)
                    batch_vecs = embeddings.embed_documents(texts)
                    all_embeddings.extend(batch_vecs)
                else:
                    raise

        # Build FAISS index from pre-computed embeddings
        import numpy as np
        from langchain_community.vectorstores import FAISS as FAISSStore

        vector_store = FAISSStore.from_embeddings(
            text_embeddings=list(zip([d.page_content for d in chunks], all_embeddings)),
            embedding=embeddings,
            metadatas=[d.metadata for d in chunks],
        )

        index_dir = os.path.join(scans_dir, INDEX_SUBDIR)
        os.makedirs(index_dir, exist_ok=True)
        vector_store.save_local(index_dir)
        print(f"  Index saved to '{index_dir}'.")

        _save_manifest(scans_dir, current_fp)
        print("[KB] Manifest updated — build complete.")
        return "built"

    except Exception as exc:
        logger.error("Knowledge base build failed: %s", exc)
        print(f"[KB] Build failed: {exc}")
        raise

# ─── Search ───────────────────────────────────────────────────────────────────

def search_knowledge_base(
    query: str,
    google_api_key: Optional[str] = None,
    scans_dir: str = SCANS_DIR,
    top_k: int = 4,
) -> str:
    """
    Search the persisted FAISS index and return formatted matching passages.
    Returns an empty string if the index is unavailable or on error.
    """
    api_key = google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return ""
    if not _index_exists(scans_dir):
        return ""

    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_community.vectorstores import FAISS

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key,
        )
        vector_store = FAISS.load_local(
            os.path.join(scans_dir, INDEX_SUBDIR),
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
        logger.warning("Knowledge base search error: %s", exc)
        return ""

# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment or .env file.")
        sys.exit(1)
    result = build_knowledge_base_if_needed(google_api_key=api_key)
    print(f"\nResult: {result}")
