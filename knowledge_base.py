"""
knowledge_base.py
-----------------
Indexes all supported documents in the `books/` folder into a persistent
FAISS vector store stored in `scans/faiss_index/`.

Supported formats: .pdf, .txt, .md

A `scans/manifest.json` fingerprints every indexed file (name + size + mtime).
On subsequent runs, if the fingerprint matches the books folder, indexing is
skipped entirely — making startup near-instant after the first build.

Public API
----------
    build_knowledge_base_if_needed(books_dir, scans_dir, google_api_key) -> str
        Returns "built" | "up_to_date" | "no_books" | "no_api_key"

    search_knowledge_base(query, google_api_key, scans_dir, top_k) -> str
        Returns formatted string of top_k matching passages.
"""

import os
import json
import hashlib
import glob
from typing import List, Optional
from datetime import datetime

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─── Paths ────────────────────────────────────────────────────────────────────

BOOKS_DIR   = "books"
SCANS_DIR   = "scans"
INDEX_SUBDIR = "faiss_index"
MANIFEST_FILE = "manifest.json"

# ─── Fingerprinting ───────────────────────────────────────────────────────────

def _file_fingerprint(path: str) -> dict:
    """Return a small dict describing a file for change detection."""
    stat = os.stat(path)
    return {
        "path": path,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }

def _books_fingerprint(books_dir: str) -> List[dict]:
    """Fingerprint every supported file in books_dir, sorted for stability."""
    patterns = ["*.pdf", "*.txt", "*.md"]
    paths: List[str] = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(books_dir, "**", pat), recursive=True))
    paths = sorted(set(paths))
    return [_file_fingerprint(p) for p in paths]

def _load_manifest(scans_dir: str) -> Optional[List[dict]]:
    """Load the saved manifest from disk, or None if it doesn't exist."""
    manifest_path = os.path.join(scans_dir, MANIFEST_FILE)
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("files", None)
    except Exception:
        return None

def _save_manifest(scans_dir: str, fingerprint: List[dict]) -> None:
    """Persist the current fingerprint to disk."""
    os.makedirs(scans_dir, exist_ok=True)
    manifest_path = os.path.join(scans_dir, MANIFEST_FILE)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"files": fingerprint, "built_at": datetime.now().isoformat()}, f, indent=2)

def _index_exists(scans_dir: str) -> bool:
    """Return True if a FAISS index already exists on disk."""
    index_dir = os.path.join(scans_dir, INDEX_SUBDIR)
    return (
        os.path.exists(os.path.join(index_dir, "index.faiss"))
        and os.path.exists(os.path.join(index_dir, "index.pkl"))
    )

# ─── Document Loading ─────────────────────────────────────────────────────────

def _load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def _load_pdf(path: str) -> str:
    """Extract text from a PDF using pypdf (graceful fallback if not installed)."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except ImportError:
        return f"[PDF support requires `pypdf`. Install with: pip install pypdf]\nFile: {path}"
    except Exception as e:
        return f"[Could not read PDF {os.path.basename(path)}: {e}]"

def _load_documents(books_dir: str) -> List[Document]:
    """Load all supported files from books_dir into Document objects."""
    docs: List[Document] = []
    patterns = ["*.pdf", "*.txt", "*.md"]
    paths: List[str] = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(books_dir, "**", pat), recursive=True))
    paths = sorted(set(paths))

    if not paths:
        return []

    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        fname = os.path.basename(path)
        try:
            if ext == ".pdf":
                raw_text = _load_pdf(path)
            else:
                raw_text = _load_txt(path)

            if raw_text.strip():
                docs.append(Document(
                    page_content=raw_text,
                    metadata={"source": fname, "path": path, "type": ext.lstrip(".")}
                ))
                print(f"  [KB] Loaded: {fname} ({len(raw_text):,} chars)")
        except Exception as e:
            print(f"  [KB] Error loading {fname}: {e}")

    return docs

# ─── Core Build Logic ─────────────────────────────────────────────────────────

def build_knowledge_base_if_needed(
    books_dir: str = BOOKS_DIR,
    scans_dir: str = SCANS_DIR,
    google_api_key: Optional[str] = None,
) -> str:
    """
    Build the FAISS knowledge base from books/ if anything has changed.

    Returns
    -------
    "built"       – index was (re)built successfully
    "up_to_date"  – nothing changed, skipping
    "no_books"    – books/ is empty, nothing to index
    "no_api_key"  – Google API key missing, cannot embed
    """
    api_key = google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[KB] No Google API key — skipping knowledge base build.")
        return "no_api_key"

    os.makedirs(books_dir, exist_ok=True)
    os.makedirs(scans_dir, exist_ok=True)

    current_fp = _books_fingerprint(books_dir)
    if not current_fp:
        print("[KB] No supported files found in books/ — nothing to index.")
        return "no_books"

    saved_fp = _load_manifest(scans_dir)
    if saved_fp == current_fp and _index_exists(scans_dir):
        print("[KB] Index is up to date — skipping rebuild.")
        return "up_to_date"

    print(f"[KB] Building knowledge base from {len(current_fp)} file(s) in '{books_dir}'...")

    # Load and chunk documents
    raw_docs = _load_documents(books_dir)
    if not raw_docs:
        print("[KB] All files loaded empty — aborting.")
        return "no_books"

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(raw_docs)
    print(f"  [KB] Created {len(chunks)} chunks from {len(raw_docs)} document(s).")

    # Embed and persist
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_community.vectorstores import FAISS

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key,
        )

        print("  [KB] Generating embeddings (this may take a moment)...")
        vector_store = FAISS.from_documents(chunks, embeddings)

        index_dir = os.path.join(scans_dir, INDEX_SUBDIR)
        os.makedirs(index_dir, exist_ok=True)
        vector_store.save_local(index_dir)
        print(f"  [KB] Index saved to '{index_dir}'.")

        _save_manifest(scans_dir, current_fp)
        print("[KB] Manifest updated.")
        return "built"

    except Exception as e:
        print(f"[KB] Error building index: {e}")
        raise


# ─── Search ───────────────────────────────────────────────────────────────────

def search_knowledge_base(
    query: str,
    google_api_key: Optional[str] = None,
    scans_dir: str = SCANS_DIR,
    top_k: int = 4,
) -> str:
    """
    Search the persisted FAISS knowledge base and return formatted matches.

    Returns an empty string if the index doesn't exist or on error.
    """
    api_key = google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return ""

    index_dir = os.path.join(scans_dir, INDEX_SUBDIR)
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
            index_dir,
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

    except Exception as e:
        print(f"[KB] Search error: {e}")
        return ""


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment or .env file.")
        sys.exit(1)
    status = build_knowledge_base_if_needed(google_api_key=api_key)
    print(f"\nResult: {status}")
