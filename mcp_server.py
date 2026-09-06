"""
mcp_server.py
-------------
FastMCP server. Every tool registered here is wrapped with:
  - A per-call wall-clock timeout (via ThreadPoolExecutor future.result(timeout=...))
  - Exponential backoff retry (with rate-limit detection on 429/quota errors)

app.py calls tools through this server and has zero retry/timeout logic of its own.

Tools registered
----------------
  arxiv_search          — arXiv Atom API
  wikipedia_search      — Wikipedia REST API
  pubmed_search         — NCBI PubMed API
  web_search            — DuckDuckGo via ddgs
  knowledge_base_search — Local FAISS index (books/ → scans/)
  memory_search         — Past conversation FAISS index (past_chats/)
"""

import os
import time
import logging
import functools
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Callable, Any

from mcp.server.fastmcp import FastMCP
from mcp_tools import search_arxiv, search_wikipedia, search_pubmed, search_web

logger = logging.getLogger(__name__)
mcp    = FastMCP("ResearchAgentMCPServer")


# ─── Tool wrapper: retry + per-call timeout ───────────────────────────────────

def _tool_wrapper(
    fn: Callable,
    *,
    max_attempts: int  = 3,
    timeout_s: float   = 15.0,
    base_delay: float  = 1.5,
) -> Callable:
    """
    Wrap a sync function with:
      1. Per-call wall-clock timeout using a single-worker executor future.
      2. Exponential-backoff retry.
      3. Special handling for 429 / quota / rate-limit errors (longer sleep).

    Returns an empty string on total failure so callers always get a string.
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        for attempt in range(max_attempts):
            try:
                with ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(fn, *args, **kwargs)
                    return fut.result(timeout=timeout_s)

            except FuturesTimeout:
                logger.warning(
                    "%s: timed out after %.0fs (attempt %d/%d)",
                    fn.__name__, timeout_s, attempt + 1, max_attempts,
                )
                if attempt == max_attempts - 1:
                    return ""

            except Exception as exc:
                err = str(exc).lower()
                is_rate_limit = "429" in err or "quota" in err or "rate" in err
                is_last       = attempt == max_attempts - 1

                if is_last:
                    logger.error(
                        "%s: failed after %d attempts — %s",
                        fn.__name__, max_attempts, exc,
                    )
                    return ""

                wait = (10 * (attempt + 1)) if is_rate_limit else (base_delay * (attempt + 1))
                logger.warning(
                    "%s: attempt %d failed (%s) — retrying in %.1fs",
                    fn.__name__, attempt + 1, exc, wait,
                )
                time.sleep(wait)

        return ""
    return wrapper


# ─── Wrapped implementations ──────────────────────────────────────────────────

_arxiv   = _tool_wrapper(search_arxiv,    timeout_s=15, max_attempts=3)
_wiki    = _tool_wrapper(search_wikipedia, timeout_s=10, max_attempts=3)
_pubmed  = _tool_wrapper(search_pubmed,   timeout_s=12, max_attempts=3)
_web     = _tool_wrapper(search_web,      timeout_s=12, max_attempts=2)


# ─── MCP Tool Registrations ───────────────────────────────────────────────────

@mcp.tool()
def arxiv_search(query: str, max_results: int = 3) -> str:
    """
    Search arXiv for academic papers.
    Returns titles, authors, publication dates, PDF URLs, and abstracts.
    """
    return _arxiv(query, max_results=max_results)


@mcp.tool()
def wikipedia_search(query: str, max_results: int = 2) -> str:
    """
    Search Wikipedia for encyclopaedic definitions and summary articles.
    """
    return _wiki(query, max_results=max_results)


@mcp.tool()
def pubmed_search(query: str, max_results: int = 3) -> str:
    """
    Search NCBI PubMed for biomedical and life sciences literature.
    """
    return _pubmed(query, max_results=max_results)


@mcp.tool()
def web_search(query: str, max_results: int = 3) -> str:
    """
    Search the general web using DuckDuckGo.
    Returns titles, URLs, and text snippets.
    """
    return _web(query, max_results=max_results)


@mcp.tool()
def knowledge_base_search(query: str, api_key: str = "", top_k: int = 4) -> str:
    """
    Search the local FAISS knowledge base built from the books/ folder.
    Requires a Google/Gemini API key for embedding the query.
    Pass api_key explicitly or set GOOGLE_API_KEY / GEMINI_API_KEY in the environment.
    """
    from knowledge_base import search_knowledge_base
    key    = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    result = _tool_wrapper(
        search_knowledge_base,
        timeout_s=20,
        max_attempts=2,
    )(query=query, google_api_key=key, top_k=top_k)
    return result or "No matches found in local knowledge base."


@mcp.tool()
def memory_search(query: str, api_key: str = "", top_k: int = 3) -> str:
    """
    Search past conversation history using the persistent Gemini+FAISS memory index.
    Falls back to keyword search if the index is not yet built.
    Pass api_key explicitly or set GOOGLE_API_KEY / GEMINI_API_KEY in the environment.
    """
    from retriever import search_memory_index
    key    = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    result = _tool_wrapper(
        search_memory_index,
        timeout_s=15,
        max_attempts=2,
    )(query=query, google_api_key=key, scans_dir="scans", top_k=top_k)
    return result or "No relevant past conversations found."


if __name__ == "__main__":
    mcp.run()
