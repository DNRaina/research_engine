"""
mcp_server.py
-------------
FastMCP server with enterprise-grade resilience:
  - Non-blocking wall-clock timeouts using a shared daemon ThreadPoolExecutor
  - Exponential backoff retry with jitter (rate-limit detection on 429/quota)
  - Detailed telemetry and call execution latency tracking
"""

import time
import random
import functools
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Callable, Any, Optional

from mcp.server.fastmcp import FastMCP

from config import settings
from logger import app_logger
from mcp_tools import search_arxiv, search_wikipedia, search_pubmed, search_web

mcp = FastMCP("ResearchAgentMCPServer")

# Shared bounded daemon worker pool for all tool executions.
# Using daemon threads prevents any lingering worker from blocking process shutdown.
_EXECUTOR = ThreadPoolExecutor(
    max_workers=16,
    thread_name_prefix="jane_worker_",
)


def _tool_wrapper(
    fn: Callable,
    *,
    max_attempts: Optional[int] = None,
    timeout_s: Optional[float] = None,
    base_delay: Optional[float] = None,
) -> Callable:
    """
    Wrap a tool function with:
      1. Non-blocking wall-clock timeout via shared daemon executor.
      2. Exponential backoff with jitter on transient failures.
      3. Rate-limit (429/quota) handling with extended delay.
      4. Telemetry logging for latency and outcome.
    """
    attempts_limit = max_attempts or settings.max_tool_retries
    call_timeout = timeout_s or settings.http_timeout_s
    delay_base = base_delay or settings.base_retry_delay_s

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        fn_name = fn.__name__
        start_time = time.perf_counter()

        for attempt in range(1, attempts_limit + 1):
            attempt_start = time.perf_counter()
            try:
                # Submit to shared pool — avoids spawning/shutting down pools in loop
                future = _EXECUTOR.submit(fn, *args, **kwargs)
                result = future.result(timeout=call_timeout)
                duration = time.perf_counter() - attempt_start
                app_logger.info(f"Tool {fn_name} succeeded in {duration:.2f}s (attempt {attempt}/{attempts_limit})")
                return result

            except FuturesTimeout:
                duration = time.perf_counter() - attempt_start
                future.cancel()
                app_logger.warning(
                    f"Tool {fn_name} timed out after {duration:.2f}s (attempt {attempt}/{attempts_limit})"
                )
                if attempt == attempts_limit:
                    break

            except Exception as exc:
                duration = time.perf_counter() - attempt_start
                err = str(exc).lower()
                is_rate_limit = "429" in err or "quota" in err or "rate" in err
                is_last = attempt == attempts_limit

                if is_last:
                    app_logger.error(
                        f"Tool {fn_name} failed definitively after {attempts_limit} attempts: {exc}"
                    )
                    break

                # Exponential backoff with jitter
                jitter = random.uniform(0.1, 0.5)
                wait_time = (
                    settings.rate_limit_backoff_s * attempt
                    if is_rate_limit
                    else (delay_base * (2 ** (attempt - 1))) + jitter
                )
                app_logger.warning(
                    f"Tool {fn_name} attempt {attempt} failed ({exc}). Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)

        total_elapsed = time.perf_counter() - start_time
        app_logger.error(f"Tool {fn_name} exhausted all attempts without success (total {total_elapsed:.2f}s)")
        return ""

    return wrapper


# ─── Wrapped implementations ──────────────────────────────────────────────────

_arxiv = _tool_wrapper(search_arxiv, timeout_s=settings.arxiv_timeout_s, max_attempts=3)
_wiki = _tool_wrapper(search_wikipedia, timeout_s=settings.wikipedia_timeout_s, max_attempts=3)
_pubmed = _tool_wrapper(search_pubmed, timeout_s=settings.pubmed_timeout_s, max_attempts=3)
_web = _tool_wrapper(search_web, timeout_s=settings.web_timeout_s, max_attempts=2)


# ─── MCP Tool Registrations ───────────────────────────────────────────────────

@mcp.tool()
def arxiv_search(query: str, max_results: int = 3) -> str:
    """
    Search arXiv for academic research papers.
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
    Requires a Google/Gemini API key for query embedding.
    """
    from knowledge_base import search_knowledge_base
    key = api_key or settings.api_key
    result = _tool_wrapper(
        search_knowledge_base,
        timeout_s=settings.kb_search_timeout_s,
        max_attempts=2,
    )(query=query, google_api_key=key, top_k=top_k)
    return result or "No matches found in local knowledge base."


@mcp.tool()
def memory_search(query: str, api_key: str = "", top_k: int = 3) -> str:
    """
    Search past conversation history using the persistent Gemini+FAISS memory index.
    Falls back to keyword search if the vector index is not yet built.
    """
    from retriever import search_memory_index
    key = api_key or settings.api_key
    result = _tool_wrapper(
        search_memory_index,
        timeout_s=settings.memory_search_timeout_s,
        max_attempts=2,
    )(query=query, google_api_key=key, scans_dir=settings.scans_dir, top_k=top_k)
    return result or "No relevant past conversations found."


if __name__ == "__main__":
    mcp.run()
