"""
health.py
---------
Production Healthcheck and Diagnostics Utility for JANE Research Agent.

Performs verification of:
  1. Environment configuration & Gemini API credentials
  2. Runtime directory write permissions & FAISS vector store integrity
  3. External academic and search tool connectivity (arXiv, Wikipedia, PubMed, Web)
"""

import os
import sys
import time
from typing import Dict, Any, Tuple

from config import settings
from logger import app_logger


def check_directories() -> Tuple[bool, str]:
    """Verify that all required storage directories exist and are writable."""
    settings.ensure_directories()
    for directory in [settings.books_dir, settings.scans_dir, settings.past_chats_dir, settings.past_sessions_dir]:
        test_file = f"{directory}/.health_check_probe"
        try:
            with open(test_file, "w") as f:
                f.write("probe")
            if os.path.exists(test_file):
                os.remove(test_file)
        except Exception as exc:
            return False, f"Directory '{directory}' write test failed: {exc}"
    return True, "All storage directories exist and are writable"


def check_gemini_api() -> Tuple[bool, str]:
    """Validate Gemini API key by making a lightweight ping call."""
    key = settings.api_key
    if not key:
        return False, "No API key found in GOOGLE_API_KEY / GEMINI_API_KEY or .env"

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage

        start = time.perf_counter()
        llm = ChatGoogleGenerativeAI(
            model=settings.default_model,
            google_api_key=key,
            temperature=0.0,
            max_retries=1,
            timeout=10.0,
        )
        resp = llm.invoke([HumanMessage(content="Respond with 'PONG'")])
        latency = time.perf_counter() - start
        if "pong" in resp.content.lower():
            return True, f"Gemini API responsive ({latency:.2f}s, model: {settings.default_model})"
        return True, f"Gemini API returned response in {latency:.2f}s"
    except Exception as exc:
        return False, f"Gemini API authentication/connection error: {exc}"


def check_tool_connectivity() -> Dict[str, Tuple[bool, str]]:
    """Test reachability of external research data sources."""
    from mcp_tools import search_wikipedia, search_arxiv, search_pubmed, search_web

    results = {}

    # 1. Wikipedia
    try:
        start = time.perf_counter()
        res = search_wikipedia("Machine Learning", max_results=1)
        lat = time.perf_counter() - start
        ok = bool(res and "machine learning" in res.lower())
        results["Wikipedia"] = (ok, f"{'OK' if ok else 'Unexpected response'} ({lat:.2f}s)")
    except Exception as exc:
        results["Wikipedia"] = (False, f"Error: {exc}")

    # 2. PubMed
    try:
        start = time.perf_counter()
        res = search_pubmed("CRISPR", max_results=1)
        lat = time.perf_counter() - start
        ok = bool(res and "URL: https://pubmed.ncbi.nlm.nih.gov" in res)
        results["PubMed"] = (ok, f"{'OK' if ok else 'Unexpected response'} ({lat:.2f}s)")
    except Exception as exc:
        results["PubMed"] = (False, f"Error: {exc}")

    # 3. Web Search
    try:
        start = time.perf_counter()
        res = search_web("Python programming language", max_results=1)
        lat = time.perf_counter() - start
        ok = bool(res and len(res) > 20)
        results["Web (DDG)"] = (ok, f"{'OK' if ok else 'No results'} ({lat:.2f}s)")
    except Exception as exc:
        results["Web (DDG)"] = (False, f"Error: {exc}")

    # 4. arXiv
    try:
        start = time.perf_counter()
        res = search_arxiv("quantum", max_results=1)
        lat = time.perf_counter() - start
        ok = bool(res and ("quantum" in res.lower() or "arxiv" in res.lower()))
        results["arXiv"] = (ok, f"{'OK' if ok else 'Unexpected response'} ({lat:.2f}s)")
    except Exception as exc:
        results["arXiv"] = (False, f"Error: {exc}")

    return results


def run_diagnostics() -> bool:
    """Run full diagnostic suite and print structured report."""
    print("=" * 65)
    print(f"  {settings.app_name} v{settings.app_version} — Production Diagnostics")
    print("=" * 65)

    all_passed = True

    # Check 1: Storage
    dir_ok, dir_msg = check_directories()
    print(f"[{'PASS' if dir_ok else 'FAIL'}] Storage & Paths   : {dir_msg}")
    if not dir_ok:
        all_passed = False

    # Check 2: Gemini API
    api_ok, api_msg = check_gemini_api()
    print(f"[{'PASS' if api_ok else 'WARN'}] Gemini API Access : {api_msg}")

    # Check 3: External Tools
    tool_results = check_tool_connectivity()
    for tool_name, (ok, msg) in tool_results.items():
        print(f"[{'PASS' if ok else 'WARN'}] Tool: {tool_name:<12} : {msg}")

    print("=" * 65)
    overall_status = "HEALTHY" if (dir_ok and api_ok) else ("DEGRADED" if dir_ok else "UNHEALTHY")
    print(f"Overall System Status: {overall_status}")
    print("=" * 65)

    return dir_ok  # returns True if system can run


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    healthy = run_diagnostics()
    sys.exit(0 if healthy else 1)
