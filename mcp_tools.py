"""
mcp_tools.py
------------
Robust, high-performance tool implementations powered by httpx.
Features HTTP connection pooling, standard headers, HTTPS enforcement,
and resilient XML/JSON payload parsing.
"""

import re
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional

import httpx

from config import settings
from logger import app_logger

# Shared HTTP client with connection pooling and keep-alive
_http_client: Optional[httpx.Client] = None


def get_http_client() -> httpx.Client:
    """Return a shared singleton HTTP client with connection pooling."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.Client(
            timeout=httpx.Timeout(settings.http_timeout_s, connect=5.0),
            headers={"User-Agent": settings.user_agent},
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )
    return _http_client


def _strip_html(text: str) -> str:
    """Strip HTML tags from snippet text."""
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()


# ─── arXiv ───────────────────────────────────────────────────────────────────

def search_arxiv(query: str, max_results: int = 3) -> str:
    """
    Query the arXiv Atom API over HTTPS.
    Returns titles, authors, dates, PDF links, and abstracts.
    """
    client = get_http_client()
    encoded_query = urllib.parse.quote_plus(query.strip())
    url = (
        f"https://export.arxiv.org/api/query"
        f"?search_query=all:{encoded_query}&start=0&max_results={max_results}"
    )

    try:
        resp = client.get(url, timeout=settings.arxiv_timeout_s)
        resp.raise_for_status()
        xml_data = resp.content
    except Exception as exc:
        app_logger.warning(f"arXiv request failed: {exc}")
        return f"arXiv search failed: {exc}"

    try:
        root = ET.fromstring(xml_data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        if not entries:
            return "No arXiv papers found."

        results: List[str] = []
        for i, entry in enumerate(entries, 1):
            title_elem = entry.find("atom:title", ns)
            title = title_elem.text.strip().replace("\n", " ") if title_elem is not None and title_elem.text else "Untitled"

            summary_elem = entry.find("atom:summary", ns)
            summary = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None and summary_elem.text else "No abstract provided."

            pub_elem = entry.find("atom:published", ns)
            published = pub_elem.text[:10] if pub_elem is not None and pub_elem.text else "Unknown date"

            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns) if a.find("atom:name", ns) is not None]
            author_str = ", ".join(filter(None, authors[:3])) + (" et al." if len(authors) > 3 else "") or "Unknown"

            pdf_link = next(
                (lnk.attrib["href"] for lnk in entry.findall("atom:link", ns) if lnk.attrib.get("title") == "pdf"),
                entry.find("atom:id", ns).text if entry.find("atom:id", ns) is not None else "No link",
            )

            results.append(
                f"[{i}] {title}\n"
                f"Authors: {author_str} ({published})\n"
                f"PDF: {pdf_link}\n"
                f"Abstract: {summary[:500]}..."
            )
        return "\n\n".join(results)
    except Exception as exc:
        app_logger.warning(f"arXiv XML parsing failed: {exc}")
        return "No arXiv papers found."


# ─── Wikipedia ───────────────────────────────────────────────────────────────

def search_wikipedia(query: str, max_results: int = 2) -> str:
    """
    Query the Wikipedia search API. Returns article titles, URLs, and summaries.
    """
    client = get_http_client()
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query.strip(),
        "format": "json",
        "utf8": "1",
    }

    try:
        resp = client.get(url, params=params, timeout=settings.wikipedia_timeout_s)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        app_logger.warning(f"Wikipedia request failed: {exc}")
        return f"Wikipedia search failed: {exc}"

    items = data.get("query", {}).get("search", [])
    if not items:
        return "No Wikipedia articles found."

    results: List[str] = []
    for i, item in enumerate(items[:max_results], 1):
        title = item.get("title", "")
        raw_snippet = item.get("snippet", "")
        snippet = _strip_html(raw_snippet)
        encoded_title = urllib.parse.quote(title.replace(" ", "_"))
        page_url = f"https://en.wikipedia.org/wiki/{encoded_title}"
        results.append(f"[{i}] {title}\nURL: {page_url}\nSummary: {snippet}")

    return "\n\n".join(results)


# ─── PubMed ──────────────────────────────────────────────────────────────────

def search_pubmed(query: str, max_results: int = 3) -> str:
    """
    Query NCBI PubMed esearch + esummary APIs. Returns titles, journals, and URLs.
    """
    client = get_http_client()
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query.strip(),
        "retmode": "json",
        "retmax": str(max_results),
    }

    try:
        resp = client.get(search_url, params=search_params, timeout=settings.pubmed_timeout_s)
        resp.raise_for_status()
        search_data = resp.json()
    except Exception as exc:
        app_logger.warning(f"PubMed search request failed: {exc}")
        return f"PubMed search failed: {exc}"

    id_list = search_data.get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return "No PubMed papers found."

    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    summary_params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "json",
    }

    try:
        resp_sum = client.get(summary_url, params=summary_params, timeout=settings.pubmed_timeout_s)
        resp_sum.raise_for_status()
        summary_data = resp_sum.json()
    except Exception as exc:
        app_logger.warning(f"PubMed summary request failed: {exc}")
        return f"PubMed summary retrieval failed: {exc}"

    result_dict = summary_data.get("result", {})
    results: List[str] = []
    for i, pmid in enumerate(id_list, 1):
        info = result_dict.get(pmid, {})
        title = info.get("title", "No Title")
        pubdate = info.get("pubdate", "")
        source = info.get("source", "")
        authors = [a.get("name", "") for a in info.get("authors", [])[:3] if a.get("name")]
        author_str = ", ".join(authors) if authors else "Unknown"
        results.append(
            f"[{i}] {title}\n"
            f"Journal: {source} ({pubdate})\n"
            f"Authors: {author_str}\n"
            f"URL: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        )

    return "\n\n".join(results)


# ─── Web Search (DuckDuckGo) ─────────────────────────────────────────────────

def search_web(query: str, max_results: int = 3) -> str:
    """
    Search the web via DuckDuckGo. Returns titles, URLs, and snippets.
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS  # type: ignore

        hits = list(DDGS().text(query, max_results=max_results))
        if not hits:
            return "No web results found."

        results: List[str] = []
        for i, r in enumerate(hits, 1):
            title = r.get("title", "Untitled")
            snippet = r.get("body", r.get("snippet", ""))
            url = r.get("href", r.get("link", ""))
            results.append(f"[{i}] {title}\nURL: {url}\n{snippet}")
        return "\n\n".join(results)
    except Exception as exc:
        app_logger.warning(f"Web search failed: {exc}")
        return "No web results found."
