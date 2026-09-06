"""
mcp_tools.py
------------
Raw tool implementations — pure functions, no retry or orchestration logic.
All resilience (retry, timeout, rate-limiting) lives in mcp_server.py.
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
from typing import List


# ─── arXiv ───────────────────────────────────────────────────────────────────

def search_arxiv(query: str, max_results: int = 3) -> str:
    """Query the arXiv Atom API. Returns titles, authors, dates, PDF links, abstracts."""
    encoded = urllib.parse.quote(query)
    url = (
        f"http://export.arxiv.org/api/query"
        f"?search_query=all:{encoded}&start=0&max_results={max_results}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "ResearchAgent/1.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        xml_data = r.read()

    root = ET.fromstring(xml_data)
    ns   = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)
    if not entries:
        return "No arXiv papers found."

    results = []
    for i, entry in enumerate(entries, 1):
        title     = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        summary   = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        published = entry.find("atom:published", ns).text[:10]
        authors   = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
        author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        pdf_link  = next(
            (lnk.attrib["href"] for lnk in entry.findall("atom:link", ns)
             if lnk.attrib.get("title") == "pdf"),
            entry.find("atom:id", ns).text,
        )
        results.append(
            f"[{i}] {title}\n"
            f"Authors: {author_str} ({published})\n"
            f"PDF: {pdf_link}\n"
            f"Abstract: {summary[:500]}..."
        )
    return "\n\n".join(results)


# ─── Wikipedia ───────────────────────────────────────────────────────────────

def search_wikipedia(query: str, max_results: int = 2) -> str:
    """Query the Wikipedia search API. Returns article titles, URLs, and snippets."""
    encoded = urllib.parse.quote(query)
    url = (
        f"https://en.wikipedia.org/w/api.php"
        f"?action=query&list=search&srsearch={encoded}&format=json&utf8=1"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "ResearchAgent/1.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read().decode("utf-8"))

    items = data.get("query", {}).get("search", [])
    if not items:
        return "No Wikipedia articles found."

    results = []
    for i, item in enumerate(items[:max_results], 1):
        title   = item.get("title", "")
        snippet = (
            item.get("snippet", "")
            .replace('<span class="searchmatch">', "")
            .replace("</span>", "")
        )
        page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
        results.append(f"[{i}] {title}\nURL: {page_url}\nSummary: {snippet}")
    return "\n\n".join(results)


# ─── PubMed ──────────────────────────────────────────────────────────────────

def search_pubmed(query: str, max_results: int = 3) -> str:
    """Query NCBI PubMed esearch + esummary APIs. Returns titles, journals, and PubMed URLs."""
    encoded = urllib.parse.quote(query)
    search_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pubmed&term={encoded}&retmode=json&retmax={max_results}"
    )
    req = urllib.request.Request(search_url, headers={"User-Agent": "ResearchAgent/1.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        search_data = json.loads(r.read().decode("utf-8"))

    id_list = search_data.get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return "No PubMed papers found."

    ids_str     = ",".join(id_list)
    summary_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        f"?db=pubmed&id={ids_str}&retmode=json"
    )
    req_s = urllib.request.Request(summary_url, headers={"User-Agent": "ResearchAgent/1.0"})
    with urllib.request.urlopen(req_s, timeout=10) as r:
        summary_data = json.loads(r.read().decode("utf-8"))

    result_dict = summary_data.get("result", {})
    results     = []
    for i, pmid in enumerate(id_list, 1):
        info       = result_dict.get(pmid, {})
        title      = info.get("title", "No Title")
        pubdate    = info.get("pubdate", "")
        source     = info.get("source", "")
        authors    = [a.get("name", "") for a in info.get("authors", [])[:3]]
        author_str = ", ".join(authors)
        results.append(
            f"[{i}] {title}\n"
            f"Journal: {source} ({pubdate})\n"
            f"Authors: {author_str}\n"
            f"URL: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        )
    return "\n\n".join(results)


# ─── Web Search (DuckDuckGo) ─────────────────────────────────────────────────

def search_web(query: str, max_results: int = 3) -> str:
    """Search the general web via DuckDuckGo. Returns titles, URLs, and snippets."""
    from ddgs import DDGS
    hits = list(DDGS().text(query, max_results=max_results))
    if not hits:
        return "No web results found."
    results = []
    for i, r in enumerate(hits, 1):
        title   = r.get("title", "Untitled")
        snippet = r.get("body", r.get("snippet", ""))
        url     = r.get("href", r.get("link", ""))
        results.append(f"[{i}] {title}\nURL: {url}\n{snippet}")
    return "\n\n".join(results)
