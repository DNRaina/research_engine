import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
from typing import List, Dict, Any

def search_arxiv(query: str, max_results: int = 3) -> str:
    """
    Queries the arXiv API for academic papers matching the search term.
    Returns titles, authors, summaries, and PDF download links.
    """
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={max_results}"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        entries = root.findall('atom:entry', ns)
        if not entries:
            return "No arXiv papers found."

        results = []
        for i, entry in enumerate(entries, 1):
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
            published = entry.find('atom:published', ns).text[:10]
            
            authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."

            pdf_link = ""
            for link in entry.findall('atom:link', ns):
                if link.attrib.get('title') == 'pdf':
                    pdf_link = link.attrib.get('href', '')
                    break
            if not pdf_link:
                pdf_link = entry.find('atom:id', ns).text

            results.append(
                f"[{i}] {title}\n"
                f"Authors: {author_str} ({published})\n"
                f"PDF URL: {pdf_link}\n"
                f"Abstract: {summary[:400]}..."
            )

        return "\n\n".join(results)
    except Exception as e:
        return f"arXiv search error: {str(e)}"


def search_wikipedia(query: str, max_results: int = 2) -> str:
    """
    Queries the Wikipedia REST API for encyclopedia definitions and summary content.
    """
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_query}&format=json&utf8=1"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))

        search_items = data.get('query', {}).get('search', [])
        if not search_items:
            return "No Wikipedia articles found."

        results = []
        for i, item in enumerate(search_items[:max_results], 1):
            title = item.get('title', '')
            snippet = item.get('snippet', '').replace('<span class="searchmatch">', '').replace('</span>', '')
            page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"

            results.append(
                f"[{i}] {title}\n"
                f"URL: {page_url}\n"
                f"Summary: {snippet}..."
            )

        return "\n\n".join(results)
    except Exception as e:
        return f"Wikipedia search error: {str(e)}"


def search_pubmed(query: str, max_results: int = 3) -> str:
    """
    Queries NCBI PubMed for biomedical and life sciences research literature.
    """
    try:
        encoded_query = urllib.parse.quote(query)
        # Search PubMed IDs
        search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={encoded_query}&retmode=json&retmax={max_results}"
        
        req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            search_data = json.loads(response.read().decode('utf-8'))

        id_list = search_data.get('esearchresult', {}).get('idlist', [])
        if not id_list:
            return "No PubMed papers found."

        # Fetch paper summaries
        ids_str = ",".join(id_list)
        summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={ids_str}&retmode=json"
        
        req_sum = urllib.request.Request(summary_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req_sum, timeout=10) as response_sum:
            summary_data = json.loads(response_sum.read().decode('utf-8'))

        results_dict = summary_data.get('result', {})
        results = []

        for i, pmid in enumerate(id_list, 1):
            paper_info = results_dict.get(pmid, {})
            title = paper_info.get('title', 'No Title')
            pubdate = paper_info.get('pubdate', '')
            source = paper_info.get('source', '')
            authors_list = [a.get('name', '') for a in paper_info.get('authors', [])[:3]]
            author_str = ", ".join(authors_list)
            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            results.append(
                f"[{i}] {title}\n"
                f"Journal: {source} ({pubdate})\n"
                f"Authors: {author_str}\n"
                f"PubMed URL: {pubmed_url}"
            )

        return "\n\n".join(results)
    except Exception as e:
        return f"PubMed search error: {str(e)}"
