from mcp.server.fastmcp import FastMCP
from mcp_tools import search_arxiv, search_wikipedia, search_pubmed

# Create FastMCP mini server
mcp = FastMCP("ResearchAgentMCPServer")

@mcp.tool()
def arxiv_search(query: str, max_results: int = 3) -> str:
    """
    Search arXiv for academic papers matching the query.
    Returns paper titles, authors, publication dates, PDF download URLs, and abstracts.
    """
    return search_arxiv(query, max_results=max_results)

@mcp.tool()
def wikipedia_search(query: str, max_results: int = 2) -> str:
    """
    Search Wikipedia for encyclopedia definitions, summary articles, and facts.
    """
    return search_wikipedia(query, max_results=max_results)

@mcp.tool()
def pubmed_search(query: str, max_results: int = 3) -> str:
    """
    Search NCBI PubMed for biomedical, healthcare, and life sciences research literature.
    """
    return search_pubmed(query, max_results=max_results)

if __name__ == "__main__":
    mcp.run()
