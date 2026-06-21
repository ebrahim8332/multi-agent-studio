"""
Search fallback chain for Multi-Agent Studio.

Tries providers in order. Falls through silently on any error or rate limit.
Returns a standard result list regardless of which service responded.

Usage in any agent:
    from utils.search_client import get_search_chain
    search = get_search_chain()
    results = search.search(query, max_results=3)

Each result is a dict:
    {"title": "...", "url": "...", "content": "..."}

Provider order:
    1. Tavily  — best quality for AI agents, 1,000 free searches/month
    2. Exa     — designed for AI agents, 1,000 free searches/month
    3. Serper  — Google results, 2,500 free searches/month
    4. Empty   — graceful degradation, never crashes the pipeline
"""

import os


# ── Provider functions ────────────────────────────────────────────────────────

def _search_tavily(query: str, max_results: int) -> list[dict]:
    """Search via Tavily. Returns normalised result list."""
    from tavily import TavilyClient
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("No TAVILY_API_KEY")
    client = TavilyClient(api_key=api_key)
    response = client.search(query=query, search_depth="advanced", max_results=max_results)
    return [
        {
            "title":   r.get("title", ""),
            "url":     r.get("url", ""),
            "content": r.get("content", ""),
        }
        for r in response.get("results", [])
    ]


def _search_exa(query: str, max_results: int) -> list[dict]:
    """Search via Exa. Returns normalised result list."""
    from exa_py import Exa
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        raise ValueError("No EXA_API_KEY")
    client = Exa(api_key=api_key)
    response = client.search(query, num_results=max_results)
    return [
        {
            "title":   r.title or "",
            "url":     r.url or "",
            "content": (r.text or "")[:400],
        }
        for r in response.results
    ]


def _search_serper(query: str, max_results: int) -> list[dict]:
    """Search via Serper (Google). Returns normalised result list."""
    import json
    import http.client
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise ValueError("No SERPER_API_KEY")
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({"q": query, "num": max_results})
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    conn.request("POST", "/search", payload, headers)
    data = json.loads(conn.getresponse().read().decode("utf-8"))
    return [
        {
            "title":   r.get("title", ""),
            "url":     r.get("link", ""),
            "content": r.get("snippet", ""),
        }
        for r in data.get("organic", [])[:max_results]
    ]


# ── Fallback chain ────────────────────────────────────────────────────────────

PROVIDERS = [
    ("Tavily",  _search_tavily),
    ("Exa",     _search_exa),
    ("Serper",  _search_serper),
]


class SearchChain:
    """
    Tries each search provider in order.
    Falls through on any error (missing key, rate limit, network failure).
    Returns an empty list only when all providers fail — never raises.
    """

    def search(self, query: str, max_results: int = 3) -> list[dict]:
        """
        Runs the query against providers in order.
        Returns the first successful result list, or [] if all fail.
        """
        for name, fn in PROVIDERS:
            try:
                results = fn(query, max_results)
                if results:
                    return results
            except Exception:
                continue
        return []

    def search_multi(self, queries: list[str], max_results: int = 3) -> tuple[dict, list[str]]:
        """
        Runs search() for each query in the list.
        Returns:
            research: {query: [result dicts]}
            sources:  deduplicated list of all URLs found
        """
        research = {}
        sources = []
        for query in queries:
            hits = self.search(query, max_results)
            research[query] = hits
            for hit in hits:
                url = hit.get("url", "")
                if url and url not in sources:
                    sources.append(url)
        return research, sources


def get_search_chain() -> SearchChain:
    """Returns a ready-to-use SearchChain. Call this from any agent."""
    return SearchChain()
