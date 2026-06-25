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

Parallel search:
    search_parallel() runs Tavily and Exa simultaneously using ThreadPoolExecutor.
    Results are merged and deduplicated by URL. Serper is used as fallback only
    if both Tavily and Exa return zero results for a query.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed


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

    def __init__(self):
        self._active_provider = ""

    def active_provider_name(self) -> str:
        """Returns the name of the provider that succeeded on the last search."""
        return self._active_provider or "Unknown"

    def search(self, query: str, max_results: int = 3) -> list[dict]:
        """
        Runs the query against providers in order.
        Returns the first successful result list, or [] if all fail.
        """
        for name, fn in PROVIDERS:
            try:
                results = fn(query, max_results)
                if results:
                    self._active_provider = name
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

    def search_parallel(self, queries: list[str], max_results: int = 3) -> tuple[dict, list[str], dict]:
        """
        Runs Tavily and Exa simultaneously for each query using ThreadPoolExecutor.
        Merges and deduplicates results by URL. If both return zero results for a query,
        falls back to Serper.

        Returns:
            research:       {query: [merged result dicts]}
            sources:        deduplicated list of all URLs across all queries
            provider_stats: {query: {"tavily": count, "exa": count, "serper": count}}
                            Counts show how many results each provider contributed per query.
        """
        research = {}
        sources = []
        provider_stats = {}

        for query in queries:
            # Run Tavily and Exa in parallel
            tavily_results = []
            exa_results = []

            with ThreadPoolExecutor(max_workers=2) as executor:
                future_tavily = executor.submit(_search_tavily, query, max_results)
                future_exa    = executor.submit(_search_exa,    query, max_results)

                try:
                    tavily_results = future_tavily.result(timeout=30)
                except Exception:
                    tavily_results = []

                try:
                    exa_results = future_exa.result(timeout=30)
                except Exception:
                    exa_results = []

            # Merge and deduplicate by URL
            seen_urls = set()
            merged = []
            for hit in tavily_results + exa_results:
                url = hit.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    merged.append(hit)

            # Serper fallback only when both primary providers returned nothing
            serper_count = 0
            if not merged:
                try:
                    serper_hits = _search_serper(query, max_results)
                    for hit in serper_hits:
                        url = hit.get("url", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            merged.append(hit)
                    serper_count = len(serper_hits)
                except Exception:
                    pass

            research[query] = merged
            provider_stats[query] = {
                "tavily": len(tavily_results),
                "exa":    len(exa_results),
                "serper": serper_count,
            }

            for hit in merged:
                url = hit.get("url", "")
                if url and url not in sources:
                    sources.append(url)

        return research, sources, provider_stats


def get_search_chain() -> SearchChain:
    """Returns a ready-to-use SearchChain. Call this from any agent."""
    return SearchChain()
