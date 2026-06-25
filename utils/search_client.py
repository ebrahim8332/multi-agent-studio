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
    search_parallel() fires ALL (N queries x 2 providers) simultaneously in a
    single ThreadPoolExecutor pool. For 5 questions that is 10 parallel threads
    instead of 5 sequential pairs. Serper is used as fallback only if both
    Tavily and Exa return zero results for a query.

Article enrichment:
    enrich_top_sources() uses Tavily Extract to fetch full article text for the
    top sources per query, replacing the 400-char snippet with up to 2,000 chars.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Domains excluded from article enrichment — low-quality or non-article content
_ENRICH_SKIP_DOMAINS = {
    "youtube.com", "youtu.be", "linkedin.com", "facebook.com",
    "twitter.com", "x.com", "instagram.com", "tiktok.com",
    "reddit.com", "quora.com", "pinterest.com", "medium.com",
}


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
        Fires ALL (N queries x 2 providers) simultaneously in a single ThreadPoolExecutor pool.
        For 5 questions that is 10 parallel threads instead of 5 sequential pairs.
        Merges and deduplicates results by URL. If both providers return zero results for a
        query, falls back to Serper for that query.

        Returns:
            research:       {query: [merged result dicts]}
            sources:        deduplicated list of all URLs across all queries
            provider_stats: {query: {"tavily": count, "exa": count, "serper": count}}
                            Counts show how many results each provider contributed per query.
        """
        # Submit all (query, provider) pairs at once
        futures = {}  # future -> (query, provider_name)
        with ThreadPoolExecutor(max_workers=len(queries) * 2) as executor:
            for query in queries:
                ft = executor.submit(_search_tavily, query, max_results)
                fe = executor.submit(_search_exa,    query, max_results)
                futures[ft] = (query, "tavily")
                futures[fe] = (query, "exa")

            # Collect results as they complete
            raw: dict[str, dict] = {q: {"tavily": [], "exa": []} for q in queries}
            for future in as_completed(futures):
                query, provider = futures[future]
                try:
                    raw[query][provider] = future.result(timeout=30)
                except Exception:
                    raw[query][provider] = []

        # Merge, deduplicate, Serper fallback per query
        research       = {}
        sources        = []
        provider_stats = {}

        for query in queries:
            tavily_results = raw[query]["tavily"]
            exa_results    = raw[query]["exa"]

            seen_urls = set()
            merged    = []
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

    def enrich_top_sources(self, research: dict, max_per_query: int = 2) -> tuple[dict, int]:
        """
        Fetches full article text for the top sources per query using Tavily Extract.

        Only enriches sources that are not from low-quality/social domains.
        Replaces the 400-char snippet with up to 2,000 chars of full text.
        Adds "enriched": True to each enriched hit.

        Returns:
            enriched_research: updated copy of the research dict
            enriched_count:    total number of sources enriched
        """
        from urllib.parse import urlparse
        from tavily import TavilyClient

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return research, 0

        try:
            client = TavilyClient(api_key=api_key)
        except Exception:
            return research, 0

        # Collect candidate URLs (up to max_per_query per query, skip bad domains)
        url_to_hits: dict[str, list] = {}  # url -> list of (query, hit_index) tuples
        for query, hits in research.items():
            count = 0
            for idx, hit in enumerate(hits):
                if count >= max_per_query:
                    break
                url = hit.get("url", "")
                if not url:
                    continue
                try:
                    domain = urlparse(url).netloc.replace("www.", "")
                except Exception:
                    domain = ""
                if domain in _ENRICH_SKIP_DOMAINS:
                    continue
                if url not in url_to_hits:
                    url_to_hits[url] = []
                url_to_hits[url].append((query, idx))
                count += 1

        if not url_to_hits:
            return research, 0

        # Fetch full text from Tavily Extract
        try:
            extract_response = client.extract(urls=list(url_to_hits.keys()))
            results_by_url = {
                r.get("url", ""): r.get("raw_content", "")
                for r in extract_response.get("results", [])
                if r.get("raw_content")
            }
        except Exception:
            return research, 0

        # Apply enrichment — work on a deep copy so original is not mutated
        import copy
        enriched_research = copy.deepcopy(research)
        enriched_count    = 0

        for url, locations in url_to_hits.items():
            full_text = results_by_url.get(url, "")
            if not full_text:
                continue
            snippet = full_text[:2000]
            for query, idx in locations:
                if idx < len(enriched_research.get(query, [])):
                    enriched_research[query][idx]["content"]  = snippet
                    enriched_research[query][idx]["enriched"] = True
                    enriched_count += 1

        return enriched_research, enriched_count


def get_search_chain() -> SearchChain:
    """Returns a ready-to-use SearchChain. Call this from any agent."""
    return SearchChain()
