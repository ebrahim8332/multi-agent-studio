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
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

# Domains excluded from article enrichment — low-quality or non-article content
_ENRICH_SKIP_DOMAINS = {
    "youtube.com", "youtu.be", "linkedin.com", "facebook.com",
    "twitter.com", "x.com", "instagram.com", "tiktok.com",
    "reddit.com", "quora.com", "pinterest.com", "medium.com",
}


# ── URL verification ─────────────────────────────────────────────────────────

def _verify_url(url: str, timeout: int = 5) -> bool:
    """
    Returns True if the URL resolves to a readable page (HTTP 200).
    Returns False for 404, 403, 410, 5xx errors, timeouts, and connection failures.
    Follows redirects — checks the final destination status code.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
        resp = requests.head(url, allow_redirects=True, timeout=timeout, headers=headers)
        return resp.status_code == 200
    except Exception:
        return False


def _verify_urls_parallel(results: list[dict]) -> list[dict]:
    """
    Filters a list of search result dicts, keeping only those whose URLs resolve.
    Runs HEAD requests in parallel — wall-clock time equals the slowest single check.
    """
    if not results:
        return results
    with ThreadPoolExecutor(max_workers=min(len(results), 6)) as executor:
        futures = {executor.submit(_verify_url, r.get("url", "")): r for r in results}
        verified = []
        for future in as_completed(futures):
            result = futures[future]
            try:
                if future.result():
                    verified.append(result)
            except Exception:
                pass
    # Preserve original order
    url_set = {r.get("url") for r in verified}
    return [r for r in results if r.get("url") in url_set]


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
            "title":          r.get("title", ""),
            "url":            r.get("url", ""),
            "content":        r.get("content", ""),
            "published_date": r.get("published_date", "") or "",
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
            "title":          r.title or "",
            "url":            r.url or "",
            "content":        (r.text or "")[:1200],
            "published_date": getattr(r, "published_date", "") or "",
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
        # Fire Tavily and Exa in two separate batches with a 1-second stagger.
        # Firing all queries at both providers simultaneously (N×2 threads) works
        # at low query counts, but at 10 queries × 5 results the burst can trip
        # provider rate limits. Staggering Exa by 1 second spreads the load without
        # meaningfully affecting total wall-clock time (both batches overlap).
        futures = {}  # future -> (query, provider_name)
        with ThreadPoolExecutor(max_workers=max(len(queries) * 2, 4)) as executor:
            for query in queries:
                ft = executor.submit(_search_tavily, query, max_results)
                futures[ft] = (query, "tavily")
            time.sleep(1)  # 1-second stagger before firing Exa
            for query in queries:
                fe = executor.submit(_search_exa, query, max_results)
                futures[fe] = (query, "exa")

            # Collect results as they complete
            raw: dict[str, dict] = {q: {"tavily": [], "exa": []} for q in queries}
            for future in as_completed(futures):
                query, provider = futures[future]
                try:
                    raw[query][provider] = future.result(timeout=45)
                except Exception:
                    raw[query][provider] = []

        # Merge, deduplicate, Serper fallback per query.
        # Cap to top 5 from Tavily + top 5 from Exa in API rank order.
        # Both providers rank by semantic relevance — first results are best.
        TOP_N_PER_PROVIDER = 5

        research       = {}
        sources        = []
        provider_stats = {}

        for query in queries:
            tavily_results = raw[query]["tavily"]
            exa_results    = raw[query]["exa"]

            # Select top 5 from each provider, deduplicate between them
            seen_urls = set()
            selected  = []
            for hit in tavily_results[:TOP_N_PER_PROVIDER]:
                url = hit.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    selected.append(hit)
            for hit in exa_results[:TOP_N_PER_PROVIDER]:
                url = hit.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    selected.append(hit)

            # Filter out dead links, paywalled pages, 404s, and low-authority
            # domains BEFORE deciding whether Serper needs to fill in. Doing
            # this after the emptiness check (the old order) meant a question
            # whose only hits turned out to be dead links or social-media
            # pages looked "non-empty" at that check, so Serper never fired
            # even though it exists for exactly this situation.
            selected = _verify_urls_parallel(selected)
            selected = [
                hit for hit in selected
                if urlparse(hit.get("url", "")).netloc.replace("www.", "") not in _ENRICH_SKIP_DOMAINS
            ]

            # Serper fallback only when Tavily+Exa left nothing usable
            serper_count = 0
            if not selected:
                try:
                    serper_hits = _search_serper(query, max_results)
                    serper_hits = _verify_urls_parallel(serper_hits)
                    serper_hits = [
                        hit for hit in serper_hits
                        if urlparse(hit.get("url", "")).netloc.replace("www.", "") not in _ENRICH_SKIP_DOMAINS
                    ]
                    for hit in serper_hits:
                        url = hit.get("url", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            selected.append(hit)
                    serper_count = len(serper_hits)
                except Exception:
                    pass

            research[query] = selected
            provider_stats[query] = {
                "tavily": min(len(tavily_results), TOP_N_PER_PROVIDER),
                "exa":    min(len(exa_results),    TOP_N_PER_PROVIDER),
                "serper": serper_count,
            }

            for hit in selected:
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

        # Brief pause before Extract call to avoid hammering Tavily immediately
        # after the parallel search burst. No effect on quality; avoids 429 errors.
        time.sleep(2)

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
            snippet = full_text[:3000]
            for query, idx in locations:
                if idx < len(enriched_research.get(query, [])):
                    enriched_research[query][idx]["content"]  = snippet
                    enriched_research[query][idx]["enriched"] = True
                    enriched_count += 1

        return enriched_research, enriched_count


def get_search_chain() -> SearchChain:
    """Returns a ready-to-use SearchChain. Call this from any agent."""
    return SearchChain()


def _extract_year(date_str: str) -> str:
    """Extracts a 4-digit year from a date string. Returns '' if none found."""
    if not date_str:
        return ""
    m = re.search(r'\b(20\d{2}|19\d{2})\b', str(date_str))
    return m.group(1) if m else ""


def _domain_to_publisher(domain: str) -> str:
    """Converts a domain to a short publisher label, e.g. 'iea.org' → 'IEA'."""
    if not domain:
        return ""
    name = domain.split(".")[0]
    return name.upper() if len(name) <= 4 else name.title()


def build_source_registry(research: dict) -> dict:
    """
    Assigns a stable citation ID (S1, S2, ...) to every unique source across
    all research questions, in first-appearance order.

    Calling this again on the same research dict produces the same IDs every
    time — Python dicts preserve insertion order, and `research` does not
    change after the Researcher/quality-gate phase finishes. That means
    Writer A, Writer B, a re-draft, the Fact Checker, and the final document
    all agree on which number means which source, without needing to pass
    the registry itself around as extra state.

    Returns: {url: {"id": "S1", "title": ..., "url": ..., "domain": ...}}
    """
    registry = {}
    counter = 1
    for hits in research.values():
        for hit in hits:
            url = hit.get("url", "")
            if not url or url in registry:
                continue
            try:
                domain = urlparse(url).netloc.replace("www.", "")
            except Exception:
                domain = ""
            registry[url] = {
                "id":             f"S{counter}",
                "title":          hit.get("title") or "Untitled",
                "url":            url,
                "domain":         domain,
                "published_year": _extract_year(hit.get("published_date", "")),
                "publisher":      _domain_to_publisher(domain),
            }
            counter += 1
    return registry


# ── Direct single-provider access ────────────────────────────────────────────
#
# The functions above form a fallback CHAIN — Tavily, then Exa, then Serper,
# each one covering for the others. Some modules need the opposite: always
# use Tavily for one job (recency, news) and always use Exa for a different
# job (semantic, qualitative research), never substituting one for the other.
# These two functions call a single provider directly and degrade to an
# empty list on any error or missing key — they never raise, matching the
# graceful-degradation behaviour of the rest of this file.

def search_tavily_only(query: str, max_results: int = 5, days: int | None = None) -> list[dict]:
    """
    Calls Tavily directly. Supports Tavily's `days` recency filter (only
    returns results published within the last N days) — the fallback
    chain's _search_tavily() does not expose this parameter.
    """
    import os
    try:
        from tavily import TavilyClient
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return []
        client = TavilyClient(api_key=api_key)
        # Explicit timeout, shorter than the SDK's own 60s default. This call
        # runs synchronously (not inside a ThreadPoolExecutor), and
        # run_data_agent makes up to 3 of these back to back -- at the SDK
        # default that's a worst case of 180s with the whole app frozen and
        # no cancel option. 20s keeps that worst case bounded while still
        # generous for a normal response.
        kwargs = {"query": query, "search_depth": "advanced", "max_results": max_results, "timeout": 20}
        if days:
            kwargs["days"] = days
        response = client.search(**kwargs)
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
            for r in response.get("results", [])
        ]
    except Exception:
        return []


def search_exa_only(query: str, max_results: int = 5) -> list[dict]:
    """
    Calls Exa directly, with full article text via the `contents` option
    on search() (plain search() with no contents option returns snippets
    too short for qualitative synthesis). Exa is tuned for semantic,
    qualitative search — moat, brand, competitive position — where Tavily
    is tuned for recency and news.

    Uses client.search(..., contents=...), not search_and_contents() or
    use_autoprompt — both are gone from the installed exa-py version.
    search_and_contents() logs a DeprecationWarning pointing at search(),
    and use_autoprompt raises ValueError: Invalid option. Confirmed by
    testing directly against the API, not just reading the changelog —
    this silently returned zero results in every earlier run of this
    module, since the caller here swallows exceptions and degrades to [].
    """
    import os
    try:
        from exa_py import Exa
        api_key = os.getenv("EXA_API_KEY")
        if not api_key:
            return []
        client = Exa(api_key=api_key)
        response = client.search(
            query, num_results=max_results, contents={"text": {"max_characters": 2000}}
        )
        return [
            {"title": r.title or "", "url": r.url or "", "content": (r.text or "")[:2000]}
            for r in response.results
        ]
    except Exception:
        return []
