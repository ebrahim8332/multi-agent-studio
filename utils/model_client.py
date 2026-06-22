"""
Model fallback chain for Multi-Agent Studio.

Tries providers in order. Locks to the first that succeeds. If the locked
model hits a rate limit later in the session, the chain continues from that
point and re-locks to the next provider that succeeds.

Usage in any module:
    from utils.model_client import get_chain
    chain = get_chain(st.session_state)
    response, model_used = chain.complete(messages)

messages format (same for all providers):
    [
        {"role": "system", "content": "You are ..."},
        {"role": "user",   "content": "Do this ..."},
    ]
"""

import os

from utils.base import BaseProvider, FallbackTrigger, AllProvidersExhausted
from utils.groq_provider import (
    GroqProvider,
    TIER1_MODEL, TIER2_MODEL, TIER3_MODEL, TIER4_MODEL, TIER5_MODEL, TIER6_MODEL,
)

SESSION_LOCK_KEY = "locked_provider_index"


def build_chain() -> list[BaseProvider]:
    """
    Builds the ordered provider list based on available API keys.

    When GEMINI_API_KEY is present, Gemini models go first (better output quality).
    Groq models follow as the fallback tier.

    Full order when both keys are set:
      [0]  gemini-2.5-pro         — best quality, may be rate-limited on free tier
      [1]  gemini-3-flash-preview — GA, matches 2.5 Pro quality, 80K output, fastest Gemini
      [2]  gemini-3.1-flash-lite  — outperforms 2.5 Flash on benchmarks, 381 t/s
      [3]  gemini-2.5-flash       — strong hybrid reasoning, 65K output
      [4]  gemini-2.5-flash-lite  — lighter 2.5, still better than 2.0 generation
      [5]  gemini-2.0-flash       — deprecated June 2026, 8K output cap
      [6]  gemini-2.0-flash-lite  — deprecated June 2026, 8K output cap
      [7]  llama-3.3-70b-versatile — best Groq all-rounder, 86% MMLU
      [8]  llama-4-scout-17b      — strong, multimodal, 84% MMLU
      [9]  qwen3-32b              — competitive coding and reasoning, 85.7% MMLU
      [10] gpt-oss-120b           — large reasoning model, benchmarks not fully published
      [11] llama-3.1-8b-instant   — smallest model, fast, high RPD
      [12] gpt-oss-20b            — smaller/faster sibling to 120B, last Groq resort
      [13] gemini-flash-latest    — unresolved alias, unpredictable limits, last resort only
    """
    # Start with Groq as the base fallback tier
    providers: list[BaseProvider] = [
        GroqProvider(TIER1_MODEL),
        GroqProvider(TIER2_MODEL),
        GroqProvider(TIER3_MODEL),
        GroqProvider(TIER4_MODEL),
        GroqProvider(TIER5_MODEL),
        GroqProvider(TIER6_MODEL),
    ]

    if os.getenv("GEMINI_API_KEY"):
        from utils.gemini_provider import GeminiProvider
        # Named Gemini models go at the front — best quality, known output limits
        # gemini-flash-latest is an alias for an unknown model with unpredictable limits
        # so it goes at the end, after Groq, as a last resort
        for model in reversed([
            "gemini-2.5-pro",
            "gemini-3-flash-preview",
            "gemini-3.1-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
        ]):
            providers.insert(0, GeminiProvider(model))
        providers.append(GeminiProvider("gemini-flash-latest"))

    return providers


class FallbackChain:
    """
    Tries each provider in order. Locks to the first one that succeeds.

    session_state is a dict-like object (st.session_state in Streamlit,
    a plain dict in tests). This keeps the chain independent of Streamlit.
    """

    def __init__(self, providers: list[BaseProvider], session_state: dict):
        self.providers = providers
        self.session_state = session_state

    def complete(self, messages: list[dict], timeout: int = 90, max_tokens: int | None = None) -> tuple[str, str]:
        """Returns (response_text, model_name) from the first provider that succeeds."""
        start_index = self.session_state.get(SESSION_LOCK_KEY) or 0

        errors = []
        for i in range(start_index, len(self.providers)):
            provider = self.providers[i]
            try:
                response = provider.complete(messages, timeout=timeout, max_tokens=max_tokens)
                self.session_state[SESSION_LOCK_KEY] = i
                return response, provider.model_name
            except FallbackTrigger as e:
                errors.append(f"{provider.model_name}: {e}")
                self.session_state["_fallback_errors"] = list(errors)
                continue

        details = " | ".join(errors[-3:])
        raise AllProvidersExhausted(
            "All models are currently unavailable. Please try again in a few minutes."
            + (f" Last attempts: {details}" if details else "")
        )

    @property
    def locked_model(self) -> str | None:
        """Returns the name of the currently locked model, or None if not yet locked."""
        locked_index = self.session_state.get(SESSION_LOCK_KEY)
        if locked_index is not None and locked_index < len(self.providers):
            return self.providers[locked_index].model_name
        return None


def get_chain(session_state: dict) -> FallbackChain:
    """Call this from any module to get a ready-to-use chain.

    Providers are built fresh on each call — never cached. Caching provider
    objects in Streamlit causes class identity mismatches on hot-reload.
    """
    providers = build_chain()
    return FallbackChain(providers, session_state)
