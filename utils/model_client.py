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

Token tracking:
    Every successful LLM call appends to st.session_state["m01_call_log"].
    Each entry: {"model": str, "input_tokens": int, "output_tokens": int}
    chain.usage_summary returns totals and estimated cost.
"""

import os

from utils.base import BaseProvider, FallbackTrigger, AllProvidersExhausted
from utils.groq_provider import (
    GroqProvider,
    TIER1_MODEL, TIER2_MODEL, TIER3_MODEL, TIER4_MODEL, TIER5_MODEL,
)

SESSION_LOCK_KEY = "locked_provider_index"
CALL_LOG_KEY     = "m01_call_log"

# Approximate pricing in USD per 1M tokens (input, output). As of June 2026.
# These are estimates — actual billing may differ.
APPROX_PRICING = {
    "gemini-2.5-pro":                            (1.25,  10.00),
    "gemini-3-flash-preview":                    (0.075,  0.30),
    "gemini-3.1-flash-lite":                     (0.075,  0.30),
    "gemini-2.5-flash":                          (0.075,  0.30),
    "gemini-2.5-flash-lite":                     (0.075,  0.30),
    "gemini-2.0-flash":                          (0.10,   0.40),
    "gemini-2.0-flash-lite":                     (0.075,  0.30),
    "gemini-flash-latest":                       (0.075,  0.30),
    "qwen/qwen3.6-27b":                          (0.60,   3.00),  # Groq listed price — verify at groq.com/pricing
    "qwen/qwen3-32b":                            (0.29,   0.59),
    "openai/gpt-oss-120b":                       (0.90,   0.90),
    "llama-3.1-8b-instant":                      (0.05,   0.08),
    "openai/gpt-oss-20b":                        (0.30,   0.60),
}


def build_chain() -> list[BaseProvider]:
    """
    Builds the ordered provider list based on available API keys.

    When GEMINI_API_KEY is present, Gemini models go first (better output quality).
    Groq models follow as the fallback tier.

    Full order when both keys are set (13 providers):
      [0]  gemini-3-flash-preview — GA, matches 2.5 Pro quality, 80K output, fastest Gemini
      [1]  gemini-3.1-flash-lite  — outperforms 2.5 Flash on benchmarks, 381 t/s
      [2]  gemini-2.5-flash       — strong hybrid reasoning, 65K output
      [3]  gemini-2.5-flash-lite  — lighter 2.5, still better than 2.0 generation
      [4]  gemini-2.0-flash       — deprecated June 2026, 8K output cap
      [5]  gemini-2.0-flash-lite  — deprecated June 2026, 8K output cap
      [6]  qwen3.6-27b            — Groq TIER1; replaces llama-3.3-70b-versatile (deprecated Jul 2, 2026)
      [7]  qwen3-32b              — competitive coding and reasoning, 85.7% MMLU
      [8]  gpt-oss-120b           — large reasoning model, benchmarks not fully published
      [9]  llama-3.1-8b-instant   — smallest model, fast, high RPD
      [10] gpt-oss-20b            — smaller/faster sibling to 120B, last Groq resort
      [11] gemini-2.5-pro         — moved to near-last: never responds on free tier (silent hang)
      [12] gemini-flash-latest    — unresolved alias, unpredictable limits, last resort only
    """
    # Start with Groq as the base fallback tier
    providers: list[BaseProvider] = [
        GroqProvider(TIER1_MODEL),
        GroqProvider(TIER2_MODEL),
        GroqProvider(TIER3_MODEL),
        GroqProvider(TIER4_MODEL),
        GroqProvider(TIER5_MODEL),
    ]

    if os.getenv("GEMINI_API_KEY"):
        from utils.gemini_provider import GeminiProvider
        # Named Gemini models go at the front — best quality, known output limits.
        # gemini-2.5-pro is excluded here: it never responds on the free tier (silent hang,
        # not a 429), so putting it first costs a full timeout on every run. It goes near
        # the end instead, after all working models, so it is still used if available.
        # gemini-flash-latest is an unresolved alias with unpredictable limits — absolute last.
        for model in reversed([
            "gemini-3-flash-preview",
            "gemini-3.1-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
        ]):
            providers.insert(0, GeminiProvider(model))
        providers.append(GeminiProvider("gemini-2.5-pro"))
        providers.append(GeminiProvider("gemini-flash-latest"))

    return providers


class FallbackChain:
    """
    Tries each provider in order. Locks to the first one that succeeds.

    session_state is a dict-like object (st.session_state in Streamlit,
    a plain dict in tests). This keeps the chain independent of Streamlit.

    Token usage is accumulated into session_state[CALL_LOG_KEY] on every
    successful call. Use chain.usage_summary to read totals.
    """

    def __init__(self, providers: list[BaseProvider], session_state: dict,
                 call_log_key: str = CALL_LOG_KEY):
        self.providers = providers
        self.session_state = session_state
        self.call_log_key = call_log_key

    def complete(self, messages: list[dict], timeout: int = 90,
                 max_tokens: int | None = None,
                 agent_label: str = "",
                 schema: dict | None = None) -> tuple[str, str]:
        """Returns (response_text, model_name) from the first provider that succeeds.
        Also accumulates token counts into session_state for the run summary.
        When schema is provided, Gemini enforces it at the API level; Groq uses JSON mode."""
        # Lock key is module-specific so m01 and m02 don't share provider state.
        # If both modules run in the same browser session, each locks to its own index.
        module_lock_key = f"{self.call_log_key.replace('_call_log', '')}_locked_provider"
        start = self.session_state.get(module_lock_key, 0)
        errors = []
        for i in range(start, len(self.providers)):
            provider = self.providers[i]
            try:
                text, input_tok, output_tok = provider.complete(
                    messages, timeout=timeout, max_tokens=max_tokens, schema=schema
                )
                self.session_state[module_lock_key] = i
                self.session_state["locked_model_name"] = provider.model_name

                # Accumulate usage
                log = list(self.session_state.get(self.call_log_key, []))
                log.append({
                    "agent":         agent_label,
                    "model":         provider.model_name,
                    "input_tokens":  input_tok,
                    "output_tokens": output_tok,
                })
                self.session_state[self.call_log_key] = log

                return text, provider.model_name
            except FallbackTrigger as e:
                errors.append(f"{provider.model_name}: {e}")
                self.session_state["_fallback_errors"] = list(errors)
                continue

        details = " | ".join(errors[-5:])
        raise AllProvidersExhausted(
            f"All {len(errors)} models failed. Please try again in a few minutes."
            + (f" Last attempts: {details}" if details else "")
        )

    def reset(self):
        """Reset the provider lock so the next call starts from model 0.
        Call this at the start of each major agent operation so rate limits
        that cleared since the last agent are tried again."""
        module_lock_key = f"{self.call_log_key.replace('_call_log', '')}_locked_provider"
        self.session_state.pop(module_lock_key, None)

    @property
    def locked_model(self) -> str | None:
        """Returns the name of the currently locked model, or None if not yet locked."""
        module_lock_key = f"{self.call_log_key.replace('_call_log', '')}_locked_provider"
        locked_index = self.session_state.get(module_lock_key)
        if locked_index is not None and locked_index < len(self.providers):
            return self.providers[locked_index].model_name
        return None

    @property
    def usage_summary(self) -> dict:
        """
        Returns accumulated token usage and estimated cost for this run.
        Reads from session_state so it works across multiple chain instances.
        Cost is estimated from APPROX_PRICING — not a billing figure.
        """
        log = self.session_state.get(self.call_log_key, [])

        total_input  = sum(e["input_tokens"]  for e in log)
        total_output = sum(e["output_tokens"] for e in log)
        total_cost   = 0.0

        for entry in log:
            model = entry["model"]
            price = APPROX_PRICING.get(model)
            if price:
                in_price, out_price = price
                total_cost += (entry["input_tokens"]  / 1_000_000) * in_price
                total_cost += (entry["output_tokens"] / 1_000_000) * out_price

        return {
            "call_count":        len(log),
            "input_tokens":      total_input,
            "output_tokens":     total_output,
            "total_tokens":      total_input + total_output,
            "estimated_cost_usd": total_cost,
            "call_log":          log,
        }


def get_chain(session_state: dict, call_log_key: str = CALL_LOG_KEY) -> FallbackChain:
    """Call this from any module to get a ready-to-use chain.

    Providers are built fresh on each call — never cached. Caching provider
    objects in Streamlit causes class identity mismatches on hot-reload.

    call_log_key defaults to "m01_call_log" so Module 1 needs no changes.
    Pass a module-specific key (e.g. "m02_call_log") so each module's run
    summary reflects only its own token usage — without this, two modules
    used in the same browser session would silently share one accumulated
    log and each would report the other's cost.
    """
    providers = build_chain()
    return FallbackChain(providers, session_state, call_log_key=call_log_key)
