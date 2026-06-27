import os
import groq as groq_errors
from groq import Groq

from utils.base import BaseProvider, FallbackTrigger

# Groq models in fallback order. Free-tier limits as of June 2026.
# llama-4-scout-17b removed June 2026 — deprecated by Groq, decommissioned July 17 2026.
# Replaced with qwen3.6-27b per Groq's own recommendation.
TIER1_MODEL = "llama-3.3-70b-versatile"  # 12K TPM, 100K TPD — best quality
TIER2_MODEL = "qwen/qwen3.6-27b"         # Groq-recommended replacement for Llama 4 Scout
TIER3_MODEL = "qwen/qwen3-32b"           # 6K TPM,  500K TPD — strong instruction following
TIER4_MODEL = "openai/gpt-oss-120b"      # 8K TPM,  200K TPD — large model
TIER5_MODEL = "llama-3.1-8b-instant"     # 6K TPM,  500K TPD — fast, high RPD
TIER6_MODEL = "openai/gpt-oss-20b"       # smaller/faster sibling to 120B — last resort


class GroqProvider(BaseProvider):
    """
    Calls the Groq API. One instance per model.
    Returns plain text — agents write prose, not JSON.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def complete(self, messages: list[dict], timeout: int = 60, temperature: float = 0.3,
                 max_tokens: int | None = None, schema: dict | None = None) -> tuple[str, int, int]:
        try:
            return self._call(messages, timeout, temperature, max_tokens, schema)

        except groq_errors.RateLimitError as e:
            raise FallbackTrigger(f"Groq rate limit on {self.model_name}") from e

        except groq_errors.APIStatusError as e:
            if e.status_code == 503:
                raise FallbackTrigger(f"Groq model unavailable: {self.model_name}") from e
            if e.status_code == 413:
                raise FallbackTrigger(f"Groq request too large for {self.model_name}") from e
            raise  # 401 auth errors surface as-is

        except groq_errors.APITimeoutError as e:
            raise FallbackTrigger(f"Groq timeout on {self.model_name}") from e

        except groq_errors.APIConnectionError as e:
            # One retry before falling back
            try:
                return self._call(messages, timeout, temperature, max_tokens)
            except Exception as retry_error:
                raise FallbackTrigger(
                    f"Groq connection error on {self.model_name} (failed after retry)"
                ) from retry_error

    def _call(self, messages: list[dict], timeout: int, temperature: float = 0.3,
              max_tokens: int | None = None, schema: dict | None = None) -> tuple[str, int, int]:
        if max_tokens is None:
            max_tokens = int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "4000"))

        kwargs = dict(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        # Groq supports JSON mode but not full schema enforcement.
        # When a schema is requested, enable JSON mode — valid JSON is guaranteed,
        # but key names are not enforced at the API level (our parser handles that).
        if schema:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)

        # Extract token counts from usage
        usage = getattr(response, "usage", None)
        input_tokens  = getattr(usage, "prompt_tokens",     0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        text = response.choices[0].message.content or ""
        # Some Groq models (e.g. qwen3.6-27b) prepend a <think>...</think> block.
        # Strip it so pipeline agents receive clean prose.
        # Case 1: complete block — take everything after </think>
        # Case 2: truncated block (no closing tag) — strip from <think> to end
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        elif "<think>" in text:
            text = text.split("<think>", 1)[0].strip()

        return text, input_tokens, output_tokens
