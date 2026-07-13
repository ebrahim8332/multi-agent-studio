import os
import groq as groq_errors
from groq import Groq

from utils.base import BaseProvider, FallbackTrigger

# Groq models in fallback order. Free-tier limits as of July 2026.
# llama-3.3-70b-versatile: deprecated Jul 2 2026, decommissioning Aug 16 2026 — still works, remove after that date.
# meta-llama/llama-4-scout-17b-16e-instruct: confirmed passing July 13 2026.
TIER1_MODEL = "llama-3.3-70b-versatile"                    # 70B — confirmed working, decommission Aug 16 2026
TIER2_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # Llama 4 Scout, newer architecture
TIER3_MODEL = "qwen/qwen3.6-27b"                           # strong instruction following
TIER4_MODEL = "qwen/qwen3-32b"                             # competitive coding and reasoning
TIER5_MODEL = "openai/gpt-oss-120b"                        # large reasoning model
TIER6_MODEL = "llama-3.1-8b-instant"                       # fast, high RPD
TIER7_MODEL = "openai/gpt-oss-20b"                         # last Groq resort


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
            if e.status_code in (500, 503):
                raise FallbackTrigger(f"Groq server error {e.status_code} on {self.model_name}") from e
            if e.status_code == 413:
                raise FallbackTrigger(f"Groq request too large for {self.model_name}") from e
            raise  # 401 auth errors surface as-is

        except groq_errors.APITimeoutError as e:
            raise FallbackTrigger(f"Groq timeout on {self.model_name}") from e

        except groq_errors.APIConnectionError as e:
            # One retry with a brief pause before falling back
            import time
            time.sleep(1)
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
            # Qwen3 models on Groq use thinking mode by default, generating a long
            # <think>...</think> block before the JSON. That block consumes most of
            # the output token budget and truncates the JSON. /no_think disables it.
            # Other models ignore this prefix silently.
            messages = list(messages)
            sys_idx = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)
            if sys_idx is not None:
                messages[sys_idx] = {**messages[sys_idx], "content": "/no_think\n\n" + messages[sys_idx]["content"]}
                kwargs["messages"] = messages
        response = self.client.chat.completions.create(**kwargs)

        # Extract token counts from usage
        usage = getattr(response, "usage", None)
        input_tokens  = getattr(usage, "prompt_tokens",     0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        choice = response.choices[0]
        # finish_reason == "length" means Groq hit the token cap and truncated the
        # response. This is silent truncation — no exception, just broken output.
        # Raise FallbackTrigger so the chain moves to the next model.
        if getattr(choice, "finish_reason", None) == "length":
            raise FallbackTrigger(
                f"Groq output truncated (finish_reason=length) on {self.model_name}"
            )
        text = choice.message.content or ""
        # Some Groq models (e.g. qwen3.6-27b) prepend a <think>...</think> block.
        # Strip it so pipeline agents receive clean prose.
        # Case 1: complete block — take everything after </think>
        # Case 2: truncated block (no closing tag) — strip from <think> to end
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        elif "<think>" in text:
            text = text.split("<think>", 1)[0].strip()

        return text, input_tokens, output_tokens
