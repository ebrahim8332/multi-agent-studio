import os
import groq as groq_errors
from groq import Groq

from utils.base import BaseProvider, FallbackTrigger

# Groq models in fallback order. Free-tier limits as of June 2026.
TIER1_MODEL = "llama-3.3-70b-versatile"                   # 12K TPM, 100K TPD — best quality
TIER2_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # 30K TPM, 500K TPD — Llama 4
TIER3_MODEL = "qwen/qwen3-32b"                             # 6K TPM,  500K TPD — strong instruction following
TIER4_MODEL = "openai/gpt-oss-120b"                        # 8K TPM,  200K TPD — large model
TIER5_MODEL = "llama-3.1-8b-instant"                       # 6K TPM,  500K TPD — last resort


class GroqProvider(BaseProvider):
    """
    Calls the Groq API. One instance per model.
    Returns plain text — agents write prose, not JSON.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def complete(self, messages: list[dict], timeout: int = 60, temperature: float = 0.3, max_tokens: int | None = None) -> str:
        try:
            return self._call(messages, timeout, temperature, max_tokens)

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

    def _call(self, messages: list[dict], timeout: int, temperature: float = 0.3, max_tokens: int | None = None) -> str:
        if max_tokens is None:
            max_tokens = int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "4000"))
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        return response.choices[0].message.content
