import os
import concurrent.futures
from google import genai
from google.genai import types, errors as gemini_errors

from utils.base import BaseProvider, FallbackTrigger

DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiProvider(BaseProvider):
    """
    Calls the Google Gemini API.

    Gemini uses a different message format than Groq/OpenAI. This provider
    converts the standard OpenAI-style messages (role + content dicts) into
    Gemini's format automatically, so every other file in the project can use
    the same message structure regardless of which provider runs.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def complete(self, messages: list[dict], timeout: int = 60, temperature: float = 0.3,
                 max_tokens: int | None = None, schema: dict | None = None) -> tuple[str, int, int]:
        try:
            return self._call(messages, timeout, temperature, max_tokens, schema)

        except gemini_errors.ClientError as e:
            if e.code in (401, 403):
                raise  # auth/permission errors — surface, don't fall back
            raise FallbackTrigger(
                f"Gemini client error {e.code} on {self.model_name}: {e}"
            ) from e

        except gemini_errors.ServerError as e:
            raise FallbackTrigger(
                f"Gemini server error {e.code} on {self.model_name}"
            ) from e

        except FallbackTrigger:
            raise  # already a clear, specific message (e.g. truncation) — don't rewrap it

        except Exception as e:
            # No raw str(e) here, matching the ServerError branch above --
            # this is the catch-all for whatever the SDK/httpx layer throws,
            # and unlike ClientError/ServerError it isn't a structured object
            # guaranteed to be free of request-level detail.
            raise FallbackTrigger(
                f"Gemini unexpected error ({type(e).__name__}) on {self.model_name}"
            ) from e

    def _call(self, messages: list[dict], timeout: int, temperature: float = 0.3,
              max_tokens: int | None = None, schema: dict | None = None) -> tuple[str, int, int]:
        system_text = ""
        gemini_contents = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_text = content
            else:
                gemini_role = "model" if role == "assistant" else "user"
                gemini_contents.append(
                    types.Content(
                        role=gemini_role,
                        parts=[types.Part(text=content)],
                    )
                )

        if max_tokens is None:
            max_tokens = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "8192"))
        config = types.GenerateContentConfig(
            system_instruction=system_text if system_text else None,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if schema else None,
            response_schema=schema if schema else None,
        )

        # Gemini SDK does not accept a timeout parameter — wrap in a thread
        # so we can enforce the caller's timeout. Without this, gemini-2.5-pro
        # on the free tier silently hangs with no response and no error.
        def _generate():
            return self.client.models.generate_content(
                model=self.model_name,
                contents=gemini_contents,
                config=config,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_generate)
            try:
                response = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                raise FallbackTrigger(
                    f"Gemini timeout ({timeout}s) on {self.model_name}"
                )

        # finish_reason != STOP means Gemini stopped generating before it was
        # actually done — hit the token cap, tripped a safety filter, or was
        # cut short by a quota edge case. Any of these can produce a
        # response.text that looks like valid JSON but is truncated mid-string,
        # which json.loads() then fails on with no clue why. Groq already
        # raises FallbackTrigger on its own truncation signal (finish_reason
        # == "length") — this applies the same check to Gemini, which had no
        # equivalent guard. Found via a real gemini-3-flash-preview response
        # that silently truncated at ~800 characters with schema-enforced
        # output, with no exception raised anywhere.
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish_reason = getattr(candidates[0], "finish_reason", None)
            if finish_reason is not None and str(finish_reason).upper() not in ("STOP", "FINISHREASON.STOP"):
                raise FallbackTrigger(
                    f"Gemini output truncated (finish_reason={finish_reason}) on {self.model_name}"
                )

        # Extract token counts from usage metadata
        usage = getattr(response, "usage_metadata", None)
        input_tokens  = getattr(usage, "prompt_token_count",     0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        return response.text, input_tokens, output_tokens
