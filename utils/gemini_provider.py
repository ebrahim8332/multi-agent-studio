import os
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

    def complete(self, messages: list[dict], timeout: int = 60, temperature: float = 0.3, max_tokens: int | None = None) -> str:
        try:
            return self._call(messages, timeout, temperature, max_tokens)

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

        except Exception as e:
            raise FallbackTrigger(
                f"Gemini unexpected error ({type(e).__name__}) on {self.model_name}: {e}"
            ) from e

    def _call(self, messages: list[dict], timeout: int, temperature: float = 0.3, max_tokens: int | None = None) -> str:
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
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=gemini_contents,
            config=config,
        )
        return response.text
