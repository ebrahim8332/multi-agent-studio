from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """Abstract interface that every model provider must implement."""
    model_name: str = ""

    @abstractmethod
    def complete(self, messages: list[dict], timeout: int = 60, temperature: float = 0.3,
                 max_tokens: int | None = None) -> tuple[str, int, int]:
        """Send messages to the model and return (response_text, input_tokens, output_tokens).
        Raise FallbackTrigger for any error that should cause the chain to try the next model.
        """
        pass


class FallbackTrigger(Exception):
    """Raised when an error is retriable and the chain should try the next model."""
    pass


class AllProvidersExhausted(Exception):
    """Raised when every model in the chain has failed."""
    pass
