"""Errors the LLM client raises."""


class LLMError(Exception):
    """Any non-retryable API failure (bad request, auth, unexpected response)."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class RateLimitError(LLMError):
    """HTTP 429: slow down and retry."""


class TransientError(LLMError):
    """HTTP 5xx or a network blip: safe to retry."""
