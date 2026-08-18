from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class ProviderError(Exception):
    """Base error for ARC LLM providers."""


class ProviderConfigurationError(ProviderError):
    """Raised when provider configuration is invalid."""


class ProviderRequestError(ProviderError):
    """Raised when a provider request fails."""


@dataclass
class ProviderResponse:
    content: str
    provider: str
    model: str
    success: bool = True
    usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: Any = None

    @property
    def text(self) -> str:
        return self.content


class BaseProvider(ABC):
    """Common interface implemented by every ARC LLM provider."""

    name: str = "base"

    def __init__(
        self,
        model: str,
        **config: Any,
    ):
        if not model or not model.strip():
            raise ProviderConfigurationError(
                "Model cannot be empty."
            )

        self.model = model
        self.config = config

    @abstractmethod
    def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a synchronous response."""

    @abstractmethod
    async def generate_async(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate an asynchronous response."""

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
        }
