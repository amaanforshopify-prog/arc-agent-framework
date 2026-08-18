from __future__ import annotations

import asyncio
from typing import Any

from .base import BaseProvider, ProviderResponse


class MockProvider(BaseProvider):
    """Deterministic provider used for tests and development."""

    name = "mock"

    def __init__(
        self,
        model: str = "mock-model",
        response: str = "Mock response",
        **config: Any,
    ):
        super().__init__(model, **config)
        self.response = response
        self.calls = 0

    def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> ProviderResponse:

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string.")

        if not prompt.strip():
            raise ValueError("prompt cannot be empty.")

        self.calls += 1

        return ProviderResponse(
            content=self.response,
            provider=self.name,
            model=self.model,
            usage={
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(
                    self.response.split()
                ),
            },
            metadata={
                "mock": True,
                "call_number": self.calls,
            },
        )

    async def generate_async(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> ProviderResponse:

        await asyncio.sleep(0)

        return self.generate(
            prompt,
            **kwargs,
        )
