from __future__ import annotations

from typing import Any

from .base import BaseProvider, ProviderError


class ProviderRegistry:
    """Registry for ARC LLM providers."""

    def __init__(self):
        self._providers: dict[str, BaseProvider] = {}
        self._default: str | None = None

    def register(
        self,
        provider: BaseProvider,
        name: str | None = None,
    ) -> BaseProvider:

        if not isinstance(provider, BaseProvider):
            raise TypeError(
                "provider must be a BaseProvider."
            )

        provider_name = (
            name or provider.name
        ).strip().lower()

        if not provider_name:
            raise ValueError(
                "Provider name cannot be empty."
            )

        if provider_name in self._providers:
            raise ProviderError(
                f"Provider already registered: "
                f"{provider_name}"
            )

        self._providers[provider_name] = provider

        if self._default is None:
            self._default = provider_name

        return provider

    def register_many(
        self,
        *providers: BaseProvider,
    ) -> None:

        for provider in providers:
            self.register(provider)

    def get(
        self,
        name: str | None = None,
    ) -> BaseProvider:

        target = (
            name.lower()
            if name
            else self._default
        )

        if target is None:
            raise ProviderError(
                "No default provider configured."
            )

        if target not in self._providers:
            raise ProviderError(
                f"Unknown provider: {target}"
            )

        return self._providers[target]

    def set_default(self, name: str) -> None:

        name = name.strip().lower()

        if name not in self._providers:
            raise ProviderError(
                f"Unknown provider: {name}"
            )

        self._default = name

    @property
    def default(self) -> str | None:
        return self._default

    def list(self) -> list[str]:
        return list(self._providers.keys())

    def has(self, name: str) -> bool:
        return name.lower() in self._providers

    def unregister(self, name: str) -> BaseProvider:

        name = name.lower()

        if name not in self._providers:
            raise ProviderError(
                f"Unknown provider: {name}"
            )

        provider = self._providers.pop(name)

        if self._default == name:
            self._default = (
                next(
                    iter(self._providers),
                    None,
                )
            )

        return provider

    def clear(self) -> None:
        self._providers.clear()
        self._default = None

    def count(self) -> int:
        return len(self._providers)

    def generate(
        self,
        prompt: str,
        provider: str | None = None,
        **kwargs: Any,
    ):
        return self.get(provider).generate(
            prompt,
            **kwargs,
        )

    async def generate_async(
        self,
        prompt: str,
        provider: str | None = None,
        **kwargs: Any,
    ):
        return await self.get(
            provider
        ).generate_async(
            prompt,
            **kwargs,
        )
