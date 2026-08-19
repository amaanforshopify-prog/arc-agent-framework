from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .retry import RetryManager, RetryPolicy
from .types import Message


load_dotenv(
    dotenv_path=os.path.join(
        os.path.dirname(
            os.path.dirname(__file__)
        ),
        ".env",
    )
)


class Model(ABC):
    """Base interface for every LLM model adapter."""

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        raise NotImplementedError

    async def generate_async(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        return await asyncio.to_thread(
            self.generate,
            messages,
            tools,
        )


class NVIDIAModel(Model):
    """NVIDIA NIM model adapter using the OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "z-ai/glm-5.2",
        retry_manager: RetryManager | None = None,
    ):
        api_key = os.getenv("NVIDIA_API_KEY")

        if not api_key:
            raise ValueError(
                "NVIDIA_API_KEY is not set. "
                "Add it to your .env file."
            )

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
            timeout=60.0,
            max_retries=0,
        )

        self.model = model

        self.retry_manager = (
            retry_manager
            if retry_manager is not None
            else RetryManager(
                    policy=RetryPolicy(
                        max_attempts=4,
                        delay=1.0,
                        backoff=2.0,
                        max_delay=8.0,
                    )
                )
        )

    def _message_to_api(
        self,
        message: Message,
    ) -> dict[str, Any]:
        item: dict[str, Any] = {
            "role": message.role,
            "content": message.content or "",
        }

        if message.name:
            item["name"] = message.name

        if message.tool_call_id:
            item["tool_call_id"] = (
                message.tool_call_id
            )

        if message.tool_calls:
            item["tool_calls"] = []

            for call in message.tool_calls:
                item["tool_calls"].append(
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": (
                                call.function.arguments
                            ),
                        },
                    }
                )

        return item

    def _build_messages(
        self,
        messages: list[Message],
    ) -> list[dict[str, Any]]:
        return [
            self._message_to_api(message)
            for message in messages
        ]

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        api_messages = self._build_messages(
            messages
        )

        def request() -> Any:
            return self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                tools=tools or None,
                temperature=1,
                top_p=1,
                max_tokens=2048,
                stream=False,
            )

        return self.retry_manager.execute(
            request
        )

    async def generate_async(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        return await asyncio.to_thread(
            self.generate,
            messages,
            tools,
        )


