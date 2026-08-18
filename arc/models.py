import os
from abc import ABC, abstractmethod
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .retry import RetryManager
from .types import Message


load_dotenv()


class Model(ABC):
    """Base interface for every LLM provider."""

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        raise NotImplementedError


class NVIDIAModel(Model):
    """NVIDIA NIM model adapter."""

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
            or RetryManager(
                max_retries=3,
                initial_delay=1.0,
                max_delay=8.0,
            )
        )

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        api_messages: list[dict[str, Any]] = []

        for message in messages:
            item: dict[str, Any] = {
                "role": message.role,
                "content": message.content,
            }

            if message.name:
                item["name"] = message.name

            if message.tool_call_id:
                item["tool_call_id"] = message.tool_call_id

            api_messages.append(item)

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

        return self.retry_manager.run(request)