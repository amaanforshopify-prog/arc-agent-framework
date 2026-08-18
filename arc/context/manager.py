from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ContextError(Exception):
    """Base error for ARC context management."""


@dataclass
class ContextMessage:
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ContextManager:
    """Manages conversation context for ARC agents."""

    VALID_ROLES = {
        "system",
        "user",
        "assistant",
        "tool",
    }

    def __init__(self, max_messages: int = 50):
        if max_messages <= 0:
            raise ValueError(
                "max_messages must be greater than 0."
            )

        self.max_messages = max_messages
        self.messages: list[ContextMessage] = []
        self.metadata: dict[str, Any] = {}

    def add(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContextMessage:

        role = role.lower().strip()

        if role not in self.VALID_ROLES:
            raise ContextError(
                f"Invalid role: {role}"
            )

        if not isinstance(content, str):
            raise TypeError(
                "content must be a string."
            )

        if not content.strip():
            raise ValueError(
                "content cannot be empty."
            )

        message = ContextMessage(
            role=role,
            content=content,
            metadata=metadata or {},
        )

        self.messages.append(message)
        self._trim()

        return message

    def system(self, content: str, **metadata: Any):
        return self.add(
            "system",
            content,
            metadata,
        )

    def user(self, content: str, **metadata: Any):
        return self.add(
            "user",
            content,
            metadata,
        )

    def assistant(self, content: str, **metadata: Any):
        return self.add(
            "assistant",
            content,
            metadata,
        )

    def tool(self, content: str, **metadata: Any):
        return self.add(
            "tool",
            content,
            metadata,
        )

    def _trim(self) -> None:
        if len(self.messages) <= self.max_messages:
            return

        system_messages = [
            message
            for message in self.messages
            if message.role == "system"
        ]

        other_messages = [
            message
            for message in self.messages
            if message.role != "system"
        ]

        available = (
            self.max_messages
            - len(system_messages)
        )

        if available < 0:
            system_messages = system_messages[
                -self.max_messages:
            ]
            available = 0

        self.messages = (
            system_messages
            + other_messages[-available:]
        )

    def get(
        self,
        role: str | None = None,
    ) -> list[ContextMessage]:

        if role is None:
            return list(self.messages)

        role = role.lower().strip()

        return [
            message
            for message in self.messages
            if message.role == role
        ]

    def last(self) -> ContextMessage | None:
        if not self.messages:
            return None

        return self.messages[-1]

    def clear(self) -> None:
        self.messages.clear()

    def count(self) -> int:
        return len(self.messages)

    def set_metadata(
        self,
        key: str,
        value: Any,
    ) -> None:
        self.metadata[key] = value

    def get_metadata(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        return self.metadata.get(
            key,
            default,
        )

    def to_dicts(self) -> list[dict[str, Any]]:
        return [
            {
                "role": message.role,
                "content": message.content,
                "metadata": message.metadata,
            }
            for message in self.messages
        ]

    def snapshot(self) -> dict[str, Any]:
        return {
            "messages": self.to_dicts(),
            "metadata": dict(self.metadata),
            "count": self.count(),
        }
