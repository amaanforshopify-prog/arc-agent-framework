from __future__ import annotations

from typing import Any


class ConversationMemory:
    """Ordered in-memory conversation history."""

    def __init__(self, max_messages: int | None = None) -> None:
        if max_messages is not None and max_messages <= 0:
            raise ValueError(
                "max_messages must be greater than 0"
            )

        self.max_messages = max_messages
        self._messages: list[Any] = []

    def add(
        self,
        message_or_role: Any,
        content: Any = None,
        **extra: Any,
    ) -> Any:
        # Supports both:
        #   add(Message(...))
        #   add("user", "hello")
        if isinstance(message_or_role, str):
            if not message_or_role.strip():
                raise ValueError("role cannot be empty")

            message = {
                "role": message_or_role,
                "content": content,
                **extra,
            }
        else:
            message = message_or_role

        self._messages.append(message)

        if self.max_messages is not None:
            overflow = (
                len(self._messages)
                - self.max_messages
            )

            if overflow > 0:
                del self._messages[:overflow]

        return message

    def get_messages(self) -> list[Any]:
        return list(self._messages)

    def get(self) -> list[Any]:
        return self.get_messages()

    def add_user(self, content: Any, **extra: Any) -> Any:
        return self.add("user", content, **extra)

    def add_assistant(self, content: Any, **extra: Any) -> Any:
        return self.add("assistant", content, **extra)

    def add_system(self, content: Any, **extra: Any) -> Any:
        return self.add("system", content, **extra)

    def add_tool(self, content: Any, **extra: Any) -> Any:
        return self.add("tool", content, **extra)

    def last(self) -> Any:
        if not self._messages:
            return None

        return self._messages[-1]

    def clear(self) -> None:
        self._messages.clear()

    def count(self) -> int:
        return len(self._messages)

    def __len__(self) -> int:
        return len(self._messages)
