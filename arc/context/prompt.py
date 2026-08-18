from __future__ import annotations

from typing import Any

from .manager import ContextManager


class PromptBuilder:
    """Build structured prompts from ARC context."""

    def __init__(self, system: str | None = None):
        self.context = ContextManager()

        if system:
            self.context.system(system)

    def add_user(self, content: str) -> "PromptBuilder":
        self.context.user(content)
        return self

    def add_assistant(self, content: str) -> "PromptBuilder":
        self.context.assistant(content)
        return self

    def add_tool(self, content: str) -> "PromptBuilder":
        self.context.tool(content)
        return self

    def add_system(self, content: str) -> "PromptBuilder":
        self.context.system(content)
        return self

    def build(self) -> str:
        parts = []

        for message in self.context.messages:
            parts.append(
                f"[{message.role.upper()}]\n"
                f"{message.content}"
            )

        return "\n\n".join(parts)

    def messages(self) -> list[dict[str, Any]]:
        return self.context.to_dicts()

    def clear(self) -> "PromptBuilder":
        self.context.clear()
        return self

    def count(self) -> int:
        return self.context.count()
