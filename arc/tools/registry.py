from __future__ import annotations

from typing import Any

from ..tool_intelligence import (
    ToolExecutionResult,
    ToolIntelligence,
    ToolPolicy,
)
from .base import Tool, ToolError


class ToolRegistry:
    """Registry for ARC tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._policies: dict[str, ToolPolicy] = {}
        self._approved: set[str] = set()

        self.intelligence = ToolIntelligence()

    def register(self, tool: Tool) -> Tool:
        if not isinstance(tool, Tool):
            raise TypeError(
                "Only Tool instances can be registered."
            )

        if tool.name in self._tools:
            raise ToolError(
                f"Tool '{tool.name}' is already registered."
            )

        self._tools[tool.name] = tool
        return tool

    def register_many(
        self,
        *tools: Tool,
    ) -> None:
        for item in tools:
            self.register(item)

    def unregister(
        self,
        name: str,
    ) -> Tool:
        if name not in self._tools:
            raise ToolError(
                f"Tool '{name}' is not registered."
            )

        self._policies.pop(
            name,
            None,
        )
        self._approved.discard(name)

        return self._tools.pop(name)

    def get(
        self,
        name: str,
    ) -> Tool:
        if name not in self._tools:
            raise ToolError(
                f"Tool '{name}' is not registered."
            )

        return self._tools[name]

    def has(
        self,
        name: str,
    ) -> bool:
        return name in self._tools

    def list(
        self,
    ) -> list[Tool]:
        return list(
            self._tools.values()
        )

    def names(
        self,
    ) -> list[str]:
        return list(
            self._tools.keys()
        )

    def count(
        self,
    ) -> int:
        return len(self._tools)

    def set_policy(
        self,
        name: str,
        policy: ToolPolicy,
    ) -> None:
        if name not in self._tools:
            raise ToolError(
                f"Tool '{name}' is not registered."
            )

        if not isinstance(
            policy,
            ToolPolicy,
        ):
            raise TypeError(
                "policy must be a ToolPolicy."
            )

        self._policies[name] = policy

    def get_policy(
        self,
        name: str,
    ) -> ToolPolicy:
        if name not in self._tools:
            raise ToolError(
                f"Tool '{name}' is not registered."
            )

        return self._policies.get(
            name,
            ToolPolicy(),
        )

    def approve(
        self,
        name: str,
    ) -> None:
        if name not in self._tools:
            raise ToolError(
                f"Tool '{name}' is not registered."
            )

        self._approved.add(name)

    def revoke_approval(
        self,
        name: str,
    ) -> None:
        self._approved.discard(name)

    def is_approved(
        self,
        name: str,
    ) -> bool:
        return name in self._approved

    def execute(
        self,
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return self.get(name).execute(
            *args,
            **kwargs,
        )

    def execute_safe(
        self,
        name: str,
        *args: Any,
        policy: ToolPolicy | None = None,
        **kwargs: Any,
    ) -> ToolExecutionResult:
        tool = self.get(name)

        effective_policy = (
            policy
            if policy is not None
            else self.get_policy(name)
        )

        return self.intelligence.execute(
            name,
            tool.function,
            *args,
            policy=effective_policy,
            approved=self.is_approved(name),
            **kwargs,
        )

    async def execute_async(
        self,
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return await self.get(
            name
        ).execute_async(
            *args,
            **kwargs,
        )

    def schemas(
        self,
    ) -> list[dict[str, Any]]:
        return [
            tool.schema()
            for tool
            in self._tools.values()
        ]

    def clear(
        self,
    ) -> None:
        self._tools.clear()
        self._policies.clear()
        self._approved.clear()

    def execution_history(self):
        return self.intelligence.history()

    def clear_execution_history(self) -> None:
        self.intelligence.clear_history()
