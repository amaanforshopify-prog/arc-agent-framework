from __future__ import annotations

from typing import Any

from .base import Tool, ToolError


class ToolRegistry:
    """
    Stores and manages ARC tools.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        """Register a tool."""
        if not isinstance(tool, Tool):
            raise TypeError("Only Tool instances can be registered.")

        if tool.name in self._tools:
            raise ToolError(
                f"Tool '{tool.name}' is already registered."
            )

        self._tools[tool.name] = tool

        return tool

    def register_many(self, *tools: Tool) -> None:
        """Register multiple tools."""
        for item in tools:
            self.register(item)

    def unregister(self, name: str) -> Tool:
        """Remove and return a registered tool."""
        if name not in self._tools:
            raise ToolError(
                f"Tool '{name}' is not registered."
            )

        return self._tools.pop(name)

    def get(self, name: str) -> Tool:
        """Get a tool by name."""
        if name not in self._tools:
            raise ToolError(
                f"Tool '{name}' is not registered."
            )

        return self._tools[name]

    def has(self, name: str) -> bool:
        """Check whether a tool exists."""
        return name in self._tools

    def list(self) -> list[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def names(self) -> list[str]:
        """Return registered tool names."""
        return list(self._tools.keys())

    def count(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)

    def execute(
        self,
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a registered tool."""
        return self.get(name).execute(*args, **kwargs)

    async def execute_async(
        self,
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a registered tool asynchronously."""
        return await self.get(name).execute_async(*args, **kwargs)

    def clear(self) -> None:
        """Remove all tools."""
        self._tools.clear()
