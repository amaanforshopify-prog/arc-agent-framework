from __future__ import annotations

from typing import Any

from .base import Tool, ToolError


class ToolRegistry:
    """Registry for ARC tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if not isinstance(tool, Tool):
            raise TypeError(
                "tool must be a Tool instance."
            )

        if tool.name in self._tools:
            raise ToolError(
                f"Tool already registered: {tool.name}"
            )

        self._tools[tool.name] = tool
        return tool

    def register_many(
        self,
        *tools: Tool,
    ) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool:
        if not isinstance(name, str):
            raise TypeError(
                "tool name must be a string."
            )

        if name not in self._tools:
            raise ToolError(
                f"Unknown tool: {name}"
            )

        return self._tools[name]

    def has(self, name: str) -> bool:
        return (
            isinstance(name, str)
            and name in self._tools
        )

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

    async def execute_async(
        self,
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        tool = self.get(name)

        if not hasattr(tool, "execute_async"):
            raise ToolError(
                f"Tool '{name}' does not support async execution."
            )

        return await tool.execute_async(
            *args,
            **kwargs,
        )

    def unregister(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(
                f"Unknown tool: {name}"
            )

        return self._tools.pop(name)

    def clear(self) -> None:
        self._tools.clear()

    def list_tools(self) -> list[Tool]:
        return list(
            self._tools.values()
        )

    def names(self) -> list[str]:
        return list(
            self._tools.keys()
        )

    def count(self) -> int:
        return len(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """
        Return OpenAI-compatible function tool schemas.

        This is consumed by the LLM runtime so the model knows
        which tools are available and what arguments they accept.
        """

        schemas: list[dict[str, Any]] = []

        for tool in self._tools.values():
            schema_method = getattr(
                tool,
                "schema",
                None,
            )

            if not callable(schema_method):
                raise ToolError(
                    f"Tool '{tool.name}' does not provide a schema()."
                )

            schemas.append(
                schema_method()
            )

        return schemas
