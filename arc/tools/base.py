from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable


class ToolError(Exception):
    """Raised when a tool operation fails."""


@dataclass
class Tool:
    """
    Represents a callable ARC tool.
    """

    name: str
    description: str
    function: Callable[..., Any]

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.function)

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool."""
        try:
            return self.function(*args, **kwargs)
        except Exception as exc:
            raise ToolError(
                f"Tool '{self.name}' failed: {exc}"
            ) from exc

    async def execute_async(self, *args: Any, **kwargs: Any) -> Any:
        """Execute sync or async tools asynchronously."""
        try:
            if self.is_async:
                return await self.function(*args, **kwargs)

            return self.function(*args, **kwargs)

        except Exception as exc:
            raise ToolError(
                f"Tool '{self.name}' failed: {exc}"
            ) from exc


def tool(
    name: str | None = None,
    description: str | None = None,
):
    """
    Decorator that converts a function into an ARC Tool.

    Example:

        @tool()
        def add(a, b):
            return a + b
    """

    def decorator(function: Callable[..., Any]) -> Tool:
        tool_name = name or function.__name__

        tool_description = (
            description
            or inspect.getdoc(function)
            or f"ARC tool: {tool_name}"
        )

        return Tool(
            name=tool_name,
            description=tool_description,
            function=function,
        )

    return decorator
