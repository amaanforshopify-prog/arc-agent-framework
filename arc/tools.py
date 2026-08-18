from abc import ABC, abstractmethod
from typing import Any, Callable


class Tool(ABC):
    """Base class for every ARC tool."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
    ):
        self.name = name
        self.description = description
        self.parameters = parameters

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool."""
        raise NotImplementedError

    def schema(self) -> dict[str, Any]:
        """Return an OpenAI-compatible tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class FunctionTool(Tool):
    """Wrap a normal Python function as an ARC tool."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        function: Callable[..., Any],
    ):
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
        )
        self.function = function

    def execute(self, **kwargs: Any) -> Any:
        return self.function(**kwargs)


class ToolRegistry:
    """Registers, finds, and executes ARC tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered."
            )

        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(
                f"Tool '{name}' is not registered."
            )

        return self._tools[name]

    def execute(
        self,
        name: str,
        **kwargs: Any,
    ) -> Any:
        tool = self.get(name)
        return tool.execute(**kwargs)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def has(self, name: str) -> bool:
        return name in self._tools

    def schemas(self) -> list[dict[str, Any]]:
        """Return schemas for all registered tools."""
        return [
            tool.schema()
            for tool in self._tools.values()
        ]