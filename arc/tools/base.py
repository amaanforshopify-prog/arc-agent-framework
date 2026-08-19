from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, get_args, get_origin, get_type_hints


class ToolError(Exception):
    """Raised when a tool operation fails."""


def _annotation_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Convert a Python annotation into a JSON-schema fragment."""

    if annotation is inspect.Signature.empty:
        return {
            "type": "string",
        }

    if annotation is Any:
        return {
            "type": "string",
        }

    origin = get_origin(annotation)
    args = get_args(annotation)

    if annotation is str:
        return {"type": "string"}

    if annotation is int:
        return {"type": "integer"}

    if annotation is float:
        return {"type": "number"}

    if annotation is bool:
        return {"type": "boolean"}

    if origin is list:
        item_schema = (
            _annotation_to_json_schema(args[0])
            if args
            else {"type": "string"}
        )

        return {
            "type": "array",
            "items": item_schema,
        }

    if origin is dict:
        return {
            "type": "object",
        }

    if origin is tuple:
        item_schema = (
            _annotation_to_json_schema(args[0])
            if args
            else {"type": "string"}
        )

        return {
            "type": "array",
            "items": item_schema,
        }

    if origin is not None and str(origin).endswith("Union"):
        non_none = [
            arg
            for arg in args
            if arg is not type(None)
        ]

        if len(non_none) == 1:
            return _annotation_to_json_schema(
                non_none[0]
            )

    return {
        "type": "string",
    }


@dataclass
class Tool:
    """
    Represents a callable ARC tool.

    Automatically derives an OpenAI-compatible function
    schema from the wrapped Python function.
    """

    name: str
    description: str
    function: Callable[..., Any]

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(
            self.function
        )

    def execute(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute the tool."""

        try:
            return self.function(
                *args,
                **kwargs,
            )

        except ToolError:
            raise

        except Exception as exc:
            raise ToolError(
                f"Tool '{self.name}' failed: {exc}"
            ) from exc

    async def execute_async(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute sync or async tools asynchronously."""

        try:
            if self.is_async:
                return await self.function(
                    *args,
                    **kwargs,
                )

            return self.function(
                *args,
                **kwargs,
            )

        except ToolError:
            raise

        except Exception as exc:
            raise ToolError(
                f"Tool '{self.name}' failed: {exc}"
            ) from exc

    def schema(self) -> dict[str, Any]:
        """
        Return an OpenAI-compatible function tool schema.

        Parameters are derived from the function signature and
        type annotations.
        """

        try:
            hints = get_type_hints(
                self.function
            )
        except Exception:
            hints = {}

        signature = inspect.signature(
            self.function
        )

        properties: dict[str, Any] = {}
        required: list[str] = []

        for parameter in signature.parameters.values():

            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            annotation = hints.get(
                parameter.name,
                parameter.annotation,
            )

            schema = _annotation_to_json_schema(
                annotation
            )

            if parameter.default is not inspect.Signature.empty:
                schema["default"] = parameter.default
            else:
                required.append(parameter.name)

            properties[parameter.name] = schema

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }


def tool(
    name: str | None = None,
    description: str | None = None,
):
    """
    Decorator that converts a Python function into an ARC Tool.

    Example:

        @tool()
        def add(a: int, b: int) -> int:
            return a + b
    """

    def decorator(
        function: Callable[..., Any]
    ) -> Tool:

        tool_name = (
            name
            or function.__name__
        )

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
