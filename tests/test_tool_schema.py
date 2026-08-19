import asyncio

import pytest

from arc import ToolError, ToolRegistry, tool


@tool(
    description="Adds two integers together."
)
def typed_add(
    a: int,
    b: int,
) -> int:
    return a + b


@tool()
def default_tool(
    name: str,
    times: int = 2,
) -> str:
    return name * times


@tool()
async def async_typed_add(
    a: int,
    b: int,
) -> int:
    await asyncio.sleep(0.001)
    return a + b


def test_tool_schema_name():
    schema = typed_add.schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "typed_add"


def test_tool_schema_description():
    schema = typed_add.schema()

    assert (
        schema["function"]["description"]
        == "Adds two integers together."
    )


def test_tool_schema_integer_parameters():
    schema = typed_add.schema()

    parameters = (
        schema["function"]["parameters"]
    )

    assert parameters["type"] == "object"

    assert (
        parameters["properties"]["a"]["type"]
        == "integer"
    )

    assert (
        parameters["properties"]["b"]["type"]
        == "integer"
    )

    assert parameters["required"] == [
        "a",
        "b",
    ]


def test_tool_schema_defaults():
    schema = default_tool.schema()

    parameters = (
        schema["function"]["parameters"]
    )

    assert (
        parameters["properties"]["name"]["type"]
        == "string"
    )

    assert (
        parameters["properties"]["times"]["type"]
        == "integer"
    )

    assert (
        parameters["properties"]["times"]["default"]
        == 2
    )

    assert parameters["required"] == [
        "name",
    ]


def test_registry_schemas():
    registry = ToolRegistry()

    registry.register(
        typed_add
    )

    schemas = registry.schemas()

    assert len(schemas) == 1
    assert (
        schemas[0]["function"]["name"]
        == "typed_add"
    )


@pytest.mark.asyncio
async def test_async_tool_schema_and_execution():
    registry = ToolRegistry()

    registry.register(
        async_typed_add
    )

    schema = registry.schemas()[0]

    assert (
        schema["function"]["name"]
        == "async_typed_add"
    )

    result = await registry.execute_async(
        "async_typed_add",
        4,
        8,
    )

    assert result == 12


def test_tool_failure_is_wrapped():
    @tool()
    def broken():
        raise ValueError("boom")

    with pytest.raises(
        ToolError,
        match="boom",
    ):
        broken.execute()


def test_multiple_tool_schemas():
    registry = ToolRegistry()

    registry.register_many(
        typed_add,
        default_tool,
    )

    assert registry.count() == 2

    names = [
        schema["function"]["name"]
        for schema in registry.schemas()
    ]

    assert names == [
        "typed_add",
        "default_tool",
    ]
