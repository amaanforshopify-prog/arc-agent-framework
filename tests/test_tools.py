import asyncio

import pytest

from arc.tools import Tool, ToolError, ToolRegistry, tool


@tool()
def add(a, b):
    return a + b


@tool(name="custom")
def custom_tool(value):
    return value * 2


@tool()
async def async_add(a, b):
    await asyncio.sleep(0.01)
    return a + b


@tool()
def failing_tool():
    raise ValueError("simulated failure")


def test_tool_creation():
    assert isinstance(add, Tool)
    assert add.name == "add"


def test_tool_description():
    assert add.description


def test_sync_execution():
    assert add.execute(2, 3) == 5


@pytest.mark.asyncio
async def test_async_execution():
    result = await async_add.execute_async(5, 10)
    assert result == 15


def test_registry_register():
    registry = ToolRegistry()

    registry.register(add)

    assert registry.has("add")
    assert registry.count() == 1


def test_registry_get():
    registry = ToolRegistry()

    registry.register(add)

    assert registry.get("add") is add


def test_registry_execute():
    registry = ToolRegistry()

    registry.register(add)

    assert registry.execute("add", 10, 20) == 30


def test_custom_name():
    registry = ToolRegistry()

    registry.register(custom_tool)

    assert registry.has("custom")
    assert registry.execute("custom", 5) == 10


def test_duplicate_registration():
    registry = ToolRegistry()

    registry.register(add)

    with pytest.raises(ToolError):
        registry.register(add)


def test_unknown_tool():
    registry = ToolRegistry()

    with pytest.raises(ToolError):
        registry.get("does_not_exist")


def test_tool_failure():
    with pytest.raises(ToolError):
        failing_tool.execute()


def test_unregister():
    registry = ToolRegistry()

    registry.register(add)

    removed = registry.unregister("add")

    assert removed is add
    assert not registry.has("add")
    assert registry.count() == 0


def test_register_many():
    registry = ToolRegistry()

    registry.register_many(
        add,
        custom_tool,
    )

    assert registry.count() == 2


def test_list_tools():
    registry = ToolRegistry()

    registry.register_many(
        add,
        custom_tool,
    )

    names = registry.names()

    assert "add" in names
    assert "custom" in names


def test_clear():
    registry = ToolRegistry()

    registry.register(add)
    registry.clear()

    assert registry.count() == 0


@pytest.mark.asyncio
async def test_registry_async_execution():
    registry = ToolRegistry()

    registry.register(async_add)

    result = await registry.execute_async(
        "async_add",
        10,
        20,
    )

    assert result == 30


def test_invalid_registration():
    registry = ToolRegistry()

    with pytest.raises(TypeError):
        registry.register(lambda: "not a Tool")
