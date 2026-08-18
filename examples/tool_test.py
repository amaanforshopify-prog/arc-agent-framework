import asyncio

from arc.tools import Tool, ToolError, ToolRegistry, tool


@tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool(
    name="multiply",
    description="Multiply two numbers.",
)
def multiply(a: int, b: int) -> int:
    return a * b


@tool()
def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")

    return a / b


@tool()
async def async_add(a: int, b: int) -> int:
    """Async addition."""
    await asyncio.sleep(0.05)
    return a + b


def main():
    print("=" * 60)
    print("ARC TOOL SYSTEM TESTS")
    print("=" * 60)

    registry = ToolRegistry()

    # --------------------------------------------------
    # 1. Register tools
    # --------------------------------------------------

    print("\n[1] Register tools")

    registry.register(add)
    registry.register(multiply)
    registry.register(divide)
    registry.register(async_add)

    print(f"  Registered: {registry.names()}")
    print(f"  Count: {registry.count()}")

    assert registry.count() == 4
    assert registry.has("add")
    assert registry.has("multiply")

    print("  PASS")

    # --------------------------------------------------
    # 2. Get tool
    # --------------------------------------------------

    print("\n[2] Get tool")

    selected = registry.get("add")

    assert selected.name == "add"
    assert selected.description == "Add two numbers."

    print(f"  Name: {selected.name}")
    print(f"  Description: {selected.description}")
    print("  PASS")

    # --------------------------------------------------
    # 3. Execute tool
    # --------------------------------------------------

    print("\n[3] Execute tool")

    result = registry.execute("add", 10, 20)

    print(f"  10 + 20 = {result}")

    assert result == 30

    print("  PASS")

    # --------------------------------------------------
    # 4. Execute named tool
    # --------------------------------------------------

    print("\n[4] Execute named tool")

    result = registry.execute("multiply", 6, 7)

    print(f"  6 × 7 = {result}")

    assert result == 42

    print("  PASS")

    # --------------------------------------------------
    # 5. Tool exception handling
    # --------------------------------------------------

    print("\n[5] Tool error handling")

    try:
        registry.execute("divide", 10, 0)
        print("  FAIL: error expected")

    except ToolError as exc:
        print(f"  PASS: {exc}")

    # --------------------------------------------------
    # 6. Unknown tool
    # --------------------------------------------------

    print("\n[6] Unknown tool handling")

    try:
        registry.execute("unknown_tool", 1, 2)
        print("  FAIL: error expected")

    except ToolError as exc:
        print(f"  PASS: {exc}")

    # --------------------------------------------------
    # 7. Duplicate registration
    # --------------------------------------------------

    print("\n[7] Duplicate registration")

    try:
        registry.register(add)
        print("  FAIL: duplicate error expected")

    except ToolError as exc:
        print(f"  PASS: {exc}")

    # --------------------------------------------------
    # 8. Async tool
    # --------------------------------------------------

    print("\n[8] Async tool")

    async def async_test():
        result = await registry.execute_async(
            "async_add",
            100,
            50,
        )

        print(f"  100 + 50 = {result}")

        assert result == 150

    asyncio.run(async_test())

    print("  PASS")

    # --------------------------------------------------
    # 9. List tools
    # --------------------------------------------------

    print("\n[9] List tools")

    tools = registry.list()

    for item in tools:
        print(
            f"  - {item.name}: {item.description}"
        )

    assert len(tools) == 4

    print("  PASS")

    # --------------------------------------------------
    # 10. Unregister
    # --------------------------------------------------

    print("\n[10] Unregister tool")

    removed = registry.unregister("multiply")

    assert removed.name == "multiply"
    assert not registry.has("multiply")
    assert registry.count() == 3

    print(f"  Removed: {removed.name}")
    print("  PASS")

    # --------------------------------------------------
    # 11. Clear
    # --------------------------------------------------

    print("\n[11] Clear registry")

    registry.clear()

    assert registry.count() == 0
    assert registry.names() == []

    print("  Registry cleared")
    print("  PASS")

    print("\n" + "=" * 60)
    print("ALL TOOL SYSTEM TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
