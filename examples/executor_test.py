import asyncio
import time

from arc import Executor, ToolRegistry, tool


attempt_counter = {"value": 0}


@tool()
def add(a: int, b: int) -> int:
    return a + b


@tool()
def slow_tool():
    time.sleep(3)
    return "finished"


@tool()
def unstable_tool():
    attempt_counter["value"] += 1

    print(
        f"    unstable_tool attempt "
        f"{attempt_counter['value']}"
    )

    if attempt_counter["value"] < 3:
        raise RuntimeError("Temporary failure")

    return "Recovered successfully"


@tool()
async def async_add(a: int, b: int) -> int:
    await asyncio.sleep(0.05)
    return a + b


async def main():
    registry = ToolRegistry()

    registry.register_many(
        add,
        slow_tool,
        unstable_tool,
        async_add,
    )

    executor = Executor(
        tools=registry,
        max_retries=2,
        retry_delay=0.1,
    )

    print("=" * 65)
    print("ARC EXECUTOR INTEGRATION TEST")
    print("=" * 65)

    # ---------------------------------------------
    # 1. Normal tool
    # ---------------------------------------------

    print("\n[1] Normal tool execution")

    result = executor.execute(
        "add",
        20,
        22,
    )

    print(f"  Success: {result.success}")
    print(f"  Result: {result.result}")
    print(f"  Attempts: {result.attempts}")

    assert result.success
    assert result.result == 42

    print("  PASS")

    # ---------------------------------------------
    # 2. Retry
    # ---------------------------------------------

    print("\n[2] Retry integration")

    attempt_counter["value"] = 0

    result = executor.execute(
        "unstable_tool",
    )

    print(f"  Success: {result.success}")
    print(f"  Result: {result.result}")
    print(f"  Attempts: {result.attempts}")

    assert result.success
    assert result.attempts == 3
    assert result.result == "Recovered successfully"

    print("  PASS")

    # ---------------------------------------------
    # 3. Timeout
    # ---------------------------------------------

    print("\n[3] Timeout integration")

    result = executor.execute(
        "slow_tool",
        timeout=0.2,
        max_retries=1,
    )

    print(f"  Success: {result.success}")
    print(f"  Error: {result.error}")
    print(f"  Attempts: {result.attempts}")

    assert not result.success
    assert result.attempts == 2

    print("  PASS")

    # ---------------------------------------------
    # 4. Async execution
    # ---------------------------------------------

    print("\n[4] Async tool execution")

    result = await executor.execute_async(
        "async_add",
        100,
        50,
    )

    print(f"  Success: {result.success}")
    print(f"  Result: {result.result}")

    assert result.success
    assert result.result == 150

    print("  PASS")

    # ---------------------------------------------
    # 5. Unknown tool
    # ---------------------------------------------

    print("\n[5] Unknown tool")

    result = executor.execute(
        "does_not_exist",
        max_retries=2,
    )

    print(f"  Success: {result.success}")
    print(f"  Error: {result.error}")
    print(f"  Attempts: {result.attempts}")

    assert not result.success
    assert result.attempts == 3

    print("  PASS")

    # ---------------------------------------------
    # 6. Execution metrics
    # ---------------------------------------------

    print("\n[6] Execution metrics")

    result = executor.execute(
        "add",
        1,
        2,
    )

    print(f"  Duration: {result.duration:.4f}s")
    print(f"  Attempts: {result.attempts}")

    assert result.duration >= 0
    assert result.attempts == 1

    print("  PASS")

    executor.shutdown()

    print("\n" + "=" * 65)
    print("ALL EXECUTOR INTEGRATION TESTS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
