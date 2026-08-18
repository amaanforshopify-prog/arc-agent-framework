import asyncio
import time

from arc import Agent, Executor, ToolRegistry, tool


@tool()
def calculate(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool()
def slow_calculation() -> str:
    """Simulate a slow operation."""
    time.sleep(3)
    return "finished"


@tool()
async def async_calculate(a: int, b: int) -> int:
    """Async calculation."""
    await asyncio.sleep(0.05)
    return a * b


def main():
    registry = ToolRegistry()

    registry.register_many(
        calculate,
        slow_calculation,
        async_calculate,
    )

    executor = Executor(
        tools=registry,
        max_retries=1,
        retry_delay=0.1,
    )

    agent = Agent(
        name="CalculatorAgent",
        description="A simple ARC calculation agent.",
        executor=executor,
    )

    print("=" * 65)
    print("ARC AGENT TEST")
    print("=" * 65)

    # -----------------------------------------
    # 1. Normal agent task
    # -----------------------------------------

    print("\n[1] Normal task")

    result = agent.run(
        "Calculate 20 + 22",
        "calculate",
        20,
        22,
    )

    print(f"  Success: {result.success}")
    print(f"  Output: {result.output}")
    print(f"  Status: {agent.status}")

    assert result.success
    assert result.output == 42
    assert agent.status == "completed"

    print("  PASS")

    # -----------------------------------------
    # 2. Timeout through Agent
    # -----------------------------------------

    print("\n[2] Timeout through Agent")

    result = agent.run(
        "Run slow calculation",
        "slow_calculation",
        timeout=0.2,
        max_retries=1,
    )

    print(f"  Success: {result.success}")
    print(f"  Error: {result.error}")
    print(f"  Attempts: {result.execution.attempts}")

    assert not result.success
    assert result.execution.attempts == 2
    assert agent.status == "failed"

    print("  PASS")

    # -----------------------------------------
    # 3. Statistics
    # -----------------------------------------

    print("\n[3] Agent statistics")

    stats = agent.stats()

    print(f"  Name: {stats['name']}")
    print(f"  Status: {stats['status']}")
    print(f"  Total runs: {stats['total_runs']}")
    print(f"  Successful: {stats['successful_runs']}")
    print(f"  Failed: {stats['failed_runs']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")

    assert stats["total_runs"] == 2
    assert stats["successful_runs"] == 1
    assert stats["failed_runs"] == 1

    print("  PASS")

    # -----------------------------------------
    # 4. History
    # -----------------------------------------

    print("\n[4] Agent history")

    assert len(agent.history) == 2

    for index, item in enumerate(agent.history, 1):
        print(
            f"  Run {index}: "
            f"success={item.success}, "
            f"tool={item.tool_name}"
        )

    print("  PASS")

    executor.shutdown()

    print("\n" + "=" * 65)
    print("AGENT SYNC TESTS PASSED")
    print("=" * 65)


async def async_main():
    registry = ToolRegistry()
    registry.register(async_calculate)

    executor = Executor(
        tools=registry,
    )

    agent = Agent(
        name="AsyncAgent",
        executor=executor,
    )

    print("\n" + "=" * 65)
    print("ASYNC AGENT TEST")
    print("=" * 65)

    result = await agent.run_async(
        "Calculate 10 × 5",
        "async_calculate",
        10,
        5,
    )

    print(f"  Success: {result.success}")
    print(f"  Output: {result.output}")

    assert result.success
    assert result.output == 50

    print("  PASS")

    executor.shutdown()


if __name__ == "__main__":
    main()
    asyncio.run(async_main())
