import asyncio
import time

import pytest

from arc import Executor, ToolRegistry, tool


@pytest.fixture
def registry():
    registry = ToolRegistry()

    @tool()
    def add(a, b):
        return a + b

    @tool()
    def slow():
        time.sleep(1)
        return "done"

    @tool()
    async def async_add(a, b):
        await asyncio.sleep(0.01)
        return a + b

    registry.register_many(
        add,
        slow,
        async_add,
    )

    return registry


def test_executor_success(registry):
    executor = Executor(registry)

    result = executor.execute(
        "add",
        10,
        20,
    )

    assert result.success is True
    assert result.result == 30
    assert result.attempts == 1

    executor.shutdown()


def test_executor_timeout(registry):
    executor = Executor(
        registry,
        max_retries=1,
    )

    result = executor.execute(
        "slow",
        timeout=0.1,
    )

    assert result.success is False
    assert result.attempts == 2
    assert result.error is not None

    executor.shutdown()


@pytest.mark.asyncio
async def test_executor_async(registry):
    executor = Executor(registry)

    result = await executor.execute_async(
        "async_add",
        5,
        7,
    )

    assert result.success is True
    assert result.result == 12

    executor.shutdown()


def test_executor_unknown_tool(registry):
    executor = Executor(
        registry,
        max_retries=2,
    )

    result = executor.execute(
        "missing",
    )

    assert result.success is False
    assert result.attempts == 3
    assert result.error is not None

    executor.shutdown()


def test_executor_custom_retries(registry):
    executor = Executor(
        registry,
        max_retries=0,
    )

    result = executor.execute(
        "missing",
        max_retries=3,
    )

    assert result.success is False
    assert result.attempts == 4

    executor.shutdown()


def test_execution_duration(registry):
    executor = Executor(registry)

    result = executor.execute(
        "add",
        1,
        1,
    )

    assert result.duration >= 0

    executor.shutdown()
