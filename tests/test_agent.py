import asyncio
import time

import pytest

from arc import Agent, Executor, ToolRegistry, tool


@pytest.fixture
def agent():
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

    executor = Executor(
        tools=registry,
        max_retries=1,
    )

    return Agent(
        name="TestAgent",
        executor=executor,
    )


def test_agent_creation(agent):
    assert agent.name == "TestAgent"
    assert agent.status == "idle"
    assert agent.total_runs == 0


def test_agent_success(agent):
    result = agent.run(
        "Add numbers",
        "add",
        10,
        20,
    )

    assert result.success
    assert result.output == 30
    assert agent.status == "completed"


def test_agent_failure(agent):
    result = agent.run(
        "Missing tool",
        "missing",
    )

    assert not result.success
    assert agent.status == "failed"


def test_agent_timeout(agent):
    result = agent.run(
        "Slow operation",
        "slow",
        timeout=0.1,
        max_retries=1,
    )

    assert not result.success
    assert result.execution is not None
    assert result.execution.attempts == 2
    assert agent.failed_runs == 1


@pytest.mark.asyncio
async def test_agent_async(agent):
    result = await agent.run_async(
        "Async addition",
        "async_add",
        5,
        7,
    )

    assert result.success
    assert result.output == 12


def test_agent_history(agent):
    agent.run(
        "First",
        "add",
        1,
        2,
    )

    agent.run(
        "Second",
        "add",
        3,
        4,
    )

    assert len(agent.history) == 2
    assert agent.total_runs == 2


def test_agent_stats(agent):
    agent.run(
        "Success",
        "add",
        1,
        1,
    )

    agent.run(
        "Failure",
        "missing",
    )

    stats = agent.stats()

    assert stats["total_runs"] == 2
    assert stats["successful_runs"] == 1
    assert stats["failed_runs"] == 1
    assert stats["success_rate"] == 0.5


def test_agent_reset(agent):
    agent.run(
        "Test",
        "add",
        1,
        1,
    )

    agent.reset()

    assert agent.status == "idle"
    assert agent.total_runs == 0
    assert agent.successful_runs == 0
    assert agent.failed_runs == 0
    assert agent.history == []


def test_empty_task(agent):
    with pytest.raises(ValueError):
        agent.run(
            "",
            "add",
            1,
            2,
        )


def test_agent_metadata(agent):
    result = agent.run(
        "Metadata test",
        "add",
        2,
        3,
    )

    assert result.metadata["agent"] == "TestAgent"
    assert result.metadata["task"] == "Metadata test"


def test_agent_duration(agent):
    result = agent.run(
        "Duration test",
        "add",
        1,
        2,
    )

    assert result.duration >= 0
