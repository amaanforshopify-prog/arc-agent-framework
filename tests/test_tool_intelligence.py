from __future__ import annotations

import time

from arc import (
    ToolIntelligenceError,
    ToolPolicy,
    ToolRegistry,
    tool,
)


@tool()
def add(
    a: int,
    b: int,
) -> int:
    return a + b


def test_execute_safe_success():
    registry = ToolRegistry()
    registry.register(add)

    result = registry.execute_safe(
        "add",
        2,
        3,
    )

    assert result.success
    assert result.output == 5
    assert result.attempts == 1
    assert result.tool_name == "add"


def test_execute_safe_validation():
    registry = ToolRegistry()
    registry.register(add)

    result = registry.execute_safe(
        "add",
        "2",
        3,
    )

    assert result.success is False
    assert isinstance(
        result.error,
        ToolIntelligenceError,
    )


def test_execute_safe_retry():
    registry = ToolRegistry()

    calls = {
        "count": 0
    }

    @tool()
    def flaky() -> str:
        calls["count"] += 1

        if calls["count"] < 3:
            raise RuntimeError(
                "temporary"
            )

        return "ok"

    registry.register(flaky)

    result = registry.execute_safe(
        "flaky",
        policy=ToolPolicy(
            max_attempts=3,
        ),
    )

    assert result.success
    assert result.output == "ok"
    assert result.attempts == 3


def test_execute_safe_disabled():
    registry = ToolRegistry()
    registry.register(add)

    result = registry.execute_safe(
        "add",
        1,
        2,
        policy=ToolPolicy(
            enabled=False
        ),
    )

    assert result.success is False


def test_execute_safe_denied():
    registry = ToolRegistry()
    registry.register(add)

    result = registry.execute_safe(
        "add",
        1,
        2,
        policy=ToolPolicy(
            allowed=False
        ),
    )

    assert result.success is False


def test_execution_history():
    registry = ToolRegistry()
    registry.register(add)

    registry.execute_safe(
        "add",
        1,
        2,
    )

    registry.execute_safe(
        "add",
        3,
        4,
    )

    history = registry.execution_history()

    assert len(history) == 2
    assert all(
        item.success
        for item in history
    )


def test_execution_history_clear():
    registry = ToolRegistry()
    registry.register(add)

    registry.execute_safe(
        "add",
        1,
        2,
    )

    assert len(
        registry.execution_history()
    ) == 1

    registry.clear_execution_history()

    assert registry.execution_history() == []


def test_timeout_policy():
    registry = ToolRegistry()

    @tool()
    def slow() -> str:
        time.sleep(0.05)
        return "done"

    registry.register(slow)

    result = registry.execute_safe(
        "slow",
        policy=ToolPolicy(
            timeout=0.01
        ),
    )

    assert result.success is False
