from __future__ import annotations

from types import SimpleNamespace

import pytest

from arc import (

    ToolRegistry,
    WorkflowOrchestrator,
    WorkflowResult,
    tool,
)
from arc.runtime import AgentRuntime
from arc.tracing import Tracer
from arc.trace import TraceManager


class FakeModel:
    def __init__(
        self,
        output: str = "Workflow completed.",
    ):
        self.output = output
        self.calls = 0

    def generate(
        self,
        messages,
        tools=None,
    ):
        self.calls += 1

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=self.output,
                        tool_calls=[],
                    )
                )
            ]
        )


@tool()
def add(a: int, b: int) -> int:
    return a + b


def build_orchestrator():
    registry = ToolRegistry()
    registry.register(add)

    runtime = AgentRuntime(
        model=FakeModel(),
        tools=registry,
        tracer=Tracer(
            enabled=False
        ),
    )

    return WorkflowOrchestrator(
        runtime=runtime,
        trace_manager=TraceManager(),
    )


def test_orchestrator_creation():
    orchestrator = build_orchestrator()

    assert (
        orchestrator.run_count()
        == 0
    )

    assert (
        orchestrator.last_result()
        is None
    )


def test_create_plan():
    orchestrator = build_orchestrator()

    plan = orchestrator.create_plan(
        "Calculate values.",
        [
            "Understand request",
            "Use arithmetic tool",
        ],
    )

    assert plan.goal == "Calculate values."
    assert len(plan.steps) == 2
    assert plan.steps[0].description == (
        "Understand request"
    )
    assert plan.steps[1].description == (
        "Use arithmetic tool"
    )


def test_workflow_run():
    orchestrator = build_orchestrator()

    result = orchestrator.run(
        goal="Answer the user.",
        system_prompt="You are ARC.",
        user_input="Hello ARC.",
    )

    assert isinstance(
        result,
        WorkflowResult,
    )

    assert result.success
    assert result.output == (
        "Workflow completed."
    )

    assert result.plan is not None
    assert result.trace is not None

    assert result.metadata["run_id"] == 1
    assert (
        result.metadata["goal"]
        == "Answer the user."
    )


def test_workflow_run_with_plan():
    orchestrator = build_orchestrator()

    result = orchestrator.run(
        goal="Calculate answer.",
        system_prompt="You are ARC.",
        user_input="Calculate.",
        steps=[
            "Inspect request",
            "Calculate",
            "Respond",
        ],
    )

    assert result.success
    assert result.plan is not None
    assert result.plan.status == (
        "completed"
    )
    assert len(result.plan.steps) == 3


def test_workflow_trace():
    orchestrator = build_orchestrator()

    result = orchestrator.run(
        goal="Trace workflow.",
        system_prompt="You are ARC.",
        user_input="Trace this.",
    )

    events = result.trace.events

    event_types = [
        event.event_type
        for event in events
    ]

    assert (
        "workflow_started"
        in event_types
    )

    assert (
        "planning_started"
        in event_types
    )

    assert (
        "planning_completed"
        in event_types
    )

    assert (
        "runtime_started"
        in event_types
    )

    assert (
        "runtime_completed"
        in event_types
    )

    assert (
        "workflow_completed"
        in event_types
    )


def test_last_result():
    orchestrator = build_orchestrator()

    assert (
        orchestrator.last_result()
        is None
    )

    first = orchestrator.run(
        "First",
        "System.",
        "First.",
    )

    second = orchestrator.run(
        "Second",
        "System.",
        "Second.",
    )

    assert (
        orchestrator.last_result()
        is second
    )

    assert (
        orchestrator.last_result()
        is not first
    )


def test_run_count():
    orchestrator = build_orchestrator()

    orchestrator.run(
        "One",
        "System.",
        "One.",
    )

    orchestrator.run(
        "Two",
        "System.",
        "Two.",
    )

    assert (
        orchestrator.run_count()
        == 2
    )


def test_snapshot():
    orchestrator = build_orchestrator()

    snapshot = (
        orchestrator.snapshot()
    )

    assert snapshot["runs"] == 0
    assert (
        snapshot["last_success"]
        is None
    )

    orchestrator.run(
        "Snapshot",
        "System.",
        "Hello.",
    )

    snapshot = (
        orchestrator.snapshot()
    )

    assert snapshot["runs"] == 1
    assert snapshot["last_success"] is True
    assert snapshot["last_run_id"] == 1
    assert snapshot["last_duration"] >= 0


def test_clear_history():
    orchestrator = build_orchestrator()

    orchestrator.run(
        "Clear",
        "System.",
        "Hello.",
    )

    assert (
        orchestrator.run_count()
        == 1
    )

    orchestrator.clear_history()

    assert (
        orchestrator.run_count()
        == 0
    )

    assert (
        orchestrator.last_result()
        is None
    )


def test_invalid_goal():
    orchestrator = build_orchestrator()

    with pytest.raises(
        ValueError
    ):
        orchestrator.run(
            "",
            "System.",
            "Hello.",
        )


def test_invalid_system_prompt():
    orchestrator = build_orchestrator()

    with pytest.raises(
        ValueError
    ):
        orchestrator.run(
            "Goal",
            "",
            "Hello.",
        )


def test_invalid_user_input():
    orchestrator = build_orchestrator()

    with pytest.raises(
        ValueError
    ):
        orchestrator.run(
            "Goal",
            "System.",
            "",
        )


def test_workflow_failure_is_captured():
    class FailingModel:
        def generate(
            self,
            messages,
            tools=None,
        ):
            raise RuntimeError(
                "model failure"
            )

    runtime = AgentRuntime(
        model=FailingModel(),
        tools=ToolRegistry(),
        tracer=Tracer(
            enabled=False
        ),
    )

    orchestrator = WorkflowOrchestrator(
        runtime=runtime,
        trace_manager=TraceManager(),
    )

    result = orchestrator.run(
        goal="Fail safely.",
        system_prompt="System.",
        user_input="Trigger.",
    )

    assert result.success is False
    assert isinstance(
        result.error,
        RuntimeError,
    )

    event_types = [
        event.event_type
        for event in result.trace.events
    ]

    assert (
        "workflow_failed"
        in event_types
    )

