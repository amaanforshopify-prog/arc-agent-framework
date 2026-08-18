import pytest

from arc import (
    Agent,
    AgentRuntime,
    ContextManager,
    Memory,
    PlanningEngine,
    TraceManager,
)


def make_agent():
    return Agent(
        name="RuntimeAgent",
        executor=PlanningEngine(),
    )


def test_runtime_creation():
    runtime = AgentRuntime(make_agent())

    assert runtime.run_count() == 0
    assert isinstance(
        runtime.trace_manager,
        TraceManager,
    )


def test_runtime_run():
    runtime = AgentRuntime(make_agent())

    run = runtime.run(
        "Simple task",
        ["Do the task"],
    )

    assert run.success
    assert run.run_id == 1
    assert run.trace is not None


def test_runtime_result():
    runtime = AgentRuntime(make_agent())

    run = runtime.run(
        "Calculate",
        ["Calculate 10 + 20"],
    )

    assert run.result == "Calculate 10 + 20"


def test_runtime_context():
    context = ContextManager()

    runtime = AgentRuntime(
        make_agent(),
        context=context,
    )

    runtime.run(
        "Hello",
        ["World"],
    )

    assert context.count() == 2


def test_runtime_multiple_runs():
    runtime = AgentRuntime(make_agent())

    runtime.run("One", ["First"])
    runtime.run("Two", ["Second"])
    runtime.run("Three", ["Third"])

    assert runtime.run_count() == 3


def test_last_run():
    runtime = AgentRuntime(make_agent())

    runtime.run(
        "Test",
        ["Step"],
    )

    assert runtime.last_run() is not None
    assert runtime.last_run().run_id == 1


def test_clear_runs():
    runtime = AgentRuntime(make_agent())

    runtime.run(
        "Test",
        ["Step"],
    )

    runtime.clear_runs()

    assert runtime.run_count() == 0
    assert runtime.last_run() is None


def test_snapshot():
    runtime = AgentRuntime(make_agent())

    runtime.run(
        "Test",
        ["Step"],
    )

    snapshot = runtime.snapshot()

    assert snapshot["runs"] == 1
    assert snapshot["context_messages"] == 2
    assert snapshot["last_run"] == 1
    assert snapshot["last_trace_events"] > 0


def test_memory_integration():
    memory = Memory()

    runtime = AgentRuntime(
        make_agent(),
        memory=memory,
    )

    run = runtime.run(
        "Remember this",
        ["Important information"],
    )

    assert run.success
    assert run.memory_used is False
    assert memory.has("run_1")


def test_runtime_invalid_task():
    runtime = AgentRuntime(make_agent())

    with pytest.raises(ValueError):
        runtime.run("")


def test_runtime_reset_context():
    context = ContextManager()

    runtime = AgentRuntime(
        make_agent(),
        context=context,
    )

    runtime.run(
        "Test",
        ["Step"],
    )

    runtime.reset_context()

    assert context.count() == 0


def test_trace_created():
    runtime = AgentRuntime(make_agent())

    run = runtime.run(
        "Trace test",
        ["Step"],
    )

    assert run.trace is not None
    assert run.trace.run_id == "run-1"


def test_trace_has_events():
    runtime = AgentRuntime(make_agent())

    run = runtime.run(
        "Trace test",
        ["Step"],
    )

    event_types = [
        event.event_type
        for event in run.trace.events
    ]

    assert "run_started" in event_types
    assert "planning_started" in event_types
    assert "planning_completed" in event_types
    assert "execution_started" in event_types
    assert "execution_completed" in event_types
    assert "run_completed" in event_types


def test_trace_event_count():
    runtime = AgentRuntime(make_agent())

    run = runtime.run(
        "Trace test",
        ["Step"],
    )

    assert run.trace.count() >= 6


def test_get_last_trace():
    runtime = AgentRuntime(make_agent())

    run = runtime.run(
        "Trace test",
        ["Step"],
    )

    trace = runtime.get_trace()

    assert trace is run.trace


def test_get_trace_by_id():
    runtime = AgentRuntime(make_agent())

    run = runtime.run(
        "Trace test",
        ["Step"],
    )

    trace = runtime.get_trace(run.run_id)

    assert trace is run.trace


def test_trace_finished():
    runtime = AgentRuntime(make_agent())

    run = runtime.run(
        "Trace test",
        ["Step"],
    )

    assert run.trace.finished_at is not None


def test_custom_trace_manager():
    manager = TraceManager()

    runtime = AgentRuntime(
        make_agent(),
        trace_manager=manager,
    )

    run = runtime.run(
        "Custom trace",
        ["Step"],
    )

    assert manager.get(
        "run-1"
    ) is run.trace


def test_trace_snapshot():
    runtime = AgentRuntime(make_agent())

    run = runtime.run(
        "Snapshot",
        ["Step"],
    )

    snapshot = run.trace.snapshot()

    assert snapshot["run_id"] == "run-1"
    assert snapshot["event_count"] > 0


def test_memory_trace():
    memory = Memory()

    runtime = AgentRuntime(
        make_agent(),
        memory=memory,
    )

    run = runtime.run(
        "Memory trace",
        ["Step"],
    )

    events = [
        event.event_type
        for event in run.trace.events
    ]

    assert "memory_search_started" in events
    assert "memory_search_completed" in events
    assert "memory_saved" in events


def test_context_trace():
    runtime = AgentRuntime(make_agent())

    run = runtime.run(
        "Context trace",
        ["Step"],
    )

    events = [
        event.event_type
        for event in run.trace.events
    ]

    assert "context_started" in events
    assert "context_completed" in events


def test_execution_trace():
    runtime = AgentRuntime(make_agent())

    run = runtime.run(
        "Execution trace",
        ["Step"],
    )

    events = [
        event.event_type
        for event in run.trace.events
    ]

    assert "execution_started" in events
    assert "execution_completed" in events
