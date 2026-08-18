from arc import (
    Agent,
    AgentRuntime,
    ContextManager,
    Memory,
    PlanningEngine,
)


def make_agent():
    engine = PlanningEngine()
    return Agent(
        name="RuntimeAgent",
        executor=engine,
    )


def test_runtime_creation():
    runtime = AgentRuntime(
        make_agent()
    )

    assert runtime.run_count() == 0


def test_runtime_run():
    runtime = AgentRuntime(
        make_agent()
    )

    run = runtime.run(
        "Simple task",
        ["Do the task"],
    )

    assert run.success
    assert run.run_id == 1


def test_runtime_result():
    runtime = AgentRuntime(
        make_agent()
    )

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
    runtime = AgentRuntime(
        make_agent()
    )

    runtime.run("One", ["First"])
    runtime.run("Two", ["Second"])
    runtime.run("Three", ["Third"])

    assert runtime.run_count() == 3


def test_last_run():
    runtime = AgentRuntime(
        make_agent()
    )

    runtime.run(
        "Test",
        ["Step"],
    )

    assert runtime.last_run() is not None
    assert runtime.last_run().run_id == 1


def test_clear_runs():
    runtime = AgentRuntime(
        make_agent()
    )

    runtime.run(
        "Test",
        ["Step"],
    )

    runtime.clear_runs()

    assert runtime.run_count() == 0
    assert runtime.last_run() is None


def test_snapshot():
    runtime = AgentRuntime(
        make_agent()
    )

    runtime.run(
        "Test",
        ["Step"],
    )

    snapshot = runtime.snapshot()

    assert snapshot["runs"] == 1
    assert snapshot["context_messages"] == 2
    assert snapshot["last_run"] == 1


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
    runtime = AgentRuntime(
        make_agent()
    )

    try:
        runtime.run("")
        assert False
    except ValueError:
        assert True


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
