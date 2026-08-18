import pytest

from arc import (
    Plan,
    Planner,
    PlanningEngine,
    PlanningEngineError,
    Tool,
    ToolRegistry,
)


def add(a, b):
    return a + b


def multiply(a, b):
    return a * b


def fail_tool():
    raise RuntimeError("tool failed")


def test_engine_creation():
    engine = PlanningEngine()

    assert engine.planner is not None
    assert engine.tools is not None


def test_engine_create_plan():
    engine = PlanningEngine()

    plan = engine.create_plan(
        "Calculate",
        ["Add numbers"],
    )

    assert isinstance(plan, Plan)
    assert plan.goal == "Calculate"


def test_execute_simple_plan():
    engine = PlanningEngine()

    plan = engine.create_plan(
        "Simple task",
        [
            "First step",
            "Second step",
        ],
    )

    result = engine.execute(plan)

    assert result.success
    assert result.completed_steps == 2


def test_execute_returns_result():
    engine = PlanningEngine()

    plan = engine.create_plan(
        "Task",
        ["Do something"],
    )

    result = engine.execute(plan)

    assert result.final_result == "Do something"


def test_execute_plan_status():
    engine = PlanningEngine()

    plan = engine.create_plan(
        "Task",
        ["Step"],
    )

    engine.execute(plan)

    assert plan.status == "completed"


def test_tool_step():
    registry = ToolRegistry()

    registry.register(
        Tool("add", "Add two numbers", add)
    )

    engine = PlanningEngine(
        tools=registry
    )

    plan = engine.create_plan(
        "Add numbers"
    )

    engine.add_tool_step(
        plan,
        "Add 2 + 3",
        "add",
        2,
        3,
    )

    result = engine.execute(plan)

    assert result.success
    assert result.final_result == 5


def test_multiple_tool_steps():
    registry = ToolRegistry()

    registry.register(
        Tool("add", "Add two numbers", add)
    )

    registry.register(
        Tool("multiply", "Multiply two numbers", multiply)
    )

    engine = PlanningEngine(
        tools=registry
    )

    plan = engine.create_plan(
        "Math"
    )

    engine.add_tool_step(
        plan,
        "Add",
        "add",
        2,
        3,
    )

    engine.add_tool_step(
        plan,
        "Multiply",
        "multiply",
        4,
        5,
    )

    result = engine.execute(plan)

    assert result.success
    assert result.completed_steps == 2
    assert result.results[0].result == 5
    assert result.results[1].result == 20


def test_tool_failure():
    registry = ToolRegistry()

    registry.register(
        Tool("fail_tool", "Always fails", fail_tool)
    )

    engine = PlanningEngine(
        tools=registry
    )

    plan = engine.create_plan(
        "Failure"
    )

    engine.add_tool_step(
        plan,
        "Fail",
        "fail_tool",
    )

    result = engine.execute(plan)

    assert not result.success
    assert result.failed_steps == 1


def test_stop_on_error():
    registry = ToolRegistry()

    registry.register(
        Tool("fail_tool", "Always fails", fail_tool)
    )

    registry.register(
        Tool("add", "Add two numbers", add)
    )

    engine = PlanningEngine(
        tools=registry
    )

    plan = engine.create_plan(
        "Failure"
    )

    engine.add_tool_step(
        plan,
        "Fail",
        "fail_tool",
    )

    engine.add_tool_step(
        plan,
        "Add",
        "add",
        1,
        2,
    )

    result = engine.execute(
        plan,
        stop_on_error=True,
    )

    assert not result.success
    assert len(result.results) == 1


def test_continue_after_error():
    registry = ToolRegistry()

    registry.register(
        Tool("fail_tool", "Always fails", fail_tool)
    )

    registry.register(
        Tool("add", "Add two numbers", add)
    )

    engine = PlanningEngine(
        tools=registry
    )

    plan = engine.create_plan(
        "Failure"
    )

    engine.add_tool_step(
        plan,
        "Fail",
        "fail_tool",
    )

    engine.add_tool_step(
        plan,
        "Add",
        "add",
        1,
        2,
    )

    result = engine.execute(
        plan,
        stop_on_error=False,
    )

    assert not result.success
    assert len(result.results) == 2
    assert result.results[1].result == 3


def test_step_result_success():
    registry = ToolRegistry()

    registry.register(
        Tool("add", "Add two numbers", add)
    )

    engine = PlanningEngine(
        tools=registry
    )

    plan = engine.create_plan("Math")

    engine.add_tool_step(
        plan,
        "Add",
        "add",
        10,
        20,
    )

    result = engine.execute(plan)

    step = result.results[0]

    assert step.step_id == 1
    assert step.success
    assert step.result == 30


def test_step_result_error():
    registry = ToolRegistry()

    registry.register(
        Tool("fail_tool", "Always fails", fail_tool)
    )

    engine = PlanningEngine(
        tools=registry
    )

    plan = engine.create_plan("Fail")

    engine.add_tool_step(
        plan,
        "Fail",
        "fail_tool",
    )

    result = engine.execute(plan)

    step = result.results[0]

    assert not step.success
    from arc import ToolError; assert isinstance(step.error, ToolError)


def test_empty_plan():
    engine = PlanningEngine()

    plan = engine.create_plan(
        "Empty"
    )

    with pytest.raises(
        PlanningEngineError
    ):
        engine.execute(plan)


def test_invalid_plan():
    engine = PlanningEngine()

    with pytest.raises(TypeError):
        engine.execute("invalid")


def test_add_callable_step():
    engine = PlanningEngine()

    plan = engine.create_plan(
        "Callable"
    )

    engine.add_callable_step(
        plan,
        "Calculate",
        add,
        5,
        7,
    )

    result = engine.execute(plan)

    assert result.success
    assert result.final_result == 12


def test_result_goal():
    engine = PlanningEngine()

    plan = engine.create_plan(
        "My goal",
        ["Step"],
    )

    result = engine.execute(plan)

    assert result.goal == "My goal"


def test_completed_step_count():
    engine = PlanningEngine()

    plan = engine.create_plan(
        "Task",
        ["One", "Two", "Three"],
    )

    result = engine.execute(plan)

    assert result.completed_steps == 3
    assert result.failed_steps == 0


def test_plan_progress_after_execution():
    engine = PlanningEngine()

    plan = engine.create_plan(
        "Task",
        ["One", "Two"],
    )

    engine.execute(plan)

    assert plan.progress() == 1.0


