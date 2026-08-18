import pytest

from arc import (
    ContextError,
    ContextManager,
    Plan,
    Planner,
    PlannerError,
    PromptBuilder,
)


# =========================
# CONTEXT
# =========================

def test_context_creation():
    context = ContextManager()

    assert context.count() == 0


def test_context_add():
    context = ContextManager()

    message = context.add(
        "user",
        "Hello ARC",
    )

    assert message.role == "user"
    assert message.content == "Hello ARC"


def test_context_system():
    context = ContextManager()

    context.system("You are ARC.")

    assert context.get("system")[0].content == (
        "You are ARC."
    )


def test_context_user():
    context = ContextManager()

    context.user("Hello")

    assert context.last().role == "user"


def test_context_assistant():
    context = ContextManager()

    context.assistant("Hi")

    assert context.last().role == "assistant"


def test_context_tool():
    context = ContextManager()

    context.tool("Tool result")

    assert context.last().role == "tool"


def test_context_invalid_role():
    context = ContextManager()

    with pytest.raises(ContextError):
        context.add(
            "invalid",
            "test",
        )


def test_context_empty_content():
    context = ContextManager()

    with pytest.raises(ValueError):
        context.user("")


def test_context_get_role():
    context = ContextManager()

    context.system("System")
    context.user("User")
    context.user("Another")

    assert len(
        context.get("user")
    ) == 2


def test_context_last():
    context = ContextManager()

    context.user("First")
    context.assistant("Second")

    assert (
        context.last().content
        == "Second"
    )


def test_context_clear():
    context = ContextManager()

    context.user("Hello")
    context.clear()

    assert context.count() == 0


def test_context_metadata():
    context = ContextManager()

    context.set_metadata(
        "agent",
        "ARC",
    )

    assert (
        context.get_metadata("agent")
        == "ARC"
    )


def test_context_snapshot():
    context = ContextManager()

    context.user("Hello")

    snapshot = context.snapshot()

    assert snapshot["count"] == 1
    assert len(snapshot["messages"]) == 1


def test_context_limit():
    context = ContextManager(
        max_messages=3
    )

    context.user("one")
    context.user("two")
    context.user("three")
    context.user("four")

    assert context.count() == 3
    assert context.last().content == "four"


def test_context_system_survives_trim():
    context = ContextManager(
        max_messages=3
    )

    context.system("System")
    context.user("one")
    context.user("two")
    context.user("three")

    assert context.count() == 3
    assert context.get("system")[0].content == (
        "System"
    )


# =========================
# PROMPT
# =========================

def test_prompt_builder():
    prompt = PromptBuilder(
        "You are ARC."
    )

    prompt.add_user(
        "What can you do?"
    )

    result = prompt.build()

    assert "[SYSTEM]" in result
    assert "[USER]" in result


def test_prompt_messages():
    prompt = PromptBuilder()

    prompt.add_user("Hello")

    messages = prompt.messages()

    assert messages[0]["role"] == "user"


def test_prompt_count():
    prompt = PromptBuilder()

    prompt.add_user("Hello")
    prompt.add_assistant("Hi")

    assert prompt.count() == 2


def test_prompt_clear():
    prompt = PromptBuilder()

    prompt.add_user("Hello")
    prompt.clear()

    assert prompt.count() == 0


# =========================
# PLANNER
# =========================

def test_plan_creation():
    planner = Planner()

    plan = planner.create(
        "Build ARC"
    )

    assert isinstance(
        plan,
        Plan,
    )


def test_plan_empty_goal():
    planner = Planner()

    with pytest.raises(PlannerError):
        planner.create("")


def test_plan_steps():
    planner = Planner()

    plan = planner.create(
        "Build ARC",
        [
            "Create tools",
            "Create agent",
            "Test system",
        ],
    )

    assert len(plan.steps) == 3
    assert plan.steps[0].id == 1
    assert plan.steps[1].id == 2
    assert plan.steps[2].id == 3


def test_plan_next_step():
    planner = Planner()

    plan = planner.create(
        "Build",
        ["Step one", "Step two"],
    )

    assert (
        plan.next_step().description
        == "Step one"
    )


def test_plan_start():
    planner = Planner()

    plan = planner.create(
        "Build",
        ["Step one"],
    )

    planner.start(plan)

    assert plan.status == "running"


def test_plan_complete_step():
    planner = Planner()

    plan = planner.create(
        "Build",
        ["Step one"],
    )

    planner.start(plan)

    planner.complete_step(
        plan,
        1,
        "done",
    )

    assert plan.steps[0].status == "completed"
    assert plan.steps[0].result == "done"
    assert plan.status == "completed"


def test_plan_progress():
    planner = Planner()

    plan = planner.create(
        "Build",
        [
            "One",
            "Two",
        ],
    )

    planner.start(plan)

    planner.complete_step(
        plan,
        1,
    )

    assert plan.progress() == 0.5


def test_plan_failure():
    planner = Planner()

    plan = planner.create(
        "Build",
        ["One"],
    )

    planner.start(plan)

    error = RuntimeError("failed")

    planner.fail_step(
        plan,
        1,
        error,
    )

    assert plan.failed()
    assert plan.status == "failed"


def test_plan_unknown_step():
    planner = Planner()

    plan = planner.create(
        "Build",
        ["One"],
    )

    with pytest.raises(PlannerError):
        planner.complete_step(
            plan,
            99,
        )


def test_plan_dict():
    planner = Planner()

    plan = planner.create(
        "Build",
        ["One"],
    )

    data = plan.to_dict()

    assert data["goal"] == "Build"
    assert len(data["steps"]) == 1


def test_planner_add_step():
    planner = Planner()

    plan = planner.create(
        "Build"
    )

    step = planner.add_step(
        plan,
        "Use calculator",
        "calculator",
        10,
        20,
    )

    assert step.tool == "calculator"
    assert step.args == (10, 20)
