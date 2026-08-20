from types import SimpleNamespace

import pytest

from arc import (
    ToolApprovalError,
    ToolAwareRuntime,
    ToolPolicy,
    ToolRegistry,
    tool,
)
from arc.tracing import Tracer


@tool()
def add(
    a: int,
    b: int,
) -> int:
    return a + b


class ToolModel:
    def __init__(self):
        self.calls = 0

    def generate(
        self,
        messages,
        tools=None,
    ):
        self.calls += 1

        if self.calls == 1:
            call = SimpleNamespace(
                id="add_1",
                function=SimpleNamespace(
                    name="add",
                    arguments='{"a": 2, "b": 3}',
                ),
            )

            message = SimpleNamespace(
                content="",
                tool_calls=[call],
            )
        else:
            message = SimpleNamespace(
                content="Done.",
                tool_calls=[],
            )

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=message
                )
            ]
        )


def test_per_tool_policy():
    registry = ToolRegistry()
    registry.register(add)

    registry.set_policy(
        "add",
        ToolPolicy(
            risk_level="medium"
        ),
    )

    policy = registry.get_policy("add")

    assert policy.risk_level == "medium"
    assert policy.requires_approval is False


def test_approval_required():
    registry = ToolRegistry()
    registry.register(add)

    registry.set_policy(
        "add",
        ToolPolicy(
            risk_level="high",
            requires_approval=True,
        ),
    )

    blocked = registry.execute_safe(
        "add",
        2,
        3,
    )

    assert blocked.success is False
    assert isinstance(
        blocked.error,
        ToolApprovalError,
    )

    registry.approve("add")

    allowed = registry.execute_safe(
        "add",
        2,
        3,
    )

    assert allowed.success
    assert allowed.output == 5


def test_revoke_approval():
    registry = ToolRegistry()
    registry.register(add)

    registry.set_policy(
        "add",
        ToolPolicy(
            requires_approval=True
        ),
    )

    registry.approve("add")

    assert registry.is_approved(
        "add"
    )

    registry.revoke_approval(
        "add"
    )

    assert not registry.is_approved(
        "add"
    )


def test_audit_contains_security_metadata():
    registry = ToolRegistry()
    registry.register(add)

    registry.set_policy(
        "add",
        ToolPolicy(
            risk_level="critical",
            requires_approval=True,
        ),
    )

    registry.approve("add")

    result = registry.execute_safe(
        "add",
        5,
        7,
    )

    assert result.success
    assert result.output == 12

    record = registry.execution_history()[-1]

    assert record.metadata[
        "risk_level"
    ] == "critical"

    assert record.metadata[
        "requires_approval"
    ] is True

    assert record.metadata[
        "approved"
    ] is True


def test_tool_aware_runtime_uses_registry_policy():
    registry = ToolRegistry()
    registry.register(add)

    registry.set_policy(
        "add",
        ToolPolicy(
            risk_level="high",
            requires_approval=True,
        ),
    )

    runtime = ToolAwareRuntime(
        model=ToolModel(),
        tools=registry,
        tracer=Tracer(
            enabled=False
        ),
    )

    blocked = registry.execution_history()

    assert blocked == []

    try:
        runtime.run(
            "Use tools.",
            "Calculate 2 + 3.",
        )
    except RuntimeError:
        pass

    history = registry.execution_history()

    assert len(history) == 1
    assert history[0].success is False
    assert history[0].metadata[
        "risk_level"
    ] == "high"


def test_tool_aware_runtime_approval():
    registry = ToolRegistry()
    registry.register(add)

    runtime = ToolAwareRuntime(
        model=ToolModel(),
        tools=registry,
        tracer=Tracer(
            enabled=False
        ),
    )

    runtime.set_tool_policy(
        "add",
        ToolPolicy(
            risk_level="high",
            requires_approval=True,
        ),
    )

    runtime.approve_tool(
        "add"
    )

    result = runtime.run(
        "Use tools.",
        "Calculate 2 + 3.",
    )

    assert result.output == "Done."

    history = registry.execution_history()

    assert history[-1].success
    assert history[-1].metadata[
        "approved"
    ] is True
