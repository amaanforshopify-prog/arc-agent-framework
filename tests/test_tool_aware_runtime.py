from types import SimpleNamespace

from arc import (
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

    def generate(self, messages, tools=None):
        self.calls += 1

        if self.calls == 1:
            call = SimpleNamespace(
                id="add_1",
                function=SimpleNamespace(
                    name="add",
                    arguments='{"a": 7, "b": 8}',
                ),
            )

            message = SimpleNamespace(
                content="",
                tool_calls=[call],
            )

        else:
            message = SimpleNamespace(
                content="The result is 15.",
                tool_calls=[],
            )

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=message
                )
            ]
        )


def test_tool_aware_runtime_success():
    registry = ToolRegistry()
    registry.register(add)

    runtime = ToolAwareRuntime(
        model=ToolModel(),
        tools=registry,
        tracer=Tracer(enabled=False),
    )

    result = runtime.run(
        "Use tools.",
        "Calculate 7 + 8.",
    )

    assert result.output == "The result is 15."

    history = registry.execution_history()

    assert len(history) == 1
    assert history[0].tool_name == "add"
    assert history[0].success


def test_tool_aware_runtime_denied_tool():
    registry = ToolRegistry()
    registry.register(add)

    runtime = ToolAwareRuntime(
        model=ToolModel(),
        tools=registry,
        tool_policy=ToolPolicy(
            allowed=False
        ),
        tracer=Tracer(enabled=False),
    )

    try:
        runtime.run(
            "Use tools.",
            "Calculate.",
        )
    except RuntimeError:
        pass

    history = registry.execution_history()

    assert len(history) == 1
    assert history[0].success is False
