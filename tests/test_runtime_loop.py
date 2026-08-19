from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from arc import ToolRegistry, tool
from arc.memory import ConversationMemory
from arc.runtime import AgentRuntime
from arc.tracing import Tracer


class DeterministicToolModel:

    def __init__(self):
        self.calls = 0
        self.seen_messages = []
        self.seen_tools = []

    def generate(
        self,
        messages,
        tools=None,
    ):
        self.calls += 1

        self.seen_messages.append(
            list(messages)
        )

        self.seen_tools.append(
            list(tools or [])
        )

        if self.calls == 1:
            tool_call = SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(
                    name="add",
                    arguments='{"a": 7, "b": 5}',
                ),
            )

            message = SimpleNamespace(
                content="",
                tool_calls=[
                    tool_call
                ],
            )

        else:
            message = SimpleNamespace(
                content="The answer is 12.",
                tool_calls=[],
            )

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=message
                )
            ]
        )


class MultiToolModel:

    def __init__(self):
        self.calls = 0

    def generate(
        self,
        messages,
        tools=None,
    ):
        self.calls += 1

        if self.calls == 1:
            calls = [
                SimpleNamespace(
                    id="call_add",
                    function=SimpleNamespace(
                        name="add",
                        arguments='{"a": 2, "b": 3}',
                    ),
                ),
                SimpleNamespace(
                    id="call_mul",
                    function=SimpleNamespace(
                        name="multiply",
                        arguments='{"a": 4, "b": 5}',
                    ),
                ),
            ]

            message = SimpleNamespace(
                content="",
                tool_calls=calls,
            )

        else:
            tool_results = [
                message.content
                for message in messages
                if getattr(
                    message,
                    "role",
                    None,
                ) == "tool"
            ]

            assert len(tool_results) == 2

            message = SimpleNamespace(
                content="Both tools completed.",
                tool_calls=[],
            )

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=message
                )
            ]
        )


@tool()
def add(
    a: int,
    b: int,
) -> int:
    return a + b


@tool()
def multiply(
    a: int,
    b: int,
) -> int:
    return a * b


def test_end_to_end_tool_call_loop():
    registry = ToolRegistry()
    registry.register(add)

    model = DeterministicToolModel()

    runtime = AgentRuntime(
        model=model,
        tools=registry,
        memory=ConversationMemory(),
        tracer=Tracer(enabled=False),
    )

    result = runtime.run(
        system_prompt=(
            "You are a tool-using assistant."
        ),
        user_input="What is 7 + 5?",
    )

    assert (
        result.output
        == "The answer is 12."
    )

    assert model.calls == 2

    assert len(
        model.seen_tools[0]
    ) == 1

    assert (
        model.seen_tools[0][0]["function"]["name"]
        == "add"
    )

    messages = result.messages

    assert messages[0].role == "system"
    assert messages[1].role == "user"
    assert messages[2].role == "assistant"
    assert len(
        messages[2].tool_calls
    ) == 1

    assert messages[3].role == "tool"
    assert messages[3].tool_call_id == "call_1"
    assert messages[3].content == "12"

    assert messages[4].role == "assistant"


def test_tool_result_reaches_second_model_call():
    registry = ToolRegistry()
    registry.register(add)

    model = DeterministicToolModel()

    runtime = AgentRuntime(
        model=model,
        tools=registry,
        tracer=Tracer(enabled=False),
    )

    result = runtime.run(
        system_prompt="Use tools.",
        user_input="Calculate.",
    )

    second_call = (
        model.seen_messages[1]
    )

    tool_messages = [
        message
        for message in second_call
        if getattr(
            message,
            "role",
            None,
        ) == "tool"
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0].content == "12"


def test_multi_tool_call_loop():
    registry = ToolRegistry()

    registry.register_many(
        add,
        multiply,
    )

    model = MultiToolModel()

    runtime = AgentRuntime(
        model=model,
        tools=registry,
        tracer=Tracer(enabled=False),
    )

    result = runtime.run(
        system_prompt="Use every required tool.",
        user_input="Calculate both.",
    )

    assert (
        result.output
        == "Both tools completed."
    )

    assert model.calls == 2

    tool_messages = [
        message
        for message in result.messages
        if getattr(
            message,
            "role",
            None,
        ) == "tool"
    ]

    assert len(tool_messages) == 2

    assert {
        message.tool_call_id
        for message in tool_messages
    } == {
        "call_add",
        "call_mul",
    }


def test_trace_records_tool_flow():
    registry = ToolRegistry()
    registry.register(add)

    model = DeterministicToolModel()

    tracer = Tracer(
        enabled=True
    )

    runtime = AgentRuntime(
        model=model,
        tools=registry,
        tracer=tracer,
    )

    result = runtime.run(
        system_prompt="Use tools.",
        user_input="Calculate.",
    )

    event_types = [
        event.event_type
        for event in result.metadata["trace"]
    ]

    assert "agent" in event_types
    assert "llm" in event_types
    assert "tool" in event_types


def test_tool_error_becomes_tool_result():
    @tool()
    def broken():
        raise ValueError("boom")

    class BrokenModel:

        def __init__(self):
            self.calls = 0

        def generate(
            self,
            messages,
            tools=None,
        ):
            self.calls += 1

            if self.calls == 1:
                tool_call = SimpleNamespace(
                    id="broken_call",
                    function=SimpleNamespace(
                        name="broken",
                        arguments="{}",
                    ),
                )

                message = SimpleNamespace(
                    content="",
                    tool_calls=[
                        tool_call
                    ],
                )

            else:
                tool_messages = [
                    message
                    for message in messages
                    if getattr(
                        message,
                        "role",
                        None,
                    ) == "tool"
                ]

                payload = json.loads(
                    tool_messages[0].content
                )

                assert "error" in payload

                message = SimpleNamespace(
                    content="Handled tool error.",
                    tool_calls=[],
                )

            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=message
                    )
                ]
            )

    registry = ToolRegistry()
    registry.register(broken)

    runtime = AgentRuntime(
        model=BrokenModel(),
        tools=registry,
        tracer=Tracer(enabled=False),
    )

    result = runtime.run(
        "Handle failures.",
        "Call broken.",
    )

    assert (
        result.output
        == "Handled tool error."
    )


def test_max_iterations():
    class EndlessModel:

        def generate(
            self,
            messages,
            tools=None,
        ):
            tool_call = SimpleNamespace(
                id="loop",
                function=SimpleNamespace(
                    name="add",
                    arguments='{"a": 1, "b": 1}',
                ),
            )

            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="",
                            tool_calls=[
                                tool_call
                            ],
                        )
                    )
                ]
            )

    registry = ToolRegistry()
    registry.register(add)

    runtime = AgentRuntime(
        model=EndlessModel(),
        tools=registry,
        max_iterations=2,
        tracer=Tracer(enabled=False),
    )

    with pytest.raises(
        RuntimeError,
        match="maximum iterations",
    ):
        runtime.run(
            "Use tools.",
            "Keep calculating.",
        )


def test_invalid_runtime_input():
    registry = ToolRegistry()
    registry.register(add)

    runtime = AgentRuntime(
        model=DeterministicToolModel(),
        tools=registry,
        tracer=Tracer(enabled=False),
    )

    with pytest.raises(ValueError):
        runtime.run(
            "",
            "hello",
        )

    with pytest.raises(ValueError):
        runtime.run(
            "system",
            "",
        )

