from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from arc import ToolRegistry, tool
from arc.memory import ConversationMemory
from arc.persistent_memory import PersistentMemory
from arc.runtime import AgentRuntime
from arc.tracing import Tracer


@tool()
def ping(message: str) -> str:
    return f"pong: {message}"


class FinalModel:
    def __init__(self):
        self.calls = 0

    def generate(self, messages, tools=None):
        self.calls += 1

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Final response.",
                        tool_calls=[],
                    )
                )
            ]
        )


class ToolThenFinalModel:
    def __init__(self):
        self.calls = 0

    def generate(self, messages, tools=None):
        self.calls += 1

        if self.calls == 1:
            call = SimpleNamespace(
                id="ping_1",
                function=SimpleNamespace(
                    name="ping",
                    arguments='{"message":"ARC"}',
                ),
            )

            message = SimpleNamespace(
                content="",
                tool_calls=[call],
            )

        else:
            message = SimpleNamespace(
                content="Tool completed.",
                tool_calls=[],
            )

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=message
                )
            ]
        )


def test_runtime_persists_conversation(tmp_path):
    db_path = Path(tmp_path) / "runtime.db"

    tools = ToolRegistry()
    tools.register(ping)

    memory = ConversationMemory()
    persistent = PersistentMemory(
        str(db_path)
    )

    runtime = AgentRuntime(
        model=FinalModel(),
        tools=tools,
        memory=memory,
        persistent_memory=persistent,
        tracer=Tracer(enabled=False),
    )

    result = runtime.run(
        "You are ARC.",
        "Hello ARC.",
    )

    assert result.output == "Final response."

    assert memory.count() == 2
    assert persistent.count() == 2

    persistent.close()


def test_runtime_restores_persisted_messages(
    tmp_path,
):
    db_path = Path(tmp_path) / "restore.db"

    first_persistent = PersistentMemory(
        str(db_path)
    )

    first_memory = ConversationMemory()

    first = AgentRuntime(
        model=FinalModel(),
        tools=ToolRegistry(),
        memory=first_memory,
        persistent_memory=first_persistent,
        tracer=Tracer(enabled=False),
    )

    first.run(
        "You are ARC.",
        "Remember this session.",
    )

    first_persistent.close()

    second_persistent = PersistentMemory(
        str(db_path)
    )

    second_memory = ConversationMemory()

    second = AgentRuntime(
        model=FinalModel(),
        tools=ToolRegistry(),
        memory=second_memory,
        persistent_memory=second_persistent,
        tracer=Tracer(enabled=False),
    )

    result = second.run(
        "Continue.",
        "Continue the session.",
    )

    assert result.output == "Final response."

    messages = second_persistent.get_messages()

    assert len(messages) == 4
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"
    assert messages[2].role == "user"
    assert messages[3].role == "assistant"

    second_persistent.close()


def test_runtime_persists_tool_call_flow(
    tmp_path,
):
    db_path = Path(tmp_path) / "tool_runtime.db"

    tools = ToolRegistry()
    tools.register(ping)

    memory = ConversationMemory()
    persistent = PersistentMemory(
        str(db_path)
    )

    model = ToolThenFinalModel()

    runtime = AgentRuntime(
        model=model,
        tools=tools,
        memory=memory,
        persistent_memory=persistent,
        tracer=Tracer(enabled=False),
    )

    result = runtime.run(
        "Use tools.",
        "Ping ARC.",
    )

    assert result.output == "Tool completed."

    persisted = persistent.get_messages()

    roles = [
        message.role
        for message in persisted
    ]

    assert roles == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]

    assert persisted[2].tool_call_id == "ping_1"
    assert persisted[2].name == "ping"
    assert persisted[2].content == "pong: ARC"

    persistent.close()


def test_trace_enabled_collects_events():
    tools = ToolRegistry()
    tools.register(ping)

    tracer = Tracer(enabled=True)

    runtime = AgentRuntime(
        model=ToolThenFinalModel(),
        tools=tools,
        tracer=tracer,
    )

    result = runtime.run(
        "Use tools.",
        "Ping.",
    )

    event_types = [
        event.event_type
        for event in result.metadata["trace"]
    ]

    assert "agent" in event_types
    assert "llm" in event_types
    assert "tool" in event_types


def test_trace_disabled_stays_silent():
    tools = ToolRegistry()
    tools.register(ping)

    tracer = Tracer(enabled=False)

    runtime = AgentRuntime(
        model=FinalModel(),
        tools=tools,
        tracer=tracer,
    )

    result = runtime.run(
        "No tracing.",
        "Hello.",
    )

    assert result.metadata["trace"] == []
