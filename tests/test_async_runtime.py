from types import SimpleNamespace

import pytest

from arc import AsyncAgentRuntime, ToolRegistry, tool
from arc.memory import ConversationMemory
from arc.tracing import Tracer


@tool()
async def async_add(
    a: int,
    b: int,
) -> int:
    return a + b


class AsyncModel:
    def __init__(self):
        self.calls = 0

    async def generate_async(
        self,
        messages,
        tools=None,
    ):
        self.calls += 1

        if self.calls == 1:
            call = SimpleNamespace(
                id="async_add_1",
                function=SimpleNamespace(
                    name="async_add",
                    arguments='{"a": 10, "b": 5}',
                ),
            )

            message = SimpleNamespace(
                content="",
                tool_calls=[call],
            )

        else:
            message = SimpleNamespace(
                content="Async result is 15.",
                tool_calls=[],
            )

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=message
                )
            ]
        )


@pytest.mark.asyncio
async def test_async_runtime_tool_loop():
    registry = ToolRegistry()
    registry.register(async_add)

    model = AsyncModel()

    runtime = AsyncAgentRuntime(
        model=model,
        tools=registry,
        memory=ConversationMemory(),
        tracer=Tracer(enabled=False),
    )

    result = await runtime.run(
        "Use tools.",
        "Calculate 10 + 5.",
    )

    assert result.output == "Async result is 15."
    assert model.calls == 2

    tool_messages = [
        message
        for message in result.messages
        if message.role == "tool"
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0].content == "15"


@pytest.mark.asyncio
async def test_async_runtime_memory_is_preserved():
    registry = ToolRegistry()

    model = AsyncModel()
    memory = ConversationMemory()

    runtime = AsyncAgentRuntime(
        model=model,
        tools=registry,
        memory=memory,
        tracer=Tracer(enabled=False),
    )

    class FinalAsyncModel:
        async def generate_async(
            self,
            messages,
            tools=None,
        ):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="done",
                            tool_calls=[],
                        )
                    )
                ]
            )

    runtime.model = FinalAsyncModel()

    result = await runtime.run(
        "System.",
        "Hello.",
    )

    assert result.output == "done"
    assert memory.count() == 2
