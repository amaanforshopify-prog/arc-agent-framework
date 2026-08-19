from __future__ import annotations

import asyncio
import json
from typing import Any

from .memory import ConversationMemory
from .models import Model
from .persistent_memory import PersistentMemory
from .tools import ToolRegistry
from .tracing import Tracer
from .types import AgentResult, Message


class AsyncAgentRuntime:
    """
    Async counterpart to AgentRuntime.

    Runs synchronous and asynchronous models/tools through
    the same iterative tool-calling architecture.
    """

    def __init__(
        self,
        model: Any,
        tools: ToolRegistry | None = None,
        max_iterations: int = 10,
        tracer: Tracer | None = None,
        memory: ConversationMemory | None = None,
        persistent_memory: PersistentMemory | None = None,
    ):
        if max_iterations <= 0:
            raise ValueError(
                "max_iterations must be greater than 0"
            )

        self.model = model
        self.tools = (
            tools
            if tools is not None
            else ToolRegistry()
        )
        self.max_iterations = max_iterations
        self.tracer = (
            tracer
            if tracer is not None
            else Tracer()
        )
        self.memory = (
            memory
            if memory is not None
            else ConversationMemory()
        )
        self.persistent_memory = persistent_memory

    def _load_messages(self) -> list[Message]:
        if self.persistent_memory is not None:
            return self.persistent_memory.get_messages()

        return self.memory.get_messages()

    def _save_message(self, message: Message) -> None:
        self.memory.add(message)

        if self.persistent_memory is not None:
            self.persistent_memory.add(message)

    async def run(
        self,
        system_prompt: str,
        user_input: str,
    ) -> AgentResult:
        if not isinstance(system_prompt, str):
            raise TypeError(
                "system_prompt must be a string."
            )

        if not system_prompt.strip():
            raise ValueError(
                "system_prompt cannot be empty."
            )

        if not isinstance(user_input, str):
            raise TypeError(
                "user_input must be a string."
            )

        if not user_input.strip():
            raise ValueError(
                "user_input cannot be empty."
            )

        self.tracer.clear()

        self.tracer.record(
            "agent",
            "Task received",
            input=user_input,
        )

        user_message = Message(
            role="user",
            content=user_input,
        )

        self._save_message(user_message)

        messages: list[Message] = [
            Message(
                role="system",
                content=system_prompt,
            ),
            *self._load_messages(),
        ]

        for iteration in range(
            1,
            self.max_iterations + 1,
        ):
            self.tracer.record(
                "llm",
                "Generating response",
                iteration=iteration,
            )

            generate_async = getattr(
                self.model,
                "generate_async",
                None,
            )

            if callable(generate_async):
                response = await generate_async(
                    messages=messages,
                    tools=self.tools.schemas(),
                )
            else:
                response = await asyncio.to_thread(
                    self.model.generate,
                    messages=messages,
                    tools=self.tools.schemas(),
                )

            assistant = response.choices[0].message

            tool_calls = list(
                getattr(
                    assistant,
                    "tool_calls",
                    None,
                )
                or []
            )

            assistant_message = Message(
                role="assistant",
                content=(
                    getattr(
                        assistant,
                        "content",
                        None,
                    )
                    or ""
                ),
                tool_calls=tool_calls,
            )

            if not tool_calls:
                self._save_message(
                    assistant_message
                )

                messages.append(
                    assistant_message
                )

                output = (
                    assistant_message.content
                    or ""
                )

                self.tracer.record(
                    "agent",
                    "Final response generated",
                    output=output,
                )

                return AgentResult(
                    output=output,
                    messages=list(messages),
                    metadata={
                        "iterations": iteration,
                        "memory_count": (
                            self.memory.count()
                        ),
                        "persistent_memory": (
                            self.persistent_memory
                            is not None
                        ),
                        "trace": (
                            self.tracer.get_events()
                        ),
                    },
                )

            self.tracer.record(
                "llm",
                "Tool call requested",
                count=len(tool_calls),
            )

            self._save_message(
                assistant_message
            )

            messages.append(
                assistant_message
            )

            for tool_call in tool_calls:
                tool_name = (
                    tool_call.function.name
                )

                raw_arguments = (
                    tool_call.function.arguments
                )

                self.tracer.record(
                    "tool",
                    "Executing tool",
                    tool=tool_name,
                    arguments=raw_arguments,
                )

                try:
                    arguments = json.loads(
                        raw_arguments
                    )

                    if not isinstance(
                        arguments,
                        dict,
                    ):
                        raise ValueError(
                            "Tool arguments must decode to an object."
                        )

                    result = await self.tools.execute_async(
                        tool_name,
                        **arguments,
                    )

                    self.tracer.record(
                        "tool",
                        "Tool execution completed",
                        tool=tool_name,
                        result=result,
                    )

                except Exception as exc:
                    result = {
                        "error": str(exc)
                    }

                    self.tracer.record(
                        "error",
                        "Tool execution failed",
                        tool=tool_name,
                        error=str(exc),
                    )

                if isinstance(
                    result,
                    str,
                ):
                    content = result
                else:
                    content = json.dumps(
                        result,
                        default=str,
                    )

                tool_message = Message(
                    role="tool",
                    content=content,
                    tool_call_id=tool_call.id,
                    name=tool_name,
                )

                messages.append(
                    tool_message
                )

                self._save_message(
                    tool_message
                )

        raise RuntimeError(
            "Agent exceeded maximum iterations: "
            f"{self.max_iterations}"
        )
