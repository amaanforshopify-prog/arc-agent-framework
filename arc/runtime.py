from __future__ import annotations

import json
from typing import Any

from .memory import ConversationMemory
from .models import Model
from .persistent_memory import PersistentMemory
from .tools import ToolRegistry
from .tracing import Tracer
from .types import AgentResult, Message


class AgentRuntime:
    """Executes an LLM agent with tools, memory, persistence and tracing."""

    def __init__(
        self,
        model: Model,
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
        self.tools = tools or ToolRegistry()
        self.max_iterations = max_iterations
        self.tracer = tracer or Tracer()
        self.memory = memory if memory is not None else ConversationMemory()
        self.persistent_memory = persistent_memory

    def _load_messages(self) -> list[Message]:
        if self.persistent_memory is not None:
            return self.persistent_memory.get_messages()

        return self.memory.get_messages()

    def _save_message(self, message: Message) -> None:
        self.memory.add(message)

        if self.persistent_memory is not None:
            self.persistent_memory.add(message)

    def _append_tool_calls(
        self,
        message: Any,
    ) -> list[Any]:
        return list(
            getattr(
                message,
                "tool_calls",
                None,
            )
            or []
        )

    def _assistant_message(
        self,
        message: Any,
    ) -> Message:
        return Message(
            role="assistant",
            content=getattr(
                message,
                "content",
                None,
            ) or "",
            tool_calls=self._append_tool_calls(
                message
            ),
        )

    def _execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ):
        """Execute a tool through the registry safety layer."""
        return self.tools.execute_safe(
            tool_name,
            **arguments,
        )

    def run(
        self,
        system_prompt: str,
        user_input: str,
    ) -> AgentResult:
        if not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string.")

        if not system_prompt.strip():
            raise ValueError(
                "system_prompt cannot be empty."
            )

        if not isinstance(user_input, str):
            raise TypeError("user_input must be a string.")

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

            response = self.model.generate(
                messages=messages,
                tools=self.tools.schemas(),
            )

            assistant = response.choices[0].message
            tool_calls = self._append_tool_calls(
                assistant
            )

            assistant_message = self._assistant_message(
                assistant
            )

            # Final answer
            if not tool_calls:
                output = (
                    getattr(
                        assistant,
                        "content",
                        None,
                    )
                    or ""
                )

                self._save_message(
                    assistant_message
                )

                messages.append(
                    assistant_message
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
                        "memory_count": self.memory.count(),
                        "persistent_memory": (
                            self.persistent_memory is not None
                        ),
                        "trace": self.tracer.get_events(),
                    },
                )

            # Tool request
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

                    safe_result = self._execute_tool(
                        tool_name,
                        arguments,
                    )

                    if not safe_result.success:
                        raise safe_result.error or RuntimeError(
                            f"Tool '{tool_name}' failed."
                        )

                    result = safe_result.output

                    self.tracer.record(
                        "tool",
                        "Tool execution completed",
                        tool=tool_name,
                        result=result,
                        attempts=safe_result.attempts,
                        duration=safe_result.duration,
                    )

                except Exception as exc:
                    result = {
                        "error": str(exc),
                    }

                    self.tracer.record(
                        "error",
                        "Tool execution failed",
                        tool=tool_name,
                        error=str(exc),
                    )

                if isinstance(result, str):
                    tool_content = result
                else:
                    tool_content = json.dumps(
                        result,
                        default=str,
                    )

                tool_message = Message(
                    role="tool",
                    content=tool_content,
                    tool_call_id=tool_call.id,
                    name=tool_name,
                )

                messages.append(
                    tool_message
                )

                self._save_message(
                    tool_message
                )

        self.tracer.record(
            "error",
            "Maximum iterations exceeded",
            max_iterations=self.max_iterations,
        )

        raise RuntimeError(
            f"Agent exceeded maximum iterations: "
            f"{self.max_iterations}"
        )




