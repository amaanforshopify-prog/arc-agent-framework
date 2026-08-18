import json
from typing import Any

from .memory import ConversationMemory
from .models import Model
from .persistent_memory import PersistentMemory
from .tools import ToolRegistry
from .tracing import Tracer
from .types import AgentResult, Message


class AgentRuntime:
    """Executes an agent with tools, memory, persistence, and tracing."""

    def __init__(
        self,
        model: Model,
        tools: ToolRegistry | None = None,
        max_iterations: int = 10,
        tracer: Tracer | None = None,
        memory: ConversationMemory | None = None,
        persistent_memory: PersistentMemory | None = None,
    ):
        self.model = model
        self.tools = tools or ToolRegistry()
        self.max_iterations = max_iterations
        self.tracer = tracer or Tracer()

        self.memory = memory or ConversationMemory()
        self.persistent_memory = persistent_memory

    def _load_messages(self) -> list[Message]:
        """Load previous conversation from persistent storage or memory."""

        if self.persistent_memory is not None:
            return self.persistent_memory.get_messages()

        return self.memory.get_messages()

    def _save_message(self, message: Message) -> None:
        """Save a message to session memory and persistent storage."""

        self.memory.add(message)

        if self.persistent_memory is not None:
            self.persistent_memory.add(message)

    def run(
        self,
        system_prompt: str,
        user_input: str,
    ) -> AgentResult:
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

        messages = [
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

            assistant_message = response.choices[0].message

            # ---------------------------------
            # No tool call = final response
            # ---------------------------------
            if not assistant_message.tool_calls:
                output = assistant_message.content or ""

                final_message = Message(
                    role="assistant",
                    content=output,
                )

                self._save_message(final_message)

                messages.append(final_message)

                self.tracer.record(
                    "agent",
                    "Final response generated",
                    output=output,
                )

                return AgentResult(
                    output=output,
                    messages=messages,
                    metadata={
                        "iterations": iteration,
                        "memory_count": self.memory.count(),
                        "persistent_memory": (
                            self.persistent_memory is not None
                        ),
                        "trace": self.tracer.get_events(),
                    },
                )

            # ---------------------------------
            # LLM requested tools
            # ---------------------------------
            self.tracer.record(
                "llm",
                "Tool call requested",
                count=len(assistant_message.tool_calls),
            )

            assistant_message_for_history = Message(
                role="assistant",
                content=assistant_message.content or "",
            )

            messages.append(
                assistant_message_for_history
            )

            self._save_message(
                assistant_message_for_history
            )

            # ---------------------------------
            # Execute tool calls
            # ---------------------------------
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name

                self.tracer.record(
                    "tool",
                    "Executing tool",
                    tool=tool_name,
                    arguments=tool_call.function.arguments,
                )

                try:
                    arguments: dict[str, Any] = json.loads(
                        tool_call.function.arguments
                    )

                    result = self.tools.execute(
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
                        "error": str(exc),
                    }

                    self.tracer.record(
                        "error",
                        "Tool execution failed",
                        tool=tool_name,
                        error=str(exc),
                    )

                tool_message = Message(
                    role="tool",
                    content=json.dumps(
                        result,
                        default=str,
                    ),
                    tool_call_id=tool_call.id,
                    name=tool_name,
                )

                messages.append(tool_message)

                self._save_message(tool_message)

        self.tracer.record(
            "error",
            "Maximum iterations exceeded",
            max_iterations=self.max_iterations,
        )

        raise RuntimeError(
            f"Agent exceeded maximum iterations: "
            f"{self.max_iterations}"
        )