from .memory import ConversationMemory
from .models import Model
from .persistent_memory import PersistentMemory
from .runtime import AgentRuntime
from .tools import ToolRegistry
from .tracing import Tracer
from .types import AgentResult


class Agent:
    """High-level AI agent."""

    def __init__(
        self,
        name: str,
        model: Model,
        system_prompt: str = "You are a helpful AI agent.",
        tools: ToolRegistry | None = None,
        tracer: Tracer | None = None,
        memory: ConversationMemory | None = None,
        persistent_memory: PersistentMemory | None = None,
    ):
        self.name = name
        self.system_prompt = system_prompt

        self.runtime = AgentRuntime(
            model=model,
            tools=tools,
            tracer=tracer,
            memory=memory,
            persistent_memory=persistent_memory,
        )

    def run(self, task: str) -> AgentResult:
        return self.runtime.run(
            system_prompt=self.system_prompt,
            user_input=task,
        )