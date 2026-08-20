from __future__ import annotations

from typing import Any

from .runtime import AgentRuntime
from .tool_intelligence import ToolPolicy
from .tools import ToolRegistry


class ToolAwareRuntime(AgentRuntime):
    """
    AgentRuntime variant using the tool security layer.

    Supports:
    - global tool policy
    - per-tool registry policies
    - approval/revocation
    - safe execution
    """

    def __init__(
        self,
        model,
        tools: ToolRegistry | None = None,
        max_iterations: int = 10,
        tracer=None,
        memory=None,
        persistent_memory=None,
        tool_policy: ToolPolicy | None = None,
    ):
        super().__init__(
            model=model,
            tools=tools,
            max_iterations=max_iterations,
            tracer=tracer,
            memory=memory,
            persistent_memory=persistent_memory,
        )

        self.tool_policy = tool_policy

    def _execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ):
        return self.tools.execute_safe(
            tool_name,
            policy=self.tool_policy,
            **arguments,
        )

    def approve_tool(
        self,
        tool_name: str,
    ) -> None:
        self.tools.approve(tool_name)

    def revoke_tool(
        self,
        tool_name: str,
    ) -> None:
        self.tools.revoke_approval(tool_name)

    def set_tool_policy(
        self,
        tool_name: str,
        policy: ToolPolicy,
    ) -> None:
        self.tools.set_policy(
            tool_name,
            policy,
        )
