from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .checkpoint_engine import (
    CheckpointResult,
    CheckpointedWorkflowEngine,
)
from .models import Model
from .runtime import AgentRuntime
from .tools import ToolRegistry
from .workflow_engine import (
    ConditionalStep,
    ParallelStep,
)
from .workflow_state import DurableWorkflowState


class AgentWorkflowError(Exception):
    """Raised when an agent workflow fails."""


@dataclass
class AgentWorkflowStep:
    name: str
    system_prompt: str
    user_input: str
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class AgentWorkflowResult:
    success: bool
    outputs: list[Any]
    completed_steps: list[str]
    resumed: bool = False
    error: Exception | None = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class AgentWorkflowEngine:
    """
    Connects ARC AgentRuntime to durable workflow execution.

    Supports:
    - LLM agent steps
    - tool-enabled agent execution
    - durable checkpoints
    - resume after failure
    - shared workflow state
    """

    def __init__(
        self,
        model: Model,
        tools: ToolRegistry | None = None,
        max_iterations: int = 10,
    ) -> None:
        self.model = model
        self.tools = (
            tools
            if tools is not None
            else ToolRegistry()
        )
        self.max_iterations = max_iterations

    def create_runtime(self) -> AgentRuntime:
        return AgentRuntime(
            model=self.model,
            tools=self.tools,
            max_iterations=self.max_iterations,
        )

    def execute_step(
        self,
        step: AgentWorkflowStep,
        state: DurableWorkflowState,
    ) -> Any:
        runtime = self.create_runtime()

        result = runtime.run(
            system_prompt=step.system_prompt,
            user_input=step.user_input,
        )

        state.set(
            f"agent_output:{step.name}",
            result.output,
        )

        state.set(
            f"agent_metadata:{step.name}",
            result.metadata,
        )

        if step.metadata:
            state.update(
                {
                    f"{step.name}:{key}": value
                    for key, value
                    in step.metadata.items()
                }
            )

        return result.output

    def execute_checkpointed(
        self,
        steps: list[AgentWorkflowStep],
        state: DurableWorkflowState,
        stop_on_error: bool = True,
    ) -> AgentWorkflowResult:

        outputs: list[Any] = []
        resumed = bool(
            state.completed_steps
        )

        for step in steps:
            if state.is_completed(step.name):
                outputs.append(
                    state.get(
                        f"result:{step.name}"
                    )
                )
                continue

            state.begin_step(step.name)
            state.save()

            try:
                output = self.execute_step(
                    step,
                    state,
                )

                state.set(
                    f"result:{step.name}",
                    output,
                )

                state.mark_completed(
                    step.name
                )

                state.metadata.pop(
                    "last_error",
                    None,
                )
                state.metadata.pop(
                    "failed_step",
                    None,
                )

                state.save()
                outputs.append(output)

            except Exception as exc:
                state.metadata[
                    "last_error"
                ] = str(exc)

                state.metadata[
                    "failed_step"
                ] = step.name

                state.save()

                if stop_on_error:
                    return AgentWorkflowResult(
                        success=False,
                        outputs=outputs,
                        completed_steps=list(
                            state.completed_steps
                        ),
                        resumed=resumed,
                        error=exc,
                    )

                outputs.append(None)

        state.current_step = None
        state.save()

        return AgentWorkflowResult(
            success=True,
            outputs=outputs,
            completed_steps=list(
                state.completed_steps
            ),
            resumed=resumed,
            metadata={
                "workflow_id": state.workflow_id,
                "steps": len(steps),
            },
        )
