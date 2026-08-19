from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .workflow_engine import (
    ConditionalStep,
    ParallelStep,
    WorkflowEngine,
    WorkflowEngineError,
    WorkflowState,
)
from .workflow_state import DurableWorkflowState


@dataclass
class CheckpointResult:
    success: bool
    outputs: list[Any]
    completed_steps: list[str]
    resumed: bool = False
    error: Exception | None = None


class CheckpointedWorkflowEngine(WorkflowEngine):
    """Workflow engine with durable checkpoint/resume support."""

    def execute_checkpointed(
        self,
        operations: list[
            ConditionalStep | ParallelStep
        ],
        state: DurableWorkflowState,
        stop_on_error: bool = True,
    ) -> CheckpointResult:

        if not isinstance(
            state,
            DurableWorkflowState,
        ):
            raise TypeError(
                "state must be DurableWorkflowState."
            )

        outputs: list[Any] = []
        resumed = bool(state.completed_steps)

        for index, operation in enumerate(
            operations,
            start=1,
        ):
            step_name = getattr(
                operation,
                "name",
                f"step-{index}",
            )

            if state.is_completed(step_name):
                outputs.append(
                    state.get(
                        f"result:{step_name}"
                    )
                )
                continue

            state.begin_step(step_name)
            state.save()

            try:
                if isinstance(
                    operation,
                    ConditionalStep,
                ):
                    result = self.run_condition(
                        operation,
                        WorkflowState(
                            state.data
                        ),
                    )

                    if not result.success:
                        raise WorkflowEngineError(
                            f"Workflow step "
                            f"'{step_name}' failed."
                        ) from result.error

                    output = result.result

                elif isinstance(
                    operation,
                    ParallelStep,
                ):
                    result_map = self.run_parallel(
                        operation,
                        WorkflowState(
                            state.data
                        ),
                        stop_on_error=stop_on_error,
                    )

                    failed = [
                        result
                        for result
                        in result_map.values()
                        if not result.success
                    ]

                    if failed and stop_on_error:
                        raise WorkflowEngineError(
                            f"Parallel workflow "
                            f"'{step_name}' failed."
                        ) from failed[0].error

                    output = {
                        name: result.result
                        for name, result
                        in result_map.items()
                    }

                else:
                    raise TypeError(
                        "Unsupported workflow operation."
                    )

                state.set(
                    f"result:{step_name}",
                    output,
                )

                state.mark_completed(
                    step_name
                )

                state.save()

                outputs.append(output)

            except Exception as exc:
                state.metadata["last_error"] = str(
                    exc
                )
                state.metadata[
                    "failed_step"
                ] = step_name
                state.save()

                if stop_on_error:
                    return CheckpointResult(
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
        state.metadata.pop(
            "last_error",
            None,
        )
        state.metadata.pop(
            "failed_step",
            None,
        )
        state.save()

        return CheckpointResult(
            success=True,
            outputs=outputs,
            completed_steps=list(
                state.completed_steps
            ),
            resumed=resumed,
        )
