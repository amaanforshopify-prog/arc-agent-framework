from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable


class WorkflowEngineError(Exception):
    """Base workflow engine error."""


@dataclass
class WorkflowState:
    """Mutable state shared across workflow steps."""

    data: dict[str, Any] = field(
        default_factory=dict
    )

    def set(
        self,
        key: str,
        value: Any,
    ) -> None:
        if not isinstance(key, str) or not key.strip():
            raise ValueError(
                "state key cannot be empty."
            )

        self.data[key] = value

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        return self.data.get(
            key,
            default,
        )

    def has(self, key: str) -> bool:
        return key in self.data

    def update(
        self,
        values: dict[str, Any],
    ) -> None:
        self.data.update(values)

    def snapshot(self) -> dict[str, Any]:
        return dict(self.data)

    def clear(self) -> None:
        self.data.clear()


@dataclass
class WorkflowStepResult:
    name: str
    success: bool
    result: Any = None
    error: Exception | None = None


@dataclass
class ConditionalStep:
    name: str
    condition: Callable[
        [WorkflowState],
        bool,
    ]
    when_true: Callable[
        [WorkflowState],
        Any,
    ]
    when_false: Callable[
        [WorkflowState],
        Any,
    ] | None = None


@dataclass
class ParallelStep:
    name: str
    tasks: dict[
        str,
        Callable[[WorkflowState], Any],
    ]


class WorkflowEngine:
    """
    Executes conditional and parallel workflow operations.

    Supports:
    - shared workflow state
    - conditional branches
    - parallel tasks
    - deterministic result collection
    - failure propagation
    """

    def __init__(
        self,
        max_workers: int = 4,
    ) -> None:
        if max_workers <= 0:
            raise ValueError(
                "max_workers must be greater than 0."
            )

        self.max_workers = max_workers

    def run_condition(
        self,
        step: ConditionalStep,
        state: WorkflowState | None = None,
    ) -> WorkflowStepResult:
        if not isinstance(
            step,
            ConditionalStep,
        ):
            raise TypeError(
                "step must be a ConditionalStep."
            )

        state = (
            state
            if state is not None
            else WorkflowState()
        )

        try:
            decision = step.condition(
                state
            )

            if decision:
                result = step.when_true(
                    state
                )
            elif step.when_false is not None:
                result = step.when_false(
                    state
                )
            else:
                result = None

            return WorkflowStepResult(
                name=step.name,
                success=True,
                result=result,
            )

        except Exception as exc:
            return WorkflowStepResult(
                name=step.name,
                success=False,
                error=exc,
            )

    def run_parallel(
        self,
        step: ParallelStep,
        state: WorkflowState | None = None,
        stop_on_error: bool = True,
    ) -> dict[str, WorkflowStepResult]:
        if not isinstance(
            step,
            ParallelStep,
        ):
            raise TypeError(
                "step must be a ParallelStep."
            )

        state = (
            state
            if state is not None
            else WorkflowState()
        )

        if not step.tasks:
            return {}

        results: dict[
            str,
            WorkflowStepResult,
        ] = {}

        with ThreadPoolExecutor(
            max_workers=min(
                self.max_workers,
                len(step.tasks),
            )
        ) as executor:

            futures = {
                executor.submit(
                    task,
                    state,
                ): name
                for name, task
                in step.tasks.items()
            }

            for future in as_completed(
                futures
            ):
                name = futures[future]

                try:
                    result = future.result()

                    results[name] = WorkflowStepResult(
                        name=name,
                        success=True,
                        result=result,
                    )

                except Exception as exc:
                    results[name] = WorkflowStepResult(
                        name=name,
                        success=False,
                        error=exc,
                    )

                    if stop_on_error:
                        for pending in futures:
                            if not pending.done():
                                pending.cancel()

                        break

        return results

    def execute(
        self,
        operations: list[
            ConditionalStep
            | ParallelStep
        ],
        state: WorkflowState | None = None,
        stop_on_error: bool = True,
    ) -> list[Any]:
        state = (
            state
            if state is not None
            else WorkflowState()
        )

        outputs: list[Any] = []

        for operation in operations:

            if isinstance(
                operation,
                ConditionalStep,
            ):
                result = self.run_condition(
                    operation,
                    state,
                )

                if not result.success:
                    if stop_on_error:
                        raise WorkflowEngineError(
                            f"Workflow step "
                            f"'{operation.name}' failed."
                        ) from result.error

                    outputs.append(result)
                    continue

                outputs.append(
                    result.result
                )
                continue

            if isinstance(
                operation,
                ParallelStep,
            ):
                result_map = self.run_parallel(
                    operation,
                    state,
                    stop_on_error=stop_on_error,
                )

                failed = [
                    result
                    for result in result_map.values()
                    if not result.success
                ]

                if failed and stop_on_error:
                    first_error = failed[0].error

                    raise WorkflowEngineError(
                        f"Parallel workflow "
                        f"'{operation.name}' failed."
                    ) from first_error

                outputs.append(
                    result_map
                )
                continue

            raise TypeError(
                "Unsupported workflow operation."
            )

        return outputs
