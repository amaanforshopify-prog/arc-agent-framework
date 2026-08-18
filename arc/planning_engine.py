from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .planner import Plan, PlanStep, Planner
from .tools import ToolRegistry


class PlanningEngineError(Exception):
    """Base error for the ARC planning engine."""


@dataclass
class StepResult:
    step_id: int
    success: bool
    result: Any = None
    error: Exception | None = None


@dataclass
class PlanningResult:
    goal: str
    success: bool
    results: list[StepResult] = field(
        default_factory=list
    )
    plan: Plan | None = None

    @property
    def completed_steps(self) -> int:
        return sum(
            result.success
            for result in self.results
        )

    @property
    def failed_steps(self) -> int:
        return sum(
            not result.success
            for result in self.results
        )

    @property
    def final_result(self) -> Any:
        successful = [
            result
            for result in self.results
            if result.success
        ]

        if not successful:
            return None

        return successful[-1].result


class PlanningEngine:
    """
    Executes ARC plans using the registered tool system.

    Supports:
    - Manual plans
    - Tool execution
    - Callable execution
    - Step tracking
    - Failure handling
    - Execution results
    """

    def __init__(
        self,
        planner: Planner | None = None,
        tools: ToolRegistry | None = None,
    ):
        self.planner = planner or Planner()
        self.tools = tools or ToolRegistry()

    def create_plan(
        self,
        goal: str,
        steps: list[str] | None = None,
    ) -> Plan:

        return self.planner.create(
            goal,
            steps,
        )

    def execute(
        self,
        plan: Plan,
        stop_on_error: bool = True,
    ) -> PlanningResult:

        if not isinstance(plan, Plan):
            raise TypeError(
                "plan must be a Plan."
            )

        if not plan.steps:
            raise PlanningEngineError(
                "Cannot execute an empty plan."
            )

        self.planner.start(plan)

        results: list[StepResult] = []

        for step in plan.steps:
            try:
                result = self._execute_step(
                    step
                )

                self.planner.complete_step(
                    plan,
                    step.id,
                    result,
                )

                results.append(
                    StepResult(
                        step_id=step.id,
                        success=True,
                        result=result,
                    )
                )

            except Exception as exc:
                self.planner.fail_step(
                    plan,
                    step.id,
                    exc,
                )

                results.append(
                    StepResult(
                        step_id=step.id,
                        success=False,
                        error=exc,
                    )
                )

                if stop_on_error:
                    break

        success = (
            len(results) == len(plan.steps)
            and all(
                result.success
                for result in results
            )
        )

        if success:
            plan.status = "completed"

        return PlanningResult(
            goal=plan.goal,
            success=success,
            results=results,
            plan=plan,
        )

    def _execute_step(
        self,
        step: PlanStep,
    ) -> Any:

        if step.tool:
            return self.tools.execute(
                step.tool,
                *step.args,
                **step.kwargs,
            )

        return self._execute_callable(
            step
        )

    def _execute_callable(
        self,
        step: PlanStep,
    ) -> Any:

        callback = step.kwargs.pop(
            "_callable",
            None,
        )

        if callback is None:
            return step.description

        if not callable(callback):
            raise TypeError(
                "_callable must be callable."
            )

        return callback(
            *step.args
        )

    def add_tool_step(
        self,
        plan: Plan,
        description: str,
        tool: str,
        *args: Any,
        **kwargs: Any,
    ) -> PlanStep:

        return self.planner.add_step(
            plan,
            description,
            tool,
            *args,
            **kwargs,
        )

    def add_callable_step(
        self,
        plan: Plan,
        description: str,
        callback: Callable[..., Any],
        *args: Any,
    ) -> PlanStep:

        return self.planner.add_step(
            plan,
            description,
            None,
            *args,
            _callable=callback,
        )
