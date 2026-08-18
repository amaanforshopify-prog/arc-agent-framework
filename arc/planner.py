from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class PlannerError(Exception):
    """Base planner error."""


@dataclass
class PlanStep:
    id: int
    description: str
    tool: str | None = None
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(
        default_factory=dict
    )
    status: str = "pending"
    result: Any = None
    error: Exception | None = None


@dataclass
class Plan:
    goal: str
    steps: list[PlanStep] = field(
        default_factory=list
    )
    status: str = "pending"
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def add_step(
        self,
        description: str,
        tool: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> PlanStep:

        step = PlanStep(
            id=len(self.steps) + 1,
            description=description,
            tool=tool,
            args=args,
            kwargs=kwargs,
        )

        self.steps.append(step)

        return step

    def next_step(self) -> PlanStep | None:
        for step in self.steps:
            if step.status == "pending":
                return step

        return None

    def completed(self) -> bool:
        return bool(self.steps) and all(
            step.status == "completed"
            for step in self.steps
        )

    def failed(self) -> bool:
        return any(
            step.status == "failed"
            for step in self.steps
        )

    def progress(self) -> float:
        if not self.steps:
            return 0.0

        completed = sum(
            step.status == "completed"
            for step in self.steps
        )

        return completed / len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "status": self.status,
            "steps": [
                {
                    "id": step.id,
                    "description": step.description,
                    "tool": step.tool,
                    "status": step.status,
                    "result": step.result,
                    "error": (
                        str(step.error)
                        if step.error
                        else None
                    ),
                }
                for step in self.steps
            ],
            "metadata": self.metadata,
        }


class Planner:
    """
    Creates and manages deterministic task plans.

    LLM-based planning can be plugged in later.
    """

    def create(
        self,
        goal: str,
        steps: list[str] | None = None,
    ) -> Plan:

        if not goal or not goal.strip():
            raise PlannerError(
                "Goal cannot be empty."
            )

        plan = Plan(
            goal=goal.strip()
        )

        if steps:
            for description in steps:
                plan.add_step(
                    description
                )

        plan.status = (
            "ready"
            if plan.steps
            else "pending"
        )

        return plan

    def add_step(
        self,
        plan: Plan,
        description: str,
        tool: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> PlanStep:

        if not isinstance(plan, Plan):
            raise TypeError(
                "plan must be a Plan."
            )

        return plan.add_step(
            description,
            tool,
            *args,
            **kwargs,
        )

    def start(self, plan: Plan) -> None:
        if not plan.steps:
            raise PlannerError(
                "Cannot start an empty plan."
            )

        plan.status = "running"

    def complete_step(
        self,
        plan: Plan,
        step_id: int,
        result: Any = None,
    ) -> None:

        step = self._find_step(
            plan,
            step_id,
        )

        step.status = "completed"
        step.result = result

        if plan.completed():
            plan.status = "completed"

    def fail_step(
        self,
        plan: Plan,
        step_id: int,
        error: Exception,
    ) -> None:

        step = self._find_step(
            plan,
            step_id,
        )

        step.status = "failed"
        step.error = error
        plan.status = "failed"

    def _find_step(
        self,
        plan: Plan,
        step_id: int,
    ) -> PlanStep:

        for step in plan.steps:
            if step.id == step_id:
                return step

        raise PlannerError(
            f"Unknown step: {step_id}"
        )
