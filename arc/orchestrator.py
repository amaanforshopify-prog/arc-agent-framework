from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .memory import Memory
from .planner import Plan, Planner
from .runtime import AgentRuntime as LLMRuntime
from .trace import Trace, TraceManager


class OrchestratorError(Exception):
    """Base error for ARC workflow orchestration."""


@dataclass
class WorkflowResult:
    """Result returned by an ARC workflow execution."""

    success: bool
    output: Any = None
    error: Exception | None = None

    plan: Plan | None = None
    trace: Trace | None = None

    duration: float = 0.0
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class WorkflowOrchestrator:
    """
    High-level ARC workflow coordinator.

    Coordinates:
    - planning
    - agent runtime
    - memory
    - tracing
    - execution lifecycle

    The underlying AgentRuntime remains responsible for
    LLM interaction and tool-calling loops.
    """

    def __init__(
        self,
        runtime: LLMRuntime,
        planner: Planner | None = None,
        memory: Memory | None = None,
        trace_manager: TraceManager | None = None,
    ) -> None:
        if runtime is None:
            raise ValueError(
                "runtime cannot be None."
            )

        self.runtime = runtime
        self.planner = (
            planner
            if planner is not None
            else Planner()
        )
        self.memory = memory
        self.trace_manager = (
            trace_manager
            if trace_manager is not None
            else TraceManager()
        )

        self._run_counter = 0
        self.history: list[WorkflowResult] = []

    def create_plan(
        self,
        goal: str,
        steps: list[str] | None = None,
    ) -> Plan:
        return self.planner.create(
            goal,
            steps,
        )

    def run(
        self,
        goal: str,
        system_prompt: str,
        user_input: str,
        steps: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkflowResult:
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError(
                "goal cannot be empty."
            )

        if not isinstance(
            system_prompt,
            str,
        ) or not system_prompt.strip():
            raise ValueError(
                "system_prompt cannot be empty."
            )

        if not isinstance(
            user_input,
            str,
        ) or not user_input.strip():
            raise ValueError(
                "user_input cannot be empty."
            )

        self._run_counter += 1

        run_id = self._run_counter
        trace_id = f"workflow-{run_id}"

        trace = self.trace_manager.start(
            trace_id
        )

        start = time.perf_counter()

        plan = self.create_plan(
            goal,
            steps,
        )

        try:
            trace.emit(
                "workflow_started",
                goal=goal,
                run_id=run_id,
            )

            trace.emit(
                "planning_started",
                goal=goal,
                step_count=len(plan.steps),
            )

            if plan.steps:
                self.planner.start(plan)

            trace.emit(
                "planning_completed",
                step_count=len(plan.steps),
                plan_status=plan.status,
            )

            trace.emit(
                "runtime_started",
                input=user_input,
            )

            result = self.runtime.run(
                system_prompt=system_prompt,
                user_input=user_input,
            )

            trace.emit(
                "runtime_completed",
                success=True,
                iterations=result.metadata.get(
                    "iterations",
                    0,
                ),
            )

            duration = (
                time.perf_counter()
                - start
            )

            if plan.steps:
                plan.status = "completed"

            trace.emit(
                "workflow_completed",
                success=True,
                duration=duration,
            )

            workflow_result = WorkflowResult(
                success=True,
                output=result.output,
                plan=plan,
                trace=trace,
                duration=duration,
                metadata={
                    "run_id": run_id,
                    "goal": goal,
                    "runtime_iterations": (
                        result.metadata.get(
                            "iterations",
                            0,
                        )
                    ),
                    "runtime_memory_count": (
                        result.metadata.get(
                            "memory_count",
                            0,
                        )
                    ),
                    **(
                        metadata
                        if metadata is not None
                        else {}
                    ),
                },
            )

            self.history.append(
                workflow_result
            )

            return workflow_result

        except Exception as exc:
            duration = (
                time.perf_counter()
                - start
            )

            trace.emit(
                "workflow_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                duration=duration,
            )

            workflow_result = WorkflowResult(
                success=False,
                error=exc,
                plan=plan,
                trace=trace,
                duration=duration,
                metadata={
                    "run_id": run_id,
                    "goal": goal,
                    **(
                        metadata
                        if metadata is not None
                        else {}
                    ),
                },
            )

            self.history.append(
                workflow_result
            )

            return workflow_result

        finally:
            self.trace_manager.finish(
                trace_id
            )

    def last_result(
        self,
    ) -> WorkflowResult | None:
        if not self.history:
            return None

        return self.history[-1]

    def run_count(self) -> int:
        return len(self.history)

    def clear_history(self) -> None:
        self.history.clear()

    def snapshot(self) -> dict[str, Any]:
        last = self.last_result()

        return {
            "runs": self.run_count(),
            "last_success": (
                last.success
                if last is not None
                else None
            ),
            "last_duration": (
                last.duration
                if last is not None
                else 0.0
            ),
            "last_run_id": (
                last.metadata.get("run_id")
                if last is not None
                else None
            ),
        }

