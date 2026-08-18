from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .agent import Agent
from .context import ContextManager
from .memory import Memory
from .planning_engine import PlanningEngine, PlanningResult
from .planner import Plan
from .trace import Trace, TraceManager


class AgentRuntimeError(Exception):
    """Base exception for AgentRuntime errors."""


@dataclass
class AgentRun:
    run_id: int
    task: str
    success: bool
    result: Any = None
    plan: Plan | None = None
    planning_result: PlanningResult | None = None
    context_count: int = 0
    memory_used: bool = False
    trace: Trace | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentRuntime:
    """Coordinates agent execution, memory, context, planning and tracing."""

    def __init__(
        self,
        agent: Agent,
        planner_engine: PlanningEngine | None = None,
        context: ContextManager | None = None,
        memory: Memory | None = None,
        trace_manager: TraceManager | None = None,
    ):
        self.agent = agent
        self.engine = planner_engine or PlanningEngine()
        self.context = context or ContextManager()
        self.memory = memory
        self.trace_manager = trace_manager or TraceManager()

        self._run_counter = 0
        self.runs: list[AgentRun] = []

    def run(
        self,
        task: str,
        steps: list[str] | None = None,
        stop_on_error: bool = True,
    ) -> AgentRun:

        if not isinstance(task, str) or not task.strip():
            raise ValueError("task cannot be empty.")

        self._run_counter += 1
        numeric_id = self._run_counter
        trace_id = f"run-{numeric_id}"

        trace = self.trace_manager.start(trace_id)

        try:
            trace.emit(
                "context_started",
                task=task,
            )

            self.context.user(task)

            memory_used = False

            if self.memory is not None:
                trace.emit("memory_search_started")

                previous = self.memory.search(task)

                if previous:
                    memory_used = True
                    self.context.system(
                        f"Relevant memory: {previous}"
                    )

                trace.emit(
                    "memory_search_completed",
                    found=memory_used,
                )

            trace.emit(
                "planning_started",
                task=task,
            )

            plan = self.engine.create_plan(
                task,
                steps,
            )

            trace.emit(
                "planning_completed",
                step_count=len(plan.steps),
            )

            trace.emit(
                "execution_started",
            )

            planning_result = self.engine.execute(
                plan,
                stop_on_error=stop_on_error,
            )

            trace.emit(
                "execution_completed",
                success=planning_result.success,
                completed_steps=planning_result.completed_steps,
            )

            result = planning_result.final_result

            if self.memory is not None and planning_result.success:
                self.memory.store(
                    f"run_{numeric_id}",
                    result,
                )

                trace.emit(
                    "memory_saved",
                    key=f"run_{numeric_id}",
                )

            if result is not None:
                self.context.assistant(
                    str(result)
                )

            trace.emit(
                "context_completed",
                message_count=self.context.count(),
            )

            run = AgentRun(
                run_id=numeric_id,
                task=task,
                success=planning_result.success,
                result=result,
                plan=plan,
                planning_result=planning_result,
                context_count=self.context.count(),
                memory_used=memory_used,
                trace=trace,
            )

            self.runs.append(run)

            return run

        except Exception as exc:
            trace.emit(
                "run_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )

            raise

        finally:
            self.trace_manager.finish(trace_id)

    def last_run(self) -> AgentRun | None:
        if not self.runs:
            return None

        return self.runs[-1]

    def run_count(self) -> int:
        return len(self.runs)

    def clear_runs(self) -> None:
        self.runs.clear()

    def reset_context(self) -> None:
        self.context.clear()

    def get_trace(
        self,
        run_id: int | None = None,
    ) -> Trace | None:

        if run_id is None:
            run = self.last_run()
            return run.trace if run else None

        return self.trace_manager.get(
            f"run-{run_id}"
        )

    def snapshot(self) -> dict[str, Any]:
        last = self.last_run()

        return {
            "runs": self.run_count(),
            "context_messages": self.context.count(),
            "last_run": (
                last.run_id
                if last
                else None
            ),
            "last_trace_events": (
                last.trace.count()
                if last and last.trace
                else 0
            ),
        }

