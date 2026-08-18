from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .agent import Agent
from .context import ContextManager
from .memory import Memory
from .planning_engine import PlanningEngine, PlanningResult
from .planner import Plan


class AgentRuntimeError(Exception):
    pass


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
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentRuntime:
    """Connects ARC Agent subsystems into one execution runtime."""

    def __init__(
        self,
        agent: Agent,
        planner_engine: PlanningEngine | None = None,
        context: ContextManager | None = None,
        memory: Memory | None = None,
    ):
        self.agent = agent
        self.engine = planner_engine or PlanningEngine()
        self.context = context or ContextManager()
        self.memory = memory
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

        self.context.user(task)

        memory_used = False

        if self.memory is not None:
            previous = self.memory.search(task)

            if previous:
                memory_used = True

                self.context.system(
                    f"Relevant memory: {previous}"
                )

        plan = self.engine.create_plan(
            task,
            steps,
        )

        planning_result = self.engine.execute(
            plan,
            stop_on_error=stop_on_error,
        )

        result = planning_result.final_result

        if self.memory is not None and planning_result.success:
            self.memory.store(
                f"run_{self._run_counter}",
                result,
            )

        if result is not None:
            self.context.assistant(
                str(result)
            )

        run = AgentRun(
            run_id=self._run_counter,
            task=task,
            success=planning_result.success,
            result=result,
            plan=plan,
            planning_result=planning_result,
            context_count=self.context.count(),
            memory_used=memory_used,
        )

        self.runs.append(run)

        return run

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

    def snapshot(self) -> dict[str, Any]:
        return {
            "runs": self.run_count(),
            "context_messages": self.context.count(),
            "last_run": (
                self.last_run().run_id
                if self.last_run()
                else None
            ),
        }
