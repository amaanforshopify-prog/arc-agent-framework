from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from .agent_workflow import AgentWorkflowEngine, AgentWorkflowStep
from .models import Model
from .tools import ToolRegistry
from .workflow_state import DurableWorkflowState


class MultiAgentError(Exception):
    """Raised when multi-agent orchestration fails."""


@dataclass
class AgentSpec:
    name: str
    system_prompt: str
    user_input: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentExecutionResult:
    name: str
    success: bool
    output: Any = None
    error: Exception | None = None


@dataclass
class MultiAgentResult:
    success: bool
    results: list[AgentExecutionResult]
    resumed: bool = False
    error: Exception | None = None


class MultiAgentCoordinator:
    """Coordinates multiple ARC agents over shared durable state."""

    def __init__(
        self,
        model: Model,
        tools: ToolRegistry | None = None,
        max_iterations: int = 10,
        max_workers: int = 4,
    ) -> None:
        if max_workers <= 0:
            raise ValueError(
                "max_workers must be greater than 0."
            )

        self.engine = AgentWorkflowEngine(
            model=model,
            tools=tools,
            max_iterations=max_iterations,
        )
        self.max_workers = max_workers

    def run_agent(
        self,
        agent: AgentSpec,
        state: DurableWorkflowState,
    ) -> AgentExecutionResult:
        try:
            step = AgentWorkflowStep(
                name=agent.name,
                system_prompt=agent.system_prompt,
                user_input=agent.user_input,
                metadata=agent.metadata,
            )

            output = self.engine.execute_step(
                step,
                state,
            )

            state.set(
                f"result:{agent.name}",
                output,
            )

            state.set(
                f"agent_result:{agent.name}",
                output,
            )

            state.mark_completed(
                agent.name
            )

            state.save()

            return AgentExecutionResult(
                name=agent.name,
                success=True,
                output=output,
            )

        except Exception as exc:
            state.metadata["last_error"] = str(exc)
            state.metadata["failed_agent"] = agent.name
            state.save()

            return AgentExecutionResult(
                name=agent.name,
                success=False,
                error=exc,
            )

    def run_parallel(
        self,
        agents: list[AgentSpec],
        state: DurableWorkflowState,
        stop_on_error: bool = True,
    ) -> MultiAgentResult:

        results: dict[
            str,
            AgentExecutionResult,
        ] = {}

        pending = [
            agent
            for agent in agents
            if not state.is_completed(agent.name)
        ]

        resumed = (
            len(pending) != len(agents)
        )

        for agent in agents:
            if state.is_completed(agent.name):
                results[agent.name] = AgentExecutionResult(
                    name=agent.name,
                    success=True,
                    output=state.get(
                        f"result:{agent.name}",
                        state.get(
                            f"agent_result:{agent.name}"
                        ),
                    ),
                )

        if pending:
            with ThreadPoolExecutor(
                max_workers=min(
                    self.max_workers,
                    len(pending),
                )
            ) as executor:

                futures = {
                    executor.submit(
                        self.run_agent,
                        agent,
                        state,
                    ): agent.name
                    for agent in pending
                }

                for future in as_completed(
                    futures
                ):
                    result = future.result()
                    results[result.name] = result

                    if (
                        not result.success
                        and stop_on_error
                    ):
                        for other in futures:
                            if not other.done():
                                other.cancel()
                        break

        ordered = [
            results[agent.name]
            for agent in agents
            if agent.name in results
        ]

        failures = [
            result
            for result in ordered
            if not result.success
        ]

        return MultiAgentResult(
            success=not failures,
            results=ordered,
            resumed=resumed,
            error=(
                failures[0].error
                if failures
                else None
            ),
        )

    def run_sequential(
        self,
        agents: list[AgentSpec],
        state: DurableWorkflowState,
        stop_on_error: bool = True,
    ) -> MultiAgentResult:

        results: list[
            AgentExecutionResult
        ] = []

        resumed = bool(
            state.completed_steps
        )

        for agent in agents:

            if state.is_completed(
                agent.name
            ):
                results.append(
                    AgentExecutionResult(
                        name=agent.name,
                        success=True,
                        output=state.get(
                            f"result:{agent.name}",
                            state.get(
                                f"agent_result:{agent.name}"
                            ),
                        ),
                    )
                )
                continue

            state.begin_step(
                agent.name
            )
            state.save()

            result = self.run_agent(
                agent,
                state,
            )

            results.append(result)

            if (
                not result.success
                and stop_on_error
            ):
                return MultiAgentResult(
                    success=False,
                    results=results,
                    resumed=resumed,
                    error=result.error,
                )

        state.current_step = None
        state.save()

        return MultiAgentResult(
            success=all(
                result.success
                for result in results
            ),
            results=results,
            resumed=resumed,
        )
