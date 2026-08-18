from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from arc.executor import ExecutionResult, Executor
from arc.memory import Memory


class AgentError(Exception):
    """Raised when an ARC agent cannot complete a task."""


@dataclass
class AgentResult:
    """Final result returned by an ARC agent."""

    success: bool
    output: Any = None
    error: Exception | None = None
    tool_name: str | None = None
    execution: ExecutionResult | None = None
    duration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class Agent:
    """
    ARC Agent.

    Handles:
    - Task execution
    - Executor integration
    - Memory
    - History
    - Statistics
    """

    def __init__(
        self,
        name: str,
        executor: Executor,
        description: str = "",
        memory: Memory | None = None,
    ):
        if not name or not name.strip():
            raise ValueError("Agent name cannot be empty.")

        self.name = name
        self.description = description
        self.executor = executor

        # Use supplied memory or create one automatically.
        self.memory = memory if memory is not None else Memory()

        self.status = "idle"
        self.current_task: str | None = None

        self.total_runs = 0
        self.successful_runs = 0
        self.failed_runs = 0

        self.history: list[AgentResult] = []

    # ---------------------------------------------------------
    # MEMORY
    # ---------------------------------------------------------

    def remember(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ):
        """Store information in agent memory."""

        return self.memory.store(
            key,
            value,
            metadata,
        )

    def recall(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """Recall information from agent memory."""

        return self.memory.recall(
            key,
            default,
        )

    def forget(self, key: str) -> Any:
        """Delete information from agent memory."""

        return self.memory.delete(key)

    def search_memory(self, query: str):
        """Search agent memory."""

        return self.memory.search(query)

    def clear_memory(self) -> None:
        """Clear all agent memory."""

        self.memory.clear()

    def remember_result(
        self,
        key: str,
        result: AgentResult,
    ) -> None:
        """Store an AgentResult output in memory."""

        self.memory.store(
            key,
            result.output,
            metadata={
                "success": result.success,
                "tool": result.tool_name,
                "duration": result.duration,
            },
        )

    # ---------------------------------------------------------
    # SYNC RUN
    # ---------------------------------------------------------

    def run(
        self,
        task: str,
        tool_name: str,
        *args: Any,
        timeout: float | None = None,
        max_retries: int | None = None,
        remember_as: str | None = None,
        **kwargs: Any,
    ) -> AgentResult:

        if not task or not task.strip():
            raise ValueError("Task cannot be empty.")

        start = time.perf_counter()

        self.status = "running"
        self.current_task = task
        self.total_runs += 1

        try:
            execution = self.executor.execute(
                tool_name,
                *args,
                timeout=timeout,
                max_retries=max_retries,
                **kwargs,
            )

            duration = time.perf_counter() - start

            if execution.success:
                self.status = "completed"
                self.successful_runs += 1

                result = AgentResult(
                    success=True,
                    output=execution.result,
                    tool_name=tool_name,
                    execution=execution,
                    duration=duration,
                    metadata={
                        "agent": self.name,
                        "task": task,
                    },
                )

            else:
                self.status = "failed"
                self.failed_runs += 1

                result = AgentResult(
                    success=False,
                    error=execution.error,
                    tool_name=tool_name,
                    execution=execution,
                    duration=duration,
                    metadata={
                        "agent": self.name,
                        "task": task,
                    },
                )

            self.history.append(result)

            if remember_as and result.success:
                self.remember_result(
                    remember_as,
                    result,
                )

            return result

        except Exception as exc:
            self.status = "failed"
            self.failed_runs += 1

            duration = time.perf_counter() - start

            result = AgentResult(
                success=False,
                error=exc,
                tool_name=tool_name,
                duration=duration,
                metadata={
                    "agent": self.name,
                    "task": task,
                },
            )

            self.history.append(result)

            return result

        finally:
            self.current_task = None

    # ---------------------------------------------------------
    # ASYNC RUN
    # ---------------------------------------------------------

    async def run_async(
        self,
        task: str,
        tool_name: str,
        *args: Any,
        timeout: float | None = None,
        max_retries: int | None = None,
        remember_as: str | None = None,
        **kwargs: Any,
    ) -> AgentResult:

        if not task or not task.strip():
            raise ValueError("Task cannot be empty.")

        start = time.perf_counter()

        self.status = "running"
        self.current_task = task
        self.total_runs += 1

        try:
            execution = await self.executor.execute_async(
                tool_name,
                *args,
                timeout=timeout,
                max_retries=max_retries,
                **kwargs,
            )

            duration = time.perf_counter() - start

            if execution.success:
                self.status = "completed"
                self.successful_runs += 1

                result = AgentResult(
                    success=True,
                    output=execution.result,
                    tool_name=tool_name,
                    execution=execution,
                    duration=duration,
                    metadata={
                        "agent": self.name,
                        "task": task,
                    },
                )

            else:
                self.status = "failed"
                self.failed_runs += 1

                result = AgentResult(
                    success=False,
                    error=execution.error,
                    tool_name=tool_name,
                    execution=execution,
                    duration=duration,
                    metadata={
                        "agent": self.name,
                        "task": task,
                    },
                )

            self.history.append(result)

            if remember_as and result.success:
                self.remember_result(
                    remember_as,
                    result,
                )

            return result

        except Exception as exc:
            self.status = "failed"
            self.failed_runs += 1

            duration = time.perf_counter() - start

            result = AgentResult(
                success=False,
                error=exc,
                tool_name=tool_name,
                duration=duration,
                metadata={
                    "agent": self.name,
                    "task": task,
                },
            )

            self.history.append(result)

            return result

        finally:
            self.current_task = None

    # ---------------------------------------------------------
    # STATS
    # ---------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "memory_entries": self.memory.count(),
            "success_rate": (
                self.successful_runs / self.total_runs
                if self.total_runs
                else 0.0
            ),
        }

    # ---------------------------------------------------------
    # RESET
    # ---------------------------------------------------------

    def reset(self) -> None:
        """Reset execution state and history."""

        self.status = "idle"
        self.current_task = None
        self.total_runs = 0
        self.successful_runs = 0
        self.failed_runs = 0
        self.history.clear()
