from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any

from arc.timeout import TimeoutError, TimeoutManager
from arc.tools import ToolError, ToolRegistry


@dataclass
class ExecutionResult:
    success: bool
    result: Any = None
    error: Exception | None = None
    attempts: int = 1
    duration: float = 0.0


class ExecutorError(Exception):
    """Raised when executor configuration or execution fails."""


class Executor:
    """
    ARC execution engine.

    Handles:
    - Tool lookup
    - Sync/async execution
    - Timeout
    - Retry
    - Execution results
    """

    def __init__(
        self,
        tools: ToolRegistry | None = None,
        timeout_manager: TimeoutManager | None = None,
        max_retries: int = 0,
        retry_delay: float = 0.0,
    ):
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        if retry_delay < 0:
            raise ValueError("retry_delay cannot be negative")

        self.tools = tools or ToolRegistry()
        self.timeout_manager = timeout_manager or TimeoutManager()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def execute(
        self,
        tool_name: str,
        *args: Any,
        timeout: float | None = None,
        max_retries: int | None = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a registered synchronous tool."""

        if max_retries is None:
            max_retries = self.max_retries

        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        total_attempts = max_retries + 1
        last_error: Exception | None = None

        start = __import__("time").perf_counter()

        for attempt in range(1, total_attempts + 1):
            try:
                tool = self.tools.get(tool_name)

                if inspect.iscoroutinefunction(tool.function):
                    raise ExecutorError(
                        f"Tool '{tool_name}' is async. "
                        "Use execute_async()."
                    )

                result = self.timeout_manager.run(
                    tool.execute,
                    *args,
                    timeout=timeout,
                    **kwargs,
                )

                duration = __import__("time").perf_counter() - start

                return ExecutionResult(
                    success=True,
                    result=result,
                    attempts=attempt,
                    duration=duration,
                )

            except Exception as exc:
                last_error = exc

                if attempt < total_attempts and self.retry_delay:
                    __import__("time").sleep(self.retry_delay)

        duration = __import__("time").perf_counter() - start

        return ExecutionResult(
            success=False,
            error=last_error,
            attempts=total_attempts,
            duration=duration,
        )

    async def execute_async(
        self,
        tool_name: str,
        *args: Any,
        timeout: float | None = None,
        max_retries: int | None = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a sync or async tool asynchronously."""

        if max_retries is None:
            max_retries = self.max_retries

        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        total_attempts = max_retries + 1
        last_error: Exception | None = None

        start = __import__("time").perf_counter()

        for attempt in range(1, total_attempts + 1):
            try:
                tool = self.tools.get(tool_name)

                result = await self.timeout_manager.run_async(
                    tool.execute_async,
                    *args,
                    timeout=timeout,
                    **kwargs,
                )

                duration = __import__("time").perf_counter() - start

                return ExecutionResult(
                    success=True,
                    result=result,
                    attempts=attempt,
                    duration=duration,
                )

            except Exception as exc:
                last_error = exc

                if attempt < total_attempts and self.retry_delay:
                    await asyncio.sleep(self.retry_delay)

        duration = __import__("time").perf_counter() - start

        return ExecutionResult(
            success=False,
            error=last_error,
            attempts=total_attempts,
            duration=duration,
        )

    def shutdown(self) -> None:
        self.timeout_manager.shutdown()
