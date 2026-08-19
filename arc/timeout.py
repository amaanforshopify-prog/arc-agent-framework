from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable


class TimeoutError(Exception):
    """Raised when an operation exceeds its allowed timeout."""


class TimeoutManager:
    """
    Manages synchronous and asynchronous operation timeouts.

    Supports:
    - Synchronous callable execution
    - Asynchronous callable execution
    - Per-call timeout overrides
    - Default timeout configuration
    - Graceful executor shutdown
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_workers: int = 8,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")

        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")

        self.timeout = timeout
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._shutdown = False

    def _resolve_timeout(self, timeout: float | None) -> float:
        """Resolve and validate the effective timeout."""

        effective_timeout = self.timeout if timeout is None else timeout

        if effective_timeout <= 0:
            raise ValueError("timeout must be greater than 0")

        return effective_timeout

    def _ensure_active(self) -> None:
        """Ensure the timeout manager has not been shut down."""

        if self._shutdown:
            raise RuntimeError("TimeoutManager has been shut down")

    def run(
        self,
        func: Callable[..., Any],
        *args: Any,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a synchronous callable with a timeout.

        Args:
            func: Callable to execute.
            *args: Positional arguments passed to the callable.
            timeout: Optional per-call timeout override.
            **kwargs: Keyword arguments passed to the callable.

        Returns:
            The callable's return value.

        Raises:
            TimeoutError: If execution exceeds the timeout.
            ValueError: If timeout is invalid.
            RuntimeError: If manager has been shut down.
        """

        self._ensure_active()
        effective_timeout = self._resolve_timeout(timeout)

        future: Future[Any] = self._executor.submit(
            func,
            *args,
            **kwargs,
        )

        try:
            return future.result(timeout=effective_timeout)

        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"Operation timed out after {effective_timeout} seconds"
            ) from exc

    async def run_async(
        self,
        func: Callable[..., Any],
        *args: Any,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a synchronous or asynchronous callable asynchronously.

        Async callables are executed directly under asyncio.wait_for().
        Synchronous callables are moved to the thread pool.

        Args:
            func: Callable or coroutine function.
            *args: Positional arguments.
            timeout: Optional per-call timeout override.
            **kwargs: Keyword arguments.

        Returns:
            The callable's return value.

        Raises:
            TimeoutError: If execution exceeds the timeout.
            ValueError: If timeout is invalid.
            RuntimeError: If manager has been shut down.
        """

        self._ensure_active()
        effective_timeout = self._resolve_timeout(timeout)

        if asyncio.iscoroutinefunction(func):
            coroutine = func(*args, **kwargs)

            try:
                return await asyncio.wait_for(
                    coroutine,
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    f"Operation timed out after {effective_timeout} seconds"
                ) from exc

        loop = asyncio.get_running_loop()

        future = loop.run_in_executor(
            self._executor,
            lambda: func(*args, **kwargs),
        )

        try:
            return await asyncio.wait_for(
                future,
                timeout=effective_timeout,
            )

        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"Operation timed out after {effective_timeout} seconds"
            ) from exc

    def shutdown(self) -> None:
        """Shutdown the underlying thread pool."""

        if self._shutdown:
            return

        self._shutdown = True
        self._executor.shutdown(
            wait=False,
            cancel_futures=True,
        )
