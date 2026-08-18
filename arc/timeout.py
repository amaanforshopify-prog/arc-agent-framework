import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class TimeoutError(Exception):
    """Raised when an operation exceeds its timeout."""


class TimeoutManager:
    """Execute operations with a maximum allowed execution time."""

    def __init__(self, default_timeout: float = 30.0):
        if default_timeout <= 0:
            raise ValueError("default_timeout must be greater than 0")

        self.default_timeout = default_timeout
        self._executor = ThreadPoolExecutor()

    def run(
        self,
        func: Callable[..., T],
        *args: Any,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> T:
        """Run a synchronous function with a timeout."""
        timeout = self._validate_timeout(timeout)

        future = self._executor.submit(func, *args, **kwargs)

        try:
            return future.result(timeout=timeout)

        except FutureTimeoutError:
            future.cancel()
            raise TimeoutError(
                f"Operation timed out after {timeout:.2f} seconds."
            )

    async def run_async(
        self,
        func: Callable[..., Any],
        *args: Any,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run sync or async functions with a timeout."""
        timeout = self._validate_timeout(timeout)

        if asyncio.iscoroutinefunction(func):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Operation timed out after {timeout:.2f} seconds."
                )

        loop = asyncio.get_running_loop()

        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    lambda: func(*args, **kwargs),
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Operation timed out after {timeout:.2f} seconds."
            )

    def _validate_timeout(self, timeout: float | None) -> float:
        value = self.default_timeout if timeout is None else timeout

        if value <= 0:
            raise ValueError("timeout must be greater than 0")

        return value

    def shutdown(self) -> None:
        """Release resources used by the manager."""
        self._executor.shutdown(wait=False)
