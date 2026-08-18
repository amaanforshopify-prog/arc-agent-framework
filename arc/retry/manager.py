from __future__ import annotations

import asyncio
import inspect
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Type


class RetryError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(
        self,
        message: str,
        last_exception: Exception | None = None,
        attempts: int = 0,
    ) -> None:
        super().__init__(message)
        self.last_exception = last_exception
        self.attempts = attempts


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    delay: float = 0.1
    backoff: float = 2.0
    max_delay: float = 30.0
    jitter: float = 0.0
    retry_exceptions: tuple[Type[BaseException], ...] = (Exception,)

    def __post_init__(self) -> None:
        if not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

        if self.delay < 0:
            raise ValueError("delay must be >= 0")

        if self.backoff < 1:
            raise ValueError("backoff must be >= 1")

        if self.max_delay < 0:
            raise ValueError("max_delay must be >= 0")

        if self.jitter < 0:
            raise ValueError("jitter must be >= 0")

    def get_delay(self, attempt: int) -> float:
        if attempt < 1:
            return 0.0

        value = self.delay * (self.backoff ** (attempt - 1))
        value = min(value, self.max_delay)

        if self.jitter:
            value += random.uniform(0, self.jitter)

        return value


@dataclass
class RetryStats:
    attempts: int = 0
    retries: int = 0
    successes: int = 0
    failures: int = 0
    total_delay: float = 0.0
    last_exception: Exception | None = None


class RetryManager:
    """Production-friendly retry manager for sync and async functions."""

    def __init__(
        self,
        policy: RetryPolicy | None = None,
    ) -> None:
        self.policy = policy or RetryPolicy()
        self.stats = RetryStats()

    def reset_stats(self) -> None:
        self.stats = RetryStats()

    def should_retry(self, exc: BaseException) -> bool:
        return isinstance(exc, self.policy.retry_exceptions)

    def _record_failure(self, exc: Exception) -> None:
        self.stats.failures += 1
        self.stats.last_exception = exc

    def execute(
        self,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not callable(function):
            raise TypeError("function must be callable")

        if inspect.iscoroutinefunction(function):
            raise TypeError(
                "Async function detected. Use execute_async()."
            )

        last_exception: Exception | None = None

        for attempt in range(1, self.policy.max_attempts + 1):
            self.stats.attempts += 1

            try:
                result = function(*args, **kwargs)

                if inspect.isawaitable(result):
                    result.close()
                    raise TypeError(
                        "Async function detected. Use execute_async()."
                    )

                self.stats.successes += 1
                return result

            except TypeError as exc:
                if str(exc) == (
                    "Async function detected. Use execute_async()."
                ):
                    raise

                last_exception = exc
                self._record_failure(exc)

                if (
                    attempt >= self.policy.max_attempts
                    or not self.should_retry(exc)
                ):
                    raise RetryError(
                        f"Operation failed after {attempt} attempts",
                        last_exception=exc,
                        attempts=attempt,
                    ) from exc

                self.stats.retries += 1

                delay = self.policy.get_delay(attempt)
                self.stats.total_delay += delay

                if delay:
                    time.sleep(delay)

            except Exception as exc:
                last_exception = exc
                self._record_failure(exc)

                if (
                    attempt >= self.policy.max_attempts
                    or not self.should_retry(exc)
                ):
                    raise RetryError(
                        f"Operation failed after {attempt} attempts",
                        last_exception=exc,
                        attempts=attempt,
                    ) from exc

                self.stats.retries += 1

                delay = self.policy.get_delay(attempt)
                self.stats.total_delay += delay

                if delay:
                    time.sleep(delay)

        raise RetryError(
            "Operation failed",
            last_exception=last_exception,
            attempts=self.policy.max_attempts,
        )

    async def execute_async(
        self,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not callable(function):
            raise TypeError("function must be callable")

        last_exception: Exception | None = None

        for attempt in range(1, self.policy.max_attempts + 1):
            self.stats.attempts += 1

            try:
                result = function(*args, **kwargs)

                if inspect.isawaitable(result):
                    result = await result

                self.stats.successes += 1
                return result

            except Exception as exc:
                last_exception = exc
                self._record_failure(exc)

                if (
                    attempt >= self.policy.max_attempts
                    or not self.should_retry(exc)
                ):
                    raise RetryError(
                        f"Async operation failed after {attempt} attempts",
                        last_exception=exc,
                        attempts=attempt,
                    ) from exc

                self.stats.retries += 1

                delay = self.policy.get_delay(attempt)
                self.stats.total_delay += delay

                if delay:
                    await asyncio.sleep(delay)

        raise RetryError(
            "Async operation failed",
            last_exception=last_exception,
            attempts=self.policy.max_attempts,
        )


__all__ = [
    "RetryError",
    "RetryPolicy",
    "RetryStats",
    "RetryManager",
]
