import time
from collections.abc import Callable
from typing import Any


class RetryManager:
    """Retries temporary failures with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 8.0,
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay

    def run(
        self,
        function: Callable[[], Any],
    ) -> Any:
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return function()

            except Exception as exc:
                last_error = exc

                if attempt >= self.max_retries:
                    break

                delay = min(
                    self.initial_delay * (2 ** attempt),
                    self.max_delay,
                )

                print(
                    f"[RETRY] Attempt {attempt + 1}/"
                    f"{self.max_retries} failed: {exc}"
                )

                print(
                    f"[RETRY] Waiting {delay:.1f}s..."
                )

                time.sleep(delay)

        raise RuntimeError(
            f"Operation failed after "
            f"{self.max_retries} retries."
        ) from last_error