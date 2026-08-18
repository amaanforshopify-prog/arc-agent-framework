import asyncio

from arc import (
    RetryManager,
    RetryPolicy,
    RetryError,
)


def main():
    print("=" * 60)
    print("ARC RETRY MANAGER")
    print("=" * 60)

    state = {"attempts": 0}

    def unreliable_operation():
        state["attempts"] += 1

        print(
            f"Attempt {state['attempts']}..."
        )

        if state["attempts"] < 3:
            raise RuntimeError(
                "Temporary failure"
            )

        return "Operation succeeded!"

    manager = RetryManager(
        RetryPolicy(
            max_attempts=5,
            delay=0.1,
            backoff=2,
            max_delay=2,
        )
    )

    try:
        result = manager.execute(
            unreliable_operation
        )

        print()
        print("RESULT:", result)

    except RetryError as exc:
        print("FAILED:", exc)

    print()
    print("Attempts:", manager.stats.attempts)
    print("Retries:", manager.stats.retries)
    print("Successes:", manager.stats.successes)
    print("Failures:", manager.stats.failures)
    print("=" * 60)


async def async_demo():
    manager = RetryManager(
        RetryPolicy(
            max_attempts=3,
            delay=0.1,
        )
    )

    state = {"attempts": 0}

    async def operation():
        state["attempts"] += 1

        if state["attempts"] < 2:
            raise RuntimeError("Async temporary failure")

        return "Async success!"

    result = await manager.execute_async(
        operation
    )

    print(result)


if __name__ == "__main__":
    main()
    asyncio.run(async_demo())
