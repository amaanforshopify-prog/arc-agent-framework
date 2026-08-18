from arc.retry import RetryManager


def main() -> None:
    attempts = 0

    retry_manager = RetryManager(
        max_retries=3,
        initial_delay=0.5,
        max_delay=2.0,
    )

    def unstable_function() -> str:
        nonlocal attempts

        attempts += 1

        print(
            f"Function attempt: {attempts}"
        )

        if attempts < 3:
            raise RuntimeError(
                "Simulated temporary failure."
            )

        return "Success!"

    result = retry_manager.run(
        unstable_function
    )

    print()
    print("Final result:", result)


if __name__ == "__main__":
    main()