import asyncio

import pytest

from arc import (
    RetryError,
    RetryManager,
    RetryPolicy,
)


def test_policy_creation():
    policy = RetryPolicy()

    assert policy.max_attempts == 3
    assert policy.delay == 0.1


def test_policy_custom():
    policy = RetryPolicy(
        max_attempts=5,
        delay=0,
        backoff=3,
    )

    assert policy.max_attempts == 5
    assert policy.backoff == 3


def test_invalid_attempts():
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)


def test_invalid_delay():
    with pytest.raises(ValueError):
        RetryPolicy(delay=-1)


def test_invalid_backoff():
    with pytest.raises(ValueError):
        RetryPolicy(backoff=0)


def test_invalid_max_delay():
    with pytest.raises(ValueError):
        RetryPolicy(max_delay=-1)


def test_invalid_jitter():
    with pytest.raises(ValueError):
        RetryPolicy(jitter=-1)


def test_delay_calculation():
    policy = RetryPolicy(
        delay=1,
        backoff=2,
        max_delay=100,
    )

    assert policy.get_delay(1) == 1
    assert policy.get_delay(2) == 2
    assert policy.get_delay(3) == 4


def test_max_delay():
    policy = RetryPolicy(
        delay=10,
        backoff=10,
        max_delay=20,
    )

    assert policy.get_delay(3) == 20


def test_success_first_attempt():
    manager = RetryManager(
        RetryPolicy(
            max_attempts=3,
            delay=0,
        )
    )

    result = manager.execute(lambda: "success")

    assert result == "success"
    assert manager.stats.attempts == 1
    assert manager.stats.successes == 1
    assert manager.stats.retries == 0


def test_retry_then_success():
    state = {"count": 0}

    def operation():
        state["count"] += 1

        if state["count"] < 3:
            raise RuntimeError("temporary failure")

        return "done"

    manager = RetryManager(
        RetryPolicy(
            max_attempts=3,
            delay=0,
        )
    )

    result = manager.execute(operation)

    assert result == "done"
    assert state["count"] == 3
    assert manager.stats.attempts == 3
    assert manager.stats.retries == 2
    assert manager.stats.successes == 1


def test_exhausted_retries():
    def operation():
        raise RuntimeError("always fails")

    manager = RetryManager(
        RetryPolicy(
            max_attempts=3,
            delay=0,
        )
    )

    with pytest.raises(RetryError) as exc:
        manager.execute(operation)

    assert exc.value.attempts == 3
    assert isinstance(exc.value.last_exception, RuntimeError)
    assert manager.stats.attempts == 3
    assert manager.stats.retries == 2


def test_non_retryable_exception():
    class PermanentError(Exception):
        pass

    manager = RetryManager(
        RetryPolicy(
            max_attempts=5,
            delay=0,
            retry_exceptions=(RuntimeError,),
        )
    )

    def operation():
        raise PermanentError("permanent")

    with pytest.raises(RetryError) as exc:
        manager.execute(operation)

    assert exc.value.attempts == 1
    assert manager.stats.attempts == 1


def test_should_retry():
    manager = RetryManager()

    assert manager.should_retry(RuntimeError())
    assert manager.should_retry(ValueError())


def test_stats_reset():
    manager = RetryManager(
        RetryPolicy(
            max_attempts=2,
            delay=0,
        )
    )

    def operation():
        raise RuntimeError()

    with pytest.raises(RetryError):
        manager.execute(operation)

    assert manager.stats.attempts == 2

    manager.reset_stats()

    assert manager.stats.attempts == 0
    assert manager.stats.retries == 0
    assert manager.stats.successes == 0
    assert manager.stats.failures == 0


@pytest.mark.asyncio
async def test_async_success():
    manager = RetryManager(
        RetryPolicy(
            max_attempts=3,
            delay=0,
        )
    )

    async def operation():
        return "async success"

    result = await manager.execute_async(operation)

    assert result == "async success"


@pytest.mark.asyncio
async def test_async_retry():
    manager = RetryManager(
        RetryPolicy(
            max_attempts=3,
            delay=0,
        )
    )

    state = {"count": 0}

    async def operation():
        state["count"] += 1

        if state["count"] < 3:
            raise RuntimeError("retry")

        return "async done"

    result = await manager.execute_async(operation)

    assert result == "async done"
    assert state["count"] == 3
    assert manager.stats.retries == 2


@pytest.mark.asyncio
async def test_async_exhausted():
    manager = RetryManager(
        RetryPolicy(
            max_attempts=2,
            delay=0,
        )
    )

    async def operation():
        raise RuntimeError("fail")

    with pytest.raises(RetryError):
        await manager.execute_async(operation)

    assert manager.stats.attempts == 2


def test_callable_arguments():
    manager = RetryManager(
        RetryPolicy(
            max_attempts=1,
            delay=0,
        )
    )

    def add(a, b):
        return a + b

    assert manager.execute(add, 5, 7) == 12


def test_keyword_arguments():
    manager = RetryManager(
        RetryPolicy(
            max_attempts=1,
            delay=0,
        )
    )

    def greet(name="world"):
        return f"hello {name}"

    assert manager.execute(greet, name="ARC") == "hello ARC"


def test_non_callable():
    manager = RetryManager()

    with pytest.raises(TypeError):
        manager.execute("not callable")


def test_sync_rejects_async():
    manager = RetryManager(
        RetryPolicy(
            max_attempts=1,
            delay=0,
        )
    )

    async def operation():
        return "async"

    with pytest.raises(TypeError):
        manager.execute(operation)


def test_jitter():
    policy = RetryPolicy(
        delay=1,
        backoff=2,
        jitter=0.5,
    )

    value = policy.get_delay(1)

    assert 1 <= value <= 1.5


def test_failure_stats():
    manager = RetryManager(
        RetryPolicy(
            max_attempts=2,
            delay=0,
        )
    )

    def operation():
        raise RuntimeError("boom")

    with pytest.raises(RetryError):
        manager.execute(operation)

    assert manager.stats.failures == 2
    assert isinstance(
        manager.stats.last_exception,
        RuntimeError,
    )
