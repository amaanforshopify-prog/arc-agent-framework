import asyncio
import time

import pytest

from arc.timeout import TimeoutError, TimeoutManager


def fast():
    return "ok"


def slow():
    time.sleep(2)
    return "too late"


async def async_fast():
    await asyncio.sleep(0.01)
    return "async ok"


async def async_slow():
    await asyncio.sleep(2)
    return "too late"


def test_sync_success():
    manager = TimeoutManager(1)

    assert manager.run(fast) == "ok"

    manager.shutdown()


def test_sync_timeout():
    manager = TimeoutManager(0.1)

    with pytest.raises(TimeoutError):
        manager.run(slow)

    manager.shutdown()


@pytest.mark.asyncio
async def test_async_success():
    manager = TimeoutManager(1)

    assert await manager.run_async(async_fast) == "async ok"

    manager.shutdown()


@pytest.mark.asyncio
async def test_async_timeout():
    manager = TimeoutManager(0.1)

    with pytest.raises(TimeoutError):
        await manager.run_async(async_slow)

    manager.shutdown()


def test_custom_timeout():
    manager = TimeoutManager(1)

    assert manager.run(fast, timeout=5) == "ok"

    manager.shutdown()


def test_invalid_timeout():
    manager = TimeoutManager(1)

    with pytest.raises(ValueError):
        manager.run(fast, timeout=0)

    manager.shutdown()
