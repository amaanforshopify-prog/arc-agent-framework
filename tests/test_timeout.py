import asyncio
import time

import pytest

from arc.timeout import TimeoutError, TimeoutManager


def test_timeout_manager_returns_result():
    manager = TimeoutManager(timeout=1)

    try:
        assert manager.run(lambda: "success") == "success"
    finally:
        manager.shutdown()


def test_timeout_manager_passes_args():
    manager = TimeoutManager(timeout=1)

    def add(a, b):
        return a + b

    try:
        assert manager.run(add, 2, 3) == 5
    finally:
        manager.shutdown()


def test_timeout_manager_passes_kwargs():
    manager = TimeoutManager(timeout=1)

    def greet(name, prefix="Hello"):
        return f"{prefix}, {name}"

    try:
        assert manager.run(
            greet,
            "ARC",
            prefix="Welcome",
        ) == "Welcome, ARC"
    finally:
        manager.shutdown()


def test_timeout_manager_times_out():
    manager = TimeoutManager(timeout=0.05)

    def slow_operation():
        time.sleep(0.2)
        return "done"

    try:
        with pytest.raises(TimeoutError, match="timed out"):
            manager.run(slow_operation)
    finally:
        manager.shutdown()


def test_timeout_manager_per_call_timeout():
    manager = TimeoutManager(timeout=1)

    def slow_operation():
        time.sleep(0.2)
        return "done"

    try:
        with pytest.raises(TimeoutError):
            manager.run(
                slow_operation,
                timeout=0.05,
            )
    finally:
        manager.shutdown()


def test_timeout_manager_default_timeout():
    manager = TimeoutManager(timeout=0.05)

    def slow_operation():
        time.sleep(0.2)

    try:
        with pytest.raises(TimeoutError):
            manager.run(slow_operation)
    finally:
        manager.shutdown()


def test_timeout_manager_rejects_invalid_timeout():
    with pytest.raises(ValueError):
        TimeoutManager(timeout=0)

    with pytest.raises(ValueError):
        TimeoutManager(timeout=-1)


def test_timeout_manager_rejects_invalid_per_call_timeout():
    manager = TimeoutManager(timeout=1)

    try:
        with pytest.raises(ValueError):
            manager.run(lambda: "ok", timeout=0)

        with pytest.raises(ValueError):
            manager.run(lambda: "ok", timeout=-1)
    finally:
        manager.shutdown()


def test_timeout_manager_async_result():
    manager = TimeoutManager(timeout=1)

    async def async_operation():
        await asyncio.sleep(0.01)
        return "async-success"

    async def run_test():
        return await manager.run_async(async_operation)

    try:
        assert asyncio.run(run_test()) == "async-success"
    finally:
        manager.shutdown()


def test_timeout_manager_async_args():
    manager = TimeoutManager(timeout=1)

    async def add(a, b):
        await asyncio.sleep(0.01)
        return a + b

    async def run_test():
        return await manager.run_async(add, 5, 7)

    try:
        assert asyncio.run(run_test()) == 12
    finally:
        manager.shutdown()


def test_timeout_manager_async_timeout():
    manager = TimeoutManager(timeout=0.05)

    async def slow_operation():
        await asyncio.sleep(0.2)
        return "done"

    async def run_test():
        with pytest.raises(TimeoutError, match="timed out"):
            await manager.run_async(slow_operation)

    try:
        asyncio.run(run_test())
    finally:
        manager.shutdown()


def test_timeout_manager_sync_function_async_execution():
    manager = TimeoutManager(timeout=1)

    def sync_operation():
        return "sync-from-async"

    async def run_test():
        return await manager.run_async(sync_operation)

    try:
        assert asyncio.run(run_test()) == "sync-from-async"
    finally:
        manager.shutdown()


def test_timeout_manager_shutdown():
    manager = TimeoutManager(timeout=1)

    manager.shutdown()

    with pytest.raises(RuntimeError, match="shut down"):
        manager.run(lambda: "should fail")


def test_timeout_manager_double_shutdown():
    manager = TimeoutManager(timeout=1)

    manager.shutdown()
    manager.shutdown()
