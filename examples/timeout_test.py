import asyncio
import time

from arc.timeout import TimeoutError, TimeoutManager


def slow_function():
    print("  Slow function started...")
    time.sleep(3)
    return "Success!"


def fast_function():
    print("  Fast function started...")
    return "Success!"


async def async_slow_function():
    print("  Async slow function started...")
    await asyncio.sleep(3)
    return "Async Success!"


async def async_fast_function():
    print("  Async fast function started...")
    await asyncio.sleep(0.1)
    return "Async Success!"


async def main():
    manager = TimeoutManager(default_timeout=1)

    print("=" * 50)
    print("ARC TimeoutManager Tests")
    print("=" * 50)

    print("\n[1] Sync timeout test")
    try:
        manager.run(slow_function)
        print("  FAIL: timeout was expected")
    except TimeoutError as e:
        print(f"  PASS: {e}")

    print("\n[2] Sync success test")
    try:
        result = manager.run(fast_function)
        print(f"  PASS: {result}")
    except TimeoutError:
        print("  FAIL: unexpected timeout")

    print("\n[3] Async timeout test")
    try:
        await manager.run_async(async_slow_function)
        print("  FAIL: timeout was expected")
    except TimeoutError as e:
        print(f"  PASS: {e}")

    print("\n[4] Async success test")
    try:
        result = await manager.run_async(async_fast_function)
        print(f"  PASS: {result}")
    except TimeoutError:
        print("  FAIL: unexpected timeout")

    print("\n[5] Custom timeout test")
    try:
        result = manager.run(fast_function, timeout=5)
        print(f"  PASS: {result}")
    except TimeoutError:
        print("  FAIL: unexpected timeout")

    print("\n[6] Invalid timeout test")
    try:
        manager.run(fast_function, timeout=0)
        print("  FAIL: ValueError was expected")
    except ValueError as e:
        print(f"  PASS: {e}")

    manager.shutdown()

    print("\n" + "=" * 50)
    print("ALL TIMEOUT TESTS COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
