from arc import Memory, MemoryError


def main():
    print("=" * 65)
    print("ARC MEMORY SYSTEM TEST")
    print("=" * 65)

    memory = Memory()

    # -----------------------------------------
    # 1. Store
    # -----------------------------------------

    print("\n[1] Store memory")

    entry = memory.store(
        "user_name",
        "Azaan",
    )

    print(f"  Key: {entry.key}")
    print(f"  Value: {entry.value}")
    print(f"  ID: {entry.memory_id}")

    assert memory.count() == 1
    assert memory.recall("user_name") == "Azaan"

    print("  PASS")

    # -----------------------------------------
    # 2. Metadata
    # -----------------------------------------

    print("\n[2] Store metadata")

    memory.store(
        "project",
        "ARC",
        metadata={
            "type": "project",
            "priority": "high",
        },
    )

    entry = memory.get("project")

    print(f"  Value: {entry.value}")
    print(f"  Metadata: {entry.metadata}")

    assert entry.metadata["priority"] == "high"

    print("  PASS")

    # -----------------------------------------
    # 3. Update
    # -----------------------------------------

    print("\n[3] Update memory")

    memory.update(
        "project",
        "ARC AI Agent Framework",
        metadata={
            "version": "0.1",
        },
    )

    assert (
        memory.recall("project")
        == "ARC AI Agent Framework"
    )

    print(
        f"  Updated: "
        f"{memory.recall('project')}"
    )

    print("  PASS")

    # -----------------------------------------
    # 4. Has
    # -----------------------------------------

    print("\n[4] Check memory")

    assert memory.has("user_name")
    assert not memory.has("missing")

    print("  user_name: EXISTS")
    print("  missing: NOT FOUND")
    print("  PASS")

    # -----------------------------------------
    # 5. Search
    # -----------------------------------------

    print("\n[5] Search memory")

    memory.store(
        "favorite_language",
        "Python",
    )

    memory.store(
        "favorite_framework",
        "ARC",
    )

    results = memory.search("python")

    print(f"  Results: {len(results)}")

    for result in results:
        print(
            f"  - {result.key}: "
            f"{result.value}"
        )

    assert len(results) == 1
    assert results[0].key == "favorite_language"

    print("  PASS")

    # -----------------------------------------
    # 6. Snapshot
    # -----------------------------------------

    print("\n[6] Snapshot")

    snapshot = memory.snapshot()

    print(f"  {snapshot}")

    assert snapshot["user_name"] == "Azaan"

    print("  PASS")

    # -----------------------------------------
    # 7. Delete
    # -----------------------------------------

    print("\n[7] Delete")

    deleted = memory.delete(
        "favorite_language"
    )

    print(f"  Deleted: {deleted}")

    assert deleted == "Python"
    assert not memory.has(
        "favorite_language"
    )

    print("  PASS")

    # -----------------------------------------
    # 8. Missing key
    # -----------------------------------------

    print("\n[8] Missing key handling")

    try:
        memory.get("does_not_exist")
        print("  FAIL")

    except MemoryError as exc:
        print(f"  PASS: {exc}")

    # -----------------------------------------
    # 9. Max entries
    # -----------------------------------------

    print("\n[9] Max entries")

    limited = Memory(max_entries=2)

    limited.store("one", 1)
    limited.store("two", 2)
    limited.store("three", 3)

    print(
        f"  Keys: {limited.keys()}"
    )

    assert limited.count() == 2
    assert not limited.has("one")
    assert limited.has("three")

    print("  PASS")

    # -----------------------------------------
    # 10. Clear
    # -----------------------------------------

    print("\n[10] Clear")

    memory.clear()

    assert memory.count() == 0
    assert memory.keys() == []

    print("  Memory cleared")
    print("  PASS")

    print("\n" + "=" * 65)
    print("ALL MEMORY TESTS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
