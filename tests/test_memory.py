import pytest

from arc import Memory, MemoryError


def test_memory_creation():
    memory = Memory()

    assert memory.count() == 0
    assert memory.keys() == []


def test_store_and_recall():
    memory = Memory()

    memory.store(
        "name",
        "Azaan",
    )

    assert memory.recall("name") == "Azaan"


def test_missing_recall_default():
    memory = Memory()

    assert (
        memory.recall(
            "missing",
            "default",
        )
        == "default"
    )


def test_get_entry():
    memory = Memory()

    entry = memory.store(
        "name",
        "Azaan",
    )

    result = memory.get("name")

    assert result is entry
    assert result.value == "Azaan"


def test_memory_metadata():
    memory = Memory()

    memory.store(
        "project",
        "ARC",
        metadata={
            "priority": "high",
        },
    )

    entry = memory.get("project")

    assert entry.metadata["priority"] == "high"


def test_update():
    memory = Memory()

    memory.store(
        "status",
        "old",
    )

    memory.update(
        "status",
        "new",
    )

    assert memory.recall("status") == "new"


def test_update_metadata():
    memory = Memory()

    memory.store(
        "project",
        "ARC",
        metadata={
            "version": "1",
        },
    )

    memory.update(
        "project",
        "ARC Framework",
        metadata={
            "version": "2",
        },
    )

    entry = memory.get("project")

    assert entry.value == "ARC Framework"
    assert entry.metadata["version"] == "2"


def test_has():
    memory = Memory()

    memory.store("exists", True)

    assert memory.has("exists")
    assert not memory.has("missing")


def test_delete():
    memory = Memory()

    memory.store(
        "name",
        "Azaan",
    )

    value = memory.delete("name")

    assert value == "Azaan"
    assert not memory.has("name")


def test_delete_missing():
    memory = Memory()

    with pytest.raises(MemoryError):
        memory.delete("missing")


def test_get_missing():
    memory = Memory()

    with pytest.raises(MemoryError):
        memory.get("missing")


def test_search_key():
    memory = Memory()

    memory.store(
        "favorite_language",
        "Python",
    )

    results = memory.search(
        "language"
    )

    assert len(results) == 1
    assert results[0].key == "favorite_language"


def test_search_value():
    memory = Memory()

    memory.store(
        "language",
        "Python programming",
    )

    results = memory.search(
        "python"
    )

    assert len(results) == 1


def test_search_case_insensitive():
    memory = Memory()

    memory.store(
        "language",
        "Python",
    )

    results = memory.search(
        "PYTHON"
    )

    assert len(results) == 1


def test_empty_search():
    memory = Memory()

    memory.store(
        "name",
        "Azaan",
    )

    assert memory.search("") == []


def test_entries():
    memory = Memory()

    memory.store("one", 1)
    memory.store("two", 2)

    entries = memory.entries()

    assert len(entries) == 2


def test_values():
    memory = Memory()

    memory.store("one", 1)
    memory.store("two", 2)

    assert memory.values() == [1, 2]


def test_snapshot():
    memory = Memory()

    memory.store("one", 1)
    memory.store("two", 2)

    assert memory.snapshot() == {
        "one": 1,
        "two": 2,
    }


def test_clear():
    memory = Memory()

    memory.store("one", 1)
    memory.store("two", 2)

    memory.clear()

    assert memory.count() == 0


def test_max_entries():
    memory = Memory(
        max_entries=2
    )

    memory.store("one", 1)
    memory.store("two", 2)
    memory.store("three", 3)

    assert memory.count() == 2
    assert not memory.has("one")
    assert memory.has("two")
    assert memory.has("three")


def test_invalid_max_entries():
    with pytest.raises(ValueError):
        Memory(max_entries=0)


def test_invalid_key_type():
    memory = Memory()

    with pytest.raises(TypeError):
        memory.store(123, "value")


def test_empty_key():
    memory = Memory()

    with pytest.raises(ValueError):
        memory.store("", "value")


def test_store_existing_key_updates():
    memory = Memory()

    first = memory.store(
        "name",
        "Azaan",
    )

    second = memory.store(
        "name",
        "ARC",
    )

    assert first is second
    assert memory.recall("name") == "ARC"


def test_memory_entry_id():
    memory = Memory()

    first = memory.store(
        "one",
        1,
    )

    second = memory.store(
        "two",
        2,
    )

    assert first.memory_id != second.memory_id
