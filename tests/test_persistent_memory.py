from arc.memory import PersistentMemory


def test_persistent_memory_save_and_load(tmp_path):
    path = tmp_path / "memory.json"

    memory = PersistentMemory(
        str(path)
    )

    memory.store(
        "username",
        "ARC",
        metadata={
            "source": "test"
        },
    )

    assert path.exists()

    loaded = PersistentMemory(
        str(path)
    )

    assert loaded.recall("username") == "ARC"

    entry = loaded.get("username")

    assert entry.metadata == {
        "source": "test"
    }


def test_persistent_memory_update(tmp_path):
    path = tmp_path / "memory.json"

    memory = PersistentMemory(
        str(path)
    )

    memory.store(
        "status",
        "pending",
    )

    memory.update(
        "status",
        "completed",
    )

    loaded = PersistentMemory(
        str(path)
    )

    assert loaded.recall("status") == "completed"


def test_persistent_memory_delete(tmp_path):
    path = tmp_path / "memory.json"

    memory = PersistentMemory(
        str(path)
    )

    memory.store(
        "temp",
        "value",
    )

    memory.delete("temp")

    loaded = PersistentMemory(
        str(path)
    )

    assert not loaded.has("temp")


def test_persistent_memory_clear(tmp_path):
    path = tmp_path / "memory.json"

    memory = PersistentMemory(
        str(path)
    )

    memory.store("one", 1)
    memory.store("two", 2)

    memory.clear()

    loaded = PersistentMemory(
        str(path)
    )

    assert loaded.count() == 0


def test_persistent_memory_max_entries(tmp_path):
    path = tmp_path / "memory.json"

    memory = PersistentMemory(
        str(path),
        max_entries=2,
    )

    memory.store("one", 1)
    memory.store("two", 2)
    memory.store("three", 3)

    loaded = PersistentMemory(
        str(path),
        max_entries=2,
    )

    assert loaded.count() == 2
    assert not loaded.has("one")
    assert loaded.has("two")
    assert loaded.has("three")


def test_persistent_memory_missing_file(tmp_path):
    path = tmp_path / "missing.json"

    memory = PersistentMemory(
        str(path)
    )

    assert memory.count() == 0


def test_persistent_memory_autosave_disabled(tmp_path):
    path = tmp_path / "memory.json"

    memory = PersistentMemory(
        str(path),
        autosave=False,
    )

    memory.store(
        "key",
        "value",
    )

    assert not path.exists()

    memory.save()

    assert path.exists()
