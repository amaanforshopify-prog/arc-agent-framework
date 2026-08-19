import pytest

from arc.memory import (
    ConversationMemory,
    Memory,
    MemoryEntry,
    MemoryError,
)


def test_memory_is_concrete():
    memory = Memory()
    assert isinstance(memory, Memory)


def test_memory_store_and_recall():
    memory = Memory()

    entry = memory.store(
        "name",
        "ARC",
    )

    assert isinstance(entry, MemoryEntry)
    assert memory.recall("name") == "ARC"


def test_memory_get():
    memory = Memory()

    memory.store(
        "key",
        "value",
    )

    entry = memory.get("key")

    assert isinstance(entry, MemoryEntry)
    assert entry.key == "key"
    assert entry.value == "value"


def test_memory_has():
    memory = Memory()

    assert not memory.has("test")

    memory.store(
        "test",
        123,
    )

    assert memory.has("test")


def test_memory_search():
    memory = Memory()

    memory.store(
        "username",
        "Azaan",
    )

    memory.store(
        "city",
        "Karachi",
    )

    results = memory.search("aza")

    assert len(results) == 1
    assert results[0].key == "username"


def test_memory_delete():
    memory = Memory()

    memory.store(
        "temp",
        "value",
    )

    assert memory.delete("temp") == "value"
    assert not memory.has("temp")


def test_memory_clear():
    memory = Memory()

    memory.store("one", 1)
    memory.store("two", 2)

    memory.clear()

    assert memory.count() == 0
    assert memory.entries() == []


def test_memory_snapshot():
    memory = Memory()

    memory.store("one", 1)
    memory.store("two", 2)

    assert memory.snapshot() == {
        "one": 1,
        "two": 2,
    }


def test_memory_max_entries():
    memory = Memory(max_entries=2)

    memory.store("one", 1)
    memory.store("two", 2)
    memory.store("three", 3)

    assert memory.count() == 2
    assert not memory.has("one")
    assert memory.has("two")
    assert memory.has("three")


def test_memory_invalid_key():
    memory = Memory()

    with pytest.raises(ValueError):
        memory.store("", "value")


def test_memory_missing_key():
    memory = Memory()

    with pytest.raises(MemoryError):
        memory.get("missing")

    with pytest.raises(MemoryError):
        memory.delete("missing")


def test_conversation_memory():
    memory = ConversationMemory()

    memory.add_user("hello")
    memory.add_assistant("hi")

    assert memory.get_messages() == [
        {
            "role": "user",
            "content": "hello",
        },
        {
            "role": "assistant",
            "content": "hi",
        },
    ]


def test_conversation_memory_tool_message():
    memory = ConversationMemory()

    memory.add_tool(
        '{"result": 42}',
        tool_call_id="call_1",
        name="calculator",
    )

    message = memory.last()

    assert message is not None
    assert message["role"] == "tool"
    assert message["tool_call_id"] == "call_1"
    assert message["name"] == "calculator"


def test_conversation_memory_limit():
    memory = ConversationMemory(
        max_messages=2,
    )

    memory.add_user("one")
    memory.add_assistant("two")
    memory.add_user("three")

    assert memory.get_messages() == [
        {
            "role": "assistant",
            "content": "two",
        },
        {
            "role": "user",
            "content": "three",
        },
    ]


def test_conversation_memory_clear():
    memory = ConversationMemory()

    memory.add_user("hello")
    memory.clear()

    assert len(memory) == 0
    assert memory.get_messages() == []


def test_conversation_memory_invalid_role():
    memory = ConversationMemory()

    with pytest.raises(ValueError):
        memory.add(
            "",
            "invalid",
        )


def test_memory_entry_update():
    memory = Memory()

    entry = memory.store(
        "status",
        "pending",
    )

    old_updated = entry.updated_at

    entry.update("done")

    assert entry.value == "done"
    assert entry.updated_at >= old_updated
