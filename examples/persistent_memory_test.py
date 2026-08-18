from arc.memory import ConversationMemory
from arc.persistent_memory import PersistentMemory
from arc.types import Message


def main() -> None:
    persistent = PersistentMemory()

    # Start clean for this test.
    persistent.clear()

    short_term = ConversationMemory()

    user_message = Message(
        role="user",
        content="My favorite language is Python.",
    )

    assistant_message = Message(
        role="assistant",
        content="Got it. Your favorite language is Python.",
    )

    short_term.add(user_message)
    short_term.add(assistant_message)

    persistent.add(user_message)
    persistent.add(assistant_message)

    print("Short-term memory:")
    for message in short_term.get_messages():
        print(f"{message.role}: {message.content}")

    print()
    print("Persistent memory:")
    for message in persistent.get_messages():
        print(f"{message.role}: {message.content}")

    print()
    print("Stored messages:", persistent.count())

    persistent.close()


if __name__ == "__main__":
    main()