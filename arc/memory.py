from .types import Message


class ConversationMemory:
    """Stores messages from the current agent conversation."""

    def __init__(self):
        self.messages: list[Message] = []

    def add(self, message: Message) -> None:
        self.messages.append(message)

    def add_many(self, messages: list[Message]) -> None:
        self.messages.extend(messages)

    def get_messages(self) -> list[Message]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()

    def count(self) -> int:
        return len(self.messages)