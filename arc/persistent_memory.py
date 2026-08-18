import sqlite3
from pathlib import Path

from .types import Message


class PersistentMemory:
    """SQLite-backed persistent conversation memory."""

    def __init__(self, db_path: str = "data/arc_memory.db"):
        self.db_path = Path(db_path)

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(
            self.db_path
        )

        self._initialize()

    def _initialize(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                name TEXT,
                tool_call_id TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        self.connection.commit()

    def add(self, message: Message) -> None:
        self.connection.execute(
            """
            INSERT INTO messages (
                role,
                content,
                name,
                tool_call_id
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                message.role,
                message.content,
                message.name,
                message.tool_call_id,
            ),
        )

        self.connection.commit()

    def get_messages(self) -> list[Message]:
        cursor = self.connection.execute(
            """
            SELECT
                role,
                content,
                name,
                tool_call_id
            FROM messages
            ORDER BY id ASC
            """
        )

        rows = cursor.fetchall()

        return [
            Message(
                role=row[0],
                content=row[1],
                name=row[2],
                tool_call_id=row[3],
            )
            for row in rows
        ]

    def count(self) -> int:
        cursor = self.connection.execute(
            "SELECT COUNT(*) FROM messages"
        )

        return int(cursor.fetchone()[0])

    def clear(self) -> None:
        self.connection.execute(
            "DELETE FROM messages"
        )

        self.connection.commit()

    def close(self) -> None:
        self.connection.close()