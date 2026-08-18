from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


class MemoryError(Exception):
    """Raised when an ARC memory operation fails."""


@dataclass
class MemoryEntry:
    """A single piece of agent memory."""

    key: str
    value: Any
    memory_id: str = field(
        default_factory=lambda: uuid.uuid4().hex
    )
    created_at: float = field(
        default_factory=time.time
    )
    updated_at: float = field(
        default_factory=time.time
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def update(
        self,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.value = value
        self.updated_at = time.time()

        if metadata is not None:
            self.metadata.update(metadata)


class Memory:
    """
    Lightweight in-memory storage for ARC agents.

    Supports:
    - Store
    - Recall
    - Update
    - Delete
    - Search
    - History
    - Metadata
    """

    def __init__(self, max_entries: int | None = None):
        if max_entries is not None and max_entries <= 0:
            raise ValueError(
                "max_entries must be greater than 0"
            )

        self.max_entries = max_entries
        self._entries: dict[str, MemoryEntry] = {}

    def store(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Create or replace a memory entry."""

        self._validate_key(key)

        if key in self._entries:
            entry = self._entries[key]

            entry.update(
                value=value,
                metadata=metadata,
            )

            return entry

        if (
            self.max_entries is not None
            and len(self._entries) >= self.max_entries
        ):
            oldest = min(
                self._entries.values(),
                key=lambda item: item.created_at,
            )

            del self._entries[oldest.key]

        entry = MemoryEntry(
            key=key,
            value=value,
            metadata=metadata or {},
        )

        self._entries[key] = entry

        return entry

    def recall(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """Return a stored value."""

        entry = self._entries.get(key)

        if entry is None:
            return default

        return entry.value

    def get(self, key: str) -> MemoryEntry:
        """Return the complete memory entry."""

        self._validate_key(key)

        if key not in self._entries:
            raise MemoryError(
                f"Memory key '{key}' does not exist."
            )

        return self._entries[key]

    def has(self, key: str) -> bool:
        """Check whether a key exists."""
        return key in self._entries

    def update(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Update an existing memory."""

        entry = self.get(key)

        entry.update(
            value=value,
            metadata=metadata,
        )

        return entry

    def delete(self, key: str) -> Any:
        """Delete a memory and return its value."""

        self._validate_key(key)

        if key not in self._entries:
            raise MemoryError(
                f"Memory key '{key}' does not exist."
            )

        return self._entries.pop(key).value

    def search(self, query: str) -> list[MemoryEntry]:
        """
        Search keys and string values.

        Matching is case-insensitive.
        """

        if not query:
            return []

        query = query.lower()
        results: list[MemoryEntry] = []

        for entry in self._entries.values():
            key_match = query in entry.key.lower()

            value_match = query in str(
                entry.value
            ).lower()

            if key_match or value_match:
                results.append(entry)

        return results

    def entries(self) -> list[MemoryEntry]:
        """Return all memory entries."""
        return list(self._entries.values())

    def keys(self) -> list[str]:
        """Return all memory keys."""
        return list(self._entries.keys())

    def values(self) -> list[Any]:
        """Return all memory values."""
        return [
            entry.value
            for entry in self._entries.values()
        ]

    def count(self) -> int:
        """Return number of stored entries."""
        return len(self._entries)

    def clear(self) -> None:
        """Delete all memories."""
        self._entries.clear()

    def snapshot(self) -> dict[str, Any]:
        """Return a simple key/value snapshot."""
        return {
            entry.key: entry.value
            for entry in self._entries.values()
        }

    def _validate_key(self, key: str) -> None:
        if not isinstance(key, str):
            raise TypeError(
                "Memory key must be a string."
            )

        if not key.strip():
            raise ValueError(
                "Memory key cannot be empty."
            )
