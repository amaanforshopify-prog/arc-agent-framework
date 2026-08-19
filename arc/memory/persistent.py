from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from .memory import Memory, MemoryEntry


class PersistentMemory(Memory):
    """
    File-backed ARC memory.

    Keeps the existing Memory API while persisting entries
    to a JSON file.
    """

    def __init__(
        self,
        path: str,
        max_entries: int | None = None,
        autosave: bool = True,
    ) -> None:
        super().__init__(max_entries=max_entries)

        if not isinstance(path, str) or not path.strip():
            raise ValueError("path cannot be empty")

        self.path = path
        self.autosave = autosave

        self._load()

    def store(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        entry = super().store(
            key=key,
            value=value,
            metadata=metadata,
        )

        if self.autosave:
            self.save()

        return entry

    def update(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        entry = super().update(
            key=key,
            value=value,
            metadata=metadata,
        )

        if self.autosave:
            self.save()

        return entry

    def delete(self, key: str) -> Any:
        value = super().delete(key)

        if self.autosave:
            self.save()

        return value

    def clear(self) -> None:
        super().clear()

        if self.autosave:
            self.save()

    def save(self) -> None:
        directory = os.path.dirname(
            os.path.abspath(self.path)
        )

        os.makedirs(
            directory,
            exist_ok=True,
        )

        data = {
            "max_entries": self.max_entries,
            "entries": [
                {
                    "key": entry.key,
                    "value": entry.value,
                    "memory_id": entry.memory_id,
                    "created_at": entry.created_at,
                    "updated_at": entry.updated_at,
                    "metadata": entry.metadata,
                }
                for entry in self._entries.values()
            ],
        }

        fd, temp_path = tempfile.mkstemp(
            prefix="arc_memory_",
            suffix=".tmp",
            dir=directory,
            text=True,
        )

        try:
            with os.fdopen(
                fd,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    data,
                    file,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )

            os.replace(
                temp_path,
                self.path,
            )

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return

        try:
            with open(
                self.path,
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not load memory file: {self.path}"
            ) from exc

        entries = data.get("entries", [])

        self._entries.clear()

        for raw in entries:
            entry = MemoryEntry(
                key=raw["key"],
                value=raw.get("value"),
                memory_id=raw.get(
                    "memory_id"
                ),
                created_at=raw.get(
                    "created_at"
                ),
                updated_at=raw.get(
                    "updated_at"
                ),
                metadata=raw.get(
                    "metadata",
                    {},
                ),
            )

            self._entries[entry.key] = entry

        self._enforce_limit()

    def _enforce_limit(self) -> None:
        if self.max_entries is None:
            return

        while len(self._entries) > self.max_entries:
            oldest = min(
                self._entries.values(),
                key=lambda item: item.created_at,
            )

            del self._entries[oldest.key]
