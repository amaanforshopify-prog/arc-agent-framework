from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class WorkflowStateError(Exception):
    """Raised when durable workflow state fails."""


@dataclass
class DurableWorkflowState:
    """SQLite-backed checkpoint state for workflows."""

    db_path: str = "data/arc_workflow.db"
    workflow_id: str = "default"
    data: dict[str, Any] = field(default_factory=dict)
    completed_steps: list[str] = field(default_factory=list)
    current_step: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(str(path))
        self._initialize()
        self.load()

    def _initialize(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_state (
                workflow_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                completed_steps TEXT NOT NULL,
                current_step TEXT,
                metadata TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def load(self) -> None:
        cursor = self.connection.execute(
            """
            SELECT
                data,
                completed_steps,
                current_step,
                metadata
            FROM workflow_state
            WHERE workflow_id = ?
            """,
            (self.workflow_id,),
        )

        row = cursor.fetchone()

        if row is None:
            return

        self.data = json.loads(row[0])
        self.completed_steps = json.loads(row[1])
        self.current_step = row[2]
        self.metadata = json.loads(row[3])

    def save(self) -> None:
        self.connection.execute(
            """
            INSERT INTO workflow_state (
                workflow_id,
                data,
                completed_steps,
                current_step,
                metadata
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(workflow_id)
            DO UPDATE SET
                data = excluded.data,
                completed_steps = excluded.completed_steps,
                current_step = excluded.current_step,
                metadata = excluded.metadata
            """,
            (
                self.workflow_id,
                json.dumps(self.data, default=str),
                json.dumps(self.completed_steps),
                self.current_step,
                json.dumps(self.metadata, default=str),
            ),
        )
        self.connection.commit()

    def set(self, key: str, value: Any) -> None:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("state key cannot be empty.")
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def update(self, values: dict[str, Any]) -> None:
        self.data.update(values)

    def begin_step(self, step: str) -> None:
        if not isinstance(step, str) or not step.strip():
            raise ValueError("step cannot be empty.")
        self.current_step = step

    def mark_completed(self, step: str) -> None:
        if not isinstance(step, str) or not step.strip():
            raise ValueError("step cannot be empty.")

        if step not in self.completed_steps:
            self.completed_steps.append(step)

        self.current_step = None

    def is_completed(self, step: str) -> bool:
        return step in self.completed_steps

    def snapshot(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "data": dict(self.data),
            "completed_steps": list(self.completed_steps),
            "current_step": self.current_step,
            "metadata": dict(self.metadata),
        }

    def reset(self) -> None:
        self.data.clear()
        self.completed_steps.clear()
        self.current_step = None
        self.metadata.clear()

        self.connection.execute(
            """
            DELETE FROM workflow_state
            WHERE workflow_id = ?
            """,
            (self.workflow_id,),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()
