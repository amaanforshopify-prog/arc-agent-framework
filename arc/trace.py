from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


class TraceError(Exception):
    """Base ARC trace error."""


@dataclass
class TraceEvent:
    event_id: str
    run_id: str
    event_type: str
    timestamp: float
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        return self.timestamp


@dataclass
class Trace:
    run_id: str
    started_at: float = field(default_factory=time.time)
    events: list[TraceEvent] = field(default_factory=list)
    finished_at: float | None = None

    def emit(
        self,
        event_type: str,
        **data: Any,
    ) -> TraceEvent:

        if not event_type:
            raise ValueError(
                "event_type cannot be empty."
            )

        event = TraceEvent(
            event_id=uuid.uuid4().hex,
            run_id=self.run_id,
            event_type=event_type,
            timestamp=time.time(),
            data=data,
        )

        self.events.append(event)
        return event

    def finish(self) -> None:
        self.finished_at = time.time()

    @property
    def duration(self) -> float:
        end = (
            self.finished_at
            if self.finished_at is not None
            else time.time()
        )

        return max(
            0.0,
            end - self.started_at,
        )

    def count(self) -> int:
        return len(self.events)

    def filter(
        self,
        event_type: str,
    ) -> list[TraceEvent]:

        return [
            event
            for event in self.events
            if event.event_type == event_type
        ]

    def last(
        self,
        event_type: str | None = None,
    ) -> TraceEvent | None:

        if event_type is None:
            return (
                self.events[-1]
                if self.events
                else None
            )

        events = self.filter(event_type)

        return (
            events[-1]
            if events
            else None
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration": self.duration,
            "event_count": self.count(),
            "events": [
                {
                    "event_id": event.event_id,
                    "run_id": event.run_id,
                    "event_type": event.event_type,
                    "timestamp": event.timestamp,
                    "data": event.data,
                }
                for event in self.events
            ],
        }


class TraceManager:
    """Stores and manages traces for ARC agent runs."""

    def __init__(self):
        self.traces: dict[str, Trace] = {}
        self.active: dict[str, Trace] = {}

    def start(
        self,
        run_id: str | None = None,
    ) -> Trace:

        run_id = (
            run_id
            or f"run-{uuid.uuid4().hex[:12]}"
        )

        if run_id in self.traces:
            raise TraceError(
                f"Trace already exists: {run_id}"
            )

        trace = Trace(run_id=run_id)

        trace.emit(
            "run_started"
        )

        self.traces[run_id] = trace
        self.active[run_id] = trace

        return trace

    def get(
        self,
        run_id: str,
    ) -> Trace | None:

        return self.traces.get(run_id)

    def finish(
        self,
        run_id: str,
    ) -> Trace:

        trace = self.get(run_id)

        if trace is None:
            raise TraceError(
                f"Unknown run: {run_id}"
            )

        if run_id not in self.active:
            return trace

        trace.emit(
            "run_completed"
        )

        trace.finish()

        self.active.pop(
            run_id,
            None,
        )

        return trace

    def emit(
        self,
        run_id: str,
        event_type: str,
        **data: Any,
    ) -> TraceEvent:

        trace = self.get(run_id)

        if trace is None:
            raise TraceError(
                f"Unknown run: {run_id}"
            )

        return trace.emit(
            event_type,
            **data,
        )

    def list_runs(self) -> list[str]:
        return list(self.traces.keys())

    def clear(self) -> None:
        self.traces.clear()
        self.active.clear()

    def remove(
        self,
        run_id: str,
    ) -> bool:

        existed = run_id in self.traces

        self.traces.pop(
            run_id,
            None,
        )

        self.active.pop(
            run_id,
            None,
        )

        return existed

    def subscribe(
        self,
        callback: Callable[
            [TraceEvent],
            Any
        ],
    ) -> None:

        if not callable(callback):
            raise TypeError(
                "callback must be callable."
            )

        if not hasattr(
            self,
            "_subscribers",
        ):
            self._subscribers = []

        self._subscribers.append(
            callback
        )

    def emit_live(
        self,
        run_id: str,
        event_type: str,
        **data: Any,
    ) -> TraceEvent:

        event = self.emit(
            run_id,
            event_type,
            **data,
        )

        for callback in getattr(
            self,
            "_subscribers",
            [],
        ):
            callback(event)

        return event
