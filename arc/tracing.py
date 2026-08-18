from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class TraceEvent:
    """A single event in an agent execution."""

    event_type: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )


class Tracer:
    """Collects and prints agent execution events."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.events: list[TraceEvent] = []

    def record(
        self,
        event_type: str,
        message: str,
        **data: Any,
    ) -> None:
        if not self.enabled:
            return

        event = TraceEvent(
            event_type=event_type,
            message=message,
            data=data,
        )

        self.events.append(event)

        print(
            f"[{event_type.upper()}] {message}"
        )

    def clear(self) -> None:
        self.events.clear()

    def get_events(self) -> list[TraceEvent]:
        return list(self.events)