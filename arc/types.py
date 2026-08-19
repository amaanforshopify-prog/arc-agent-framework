from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolFunction:
    name: str
    arguments: str


@dataclass
class ToolCall:
    id: str
    function: ToolFunction


@dataclass
class Message:
    role: Role
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class AgentResult:
    output: str
    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
