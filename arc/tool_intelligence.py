from __future__ import annotations

import inspect
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, get_type_hints

from .retry import RetryManager, RetryPolicy


RiskLevel = Literal[
    "low",
    "medium",
    "high",
    "critical",
]


class ToolIntelligenceError(Exception):
    """Base error for tool intelligence operations."""


class ToolApprovalError(ToolIntelligenceError):
    """Raised when a protected tool requires approval."""


@dataclass
class ToolPolicy:
    """Execution and security policy for tools."""

    enabled: bool = True
    allowed: bool = True

    allowed_tools: set[str] | None = None
    denied_tools: set[str] = field(
        default_factory=set
    )

    require_confirmation: bool = False
    confirmation_callback: Callable[
        [str, dict[str, Any]],
        bool,
    ] | None = None

    max_attempts: int = 1
    retry_delay: float = 0.0
    timeout: float | None = None

    risk_level: RiskLevel = "low"
    requires_approval: bool = False

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(
                "max_attempts must be >= 1"
            )

        if self.retry_delay < 0:
            raise ValueError(
                "retry_delay must be >= 0"
            )

        if self.timeout is not None and self.timeout <= 0:
            raise ValueError(
                "timeout must be > 0"
            )

        if self.allowed_tools is not None:
            self.allowed_tools = set(
                self.allowed_tools
            )

        self.denied_tools = set(
            self.denied_tools
        )

        if self.risk_level not in {
            "low",
            "medium",
            "high",
            "critical",
        }:
            raise ValueError(
                "invalid risk_level"
            )

    def is_tool_allowed(
        self,
        tool_name: str,
    ) -> bool:
        if not self.enabled:
            return False

        if not self.allowed:
            return False

        if (
            self.allowed_tools is not None
            and tool_name not in self.allowed_tools
        ):
            return False

        if tool_name in self.denied_tools:
            return False

        return True

    def confirm(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> bool:
        """Return whether a tool call is explicitly confirmed."""
        if not self.require_confirmation:
            return True

        if self.confirmation_callback is None:
            return False

        return bool(
            self.confirmation_callback(
                tool_name,
                arguments,
            )
        )

    def requires_confirmation_for(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> bool:
        if not self.require_confirmation:
            return False

        if self.confirmation_callback is None:
            return True

        return not bool(
            self.confirmation_callback(
                tool_name,
                arguments,
            )
        )


@dataclass
class ToolExecutionResult:
    """Structured result returned by safe tool execution."""

    success: bool
    output: Any = None
    error: Exception | None = None
    tool_name: str | None = None
    duration: float = 0.0
    attempts: int = 0
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class ToolExecutionRecord:
    """Historical record of a tool execution."""

    tool_name: str
    success: bool
    duration: float
    attempts: int
    error: str | None = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class ToolValidator:
    """Validates tool arguments against its callable signature."""

    def validate(
        self,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if not callable(function):
            raise TypeError(
                "function must be callable."
            )

        signature = inspect.signature(function)

        try:
            bound = signature.bind(
                *args,
                **kwargs,
            )
        except TypeError as exc:
            raise ToolIntelligenceError(
                f"Invalid tool arguments: {exc}"
            ) from exc

        try:
            hints = get_type_hints(function)
        except Exception:
            hints = {}

        for name, value in bound.arguments.items():
            annotation = hints.get(name)

            if annotation is None:
                continue

            if not self._matches(
                value,
                annotation,
            ):
                raise ToolIntelligenceError(
                    f"Invalid argument '{name}': "
                    f"expected {annotation}, "
                    f"got {type(value).__name__}."
                )

    def _matches(
        self,
        value: Any,
        annotation: Any,
    ) -> bool:
        origin = getattr(
            annotation,
            "__origin__",
            None,
        )

        if origin is None:
            try:
                return isinstance(
                    value,
                    annotation,
                )
            except TypeError:
                return True

        if origin is list:
            return isinstance(value, list)

        if origin is dict:
            return isinstance(value, dict)

        if origin is tuple:
            return isinstance(value, tuple)

        if origin is set:
            return isinstance(value, set)

        return True


class ToolIntelligence:
    """
    Safe tool execution layer.

    Handles:
    validation,
    permissions,
    approval,
    retry,
    timeout,
    timing,
    execution history.
    """

    def __init__(
        self,
        validator: ToolValidator | None = None,
    ) -> None:
        self.validator = (
            validator
            if validator is not None
            else ToolValidator()
        )

        self.records: list[
            ToolExecutionRecord
        ] = []

    def execute(
        self,
        tool_name: str,
        function: Callable[..., Any],
        *args: Any,
        policy: ToolPolicy | None = None,
        approved: bool = False,
        **kwargs: Any,
    ) -> ToolExecutionResult:
        policy = (
            policy
            if policy is not None
            else ToolPolicy()
        )

        started = time.perf_counter()
        attempts = 0

        try:
            if not policy.is_tool_allowed(
                tool_name
            ):
                raise ToolIntelligenceError(
                    f"Tool '{tool_name}' is not allowed."
                )

            arguments = dict(kwargs)

            if policy.requires_confirmation_for(
                tool_name,
                arguments,
            ):
                raise ToolApprovalError(
                    f"Tool '{tool_name}' requires confirmation."
                )

            if (
                policy.requires_approval
                and not approved
            ):
                raise ToolApprovalError(
                    f"Tool '{tool_name}' requires approval "
                    f"before execution."
                )

            self.validator.validate(
                function,
                *args,
                **kwargs,
            )

            retry_manager = RetryManager(
                RetryPolicy(
                    max_attempts=policy.max_attempts,
                    delay=policy.retry_delay,
                )
            )

            def call() -> Any:
                nonlocal attempts

                attempts += 1

                if policy.timeout is None:
                    return function(
                        *args,
                        **kwargs,
                    )

                with ThreadPoolExecutor(
                    max_workers=1
                ) as executor:
                    future = executor.submit(
                        function,
                        *args,
                        **kwargs,
                    )

                    try:
                        return future.result(
                            timeout=policy.timeout
                        )
                    except TimeoutError as exc:
                        raise ToolIntelligenceError(
                            f"Tool '{tool_name}' timed out "
                            f"after {policy.timeout}s."
                        ) from exc

            output = retry_manager.execute(
                call
            )

            duration = (
                time.perf_counter()
                - started
            )

            result = ToolExecutionResult(
                success=True,
                output=output,
                tool_name=tool_name,
                duration=duration,
                attempts=attempts,
                metadata={
                    "retries": max(
                        0,
                        attempts - 1,
                    ),
                    "risk_level": policy.risk_level,
                    "requires_approval": (
                        policy.requires_approval
                    ),
                    "approved": approved,
                    "policy_allowed": policy.is_tool_allowed(tool_name),
                    "allowed_tools": (
                        sorted(policy.allowed_tools)
                        if policy.allowed_tools is not None
                        else None
                    ),
                    "denied_tools": sorted(policy.denied_tools),
                    "requires_confirmation": policy.require_confirmation,
                    "confirmation_required": policy.require_confirmation,
                },
            )

        except Exception as exc:
            duration = (
                time.perf_counter()
                - started
            )

            result = ToolExecutionResult(
                success=False,
                error=exc,
                tool_name=tool_name,
                duration=duration,
                attempts=attempts,
                metadata={
                    "retries": max(
                        0,
                        attempts - 1,
                    ),
                    "risk_level": policy.risk_level,
                    "requires_approval": (
                        policy.requires_approval
                    ),
                    "approved": approved,
                    "policy_allowed": policy.is_tool_allowed(tool_name),
                    "allowed_tools": (
                        sorted(policy.allowed_tools)
                        if policy.allowed_tools is not None
                        else None
                    ),
                    "denied_tools": sorted(policy.denied_tools),
                    "requires_confirmation": policy.require_confirmation,
                    "confirmation_required": policy.require_confirmation,
                },
            )

        self.records.append(
            ToolExecutionRecord(
                tool_name=tool_name,
                success=result.success,
                duration=result.duration,
                attempts=result.attempts,
                error=(
                    str(result.error)
                    if result.error
                    else None
                ),
                metadata=dict(
                    result.metadata
                ),
            )
        )

        return result

    def history(
        self,
    ) -> list[ToolExecutionRecord]:
        return list(self.records)

    def clear_history(self) -> None:
        self.records.clear()


__all__ = [
    "RiskLevel",
    "ToolApprovalError",
    "ToolExecutionRecord",
    "ToolExecutionResult",
    "ToolIntelligence",
    "ToolIntelligenceError",
    "ToolPolicy",
    "ToolValidator",
]






