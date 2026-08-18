import time

import pytest

from arc import (
    Trace,
    TraceError,
    TraceManager,
)


def test_trace_creation():
    trace = Trace("run-1")

    assert trace.run_id == "run-1"
    assert trace.count() == 0


def test_emit():
    trace = Trace("run-1")

    event = trace.emit(
        "planning",
        goal="test",
    )

    assert event.event_type == "planning"
    assert event.data["goal"] == "test"
    assert trace.count() == 1


def test_event_has_id():
    trace = Trace("run-1")

    event = trace.emit("test")

    assert event.event_id
    assert event.run_id == "run-1"


def test_empty_event_type():
    trace = Trace("run-1")

    with pytest.raises(ValueError):
        trace.emit("")


def test_finish():
    trace = Trace("run-1")

    trace.finish()

    assert trace.finished_at is not None


def test_duration():
    trace = Trace("run-1")

    time.sleep(0.01)

    assert trace.duration > 0


def test_filter():
    trace = Trace("run-1")

    trace.emit("tool")
    trace.emit("planning")
    trace.emit("tool")

    assert len(trace.filter("tool")) == 2
    assert len(trace.filter("planning")) == 1


def test_last():
    trace = Trace("run-1")

    trace.emit("first")
    trace.emit("second")

    assert trace.last().event_type == "second"


def test_last_filtered():
    trace = Trace("run-1")

    trace.emit("tool")
    trace.emit("planning")
    trace.emit("tool")

    assert trace.last("tool").event_type == "tool"


def test_last_empty():
    trace = Trace("run-1")

    assert trace.last() is None


def test_snapshot():
    trace = Trace("run-1")

    trace.emit(
        "test",
        value=123,
    )

    snapshot = trace.snapshot()

    assert snapshot["run_id"] == "run-1"
    assert snapshot["event_count"] == 1
    assert snapshot["events"][0]["data"]["value"] == 123


def test_manager_creation():
    manager = TraceManager()

    assert manager.list_runs() == []


def test_manager_start():
    manager = TraceManager()

    trace = manager.start("run-1")

    assert trace.run_id == "run-1"
    assert manager.get("run-1") is trace
    assert "run-1" in manager.active


def test_start_generates_id():
    manager = TraceManager()

    trace = manager.start()

    assert trace.run_id.startswith("run-")


def test_duplicate_run():
    manager = TraceManager()

    manager.start("run-1")

    with pytest.raises(TraceError):
        manager.start("run-1")


def test_manager_emit():
    manager = TraceManager()

    manager.start("run-1")

    event = manager.emit(
        "run-1",
        "tool_called",
        tool="search",
    )

    assert event.data["tool"] == "search"


def test_unknown_emit():
    manager = TraceManager()

    with pytest.raises(TraceError):
        manager.emit(
            "missing",
            "test",
        )


def test_manager_finish():
    manager = TraceManager()

    manager.start("run-1")
    trace = manager.finish("run-1")

    assert trace.finished_at is not None
    assert "run-1" not in manager.active
    assert trace.last().event_type == "run_completed"


def test_unknown_finish():
    manager = TraceManager()

    with pytest.raises(TraceError):
        manager.finish("missing")


def test_list_runs():
    manager = TraceManager()

    manager.start("one")
    manager.start("two")

    assert manager.list_runs() == [
        "one",
        "two",
    ]


def test_remove():
    manager = TraceManager()

    manager.start("run-1")

    assert manager.remove("run-1")
    assert manager.get("run-1") is None


def test_remove_missing():
    manager = TraceManager()

    assert not manager.remove("missing")


def test_clear():
    manager = TraceManager()

    manager.start("one")
    manager.start("two")

    manager.clear()

    assert manager.list_runs() == []


def test_subscriber():
    manager = TraceManager()

    received = []

    manager.subscribe(
        lambda event: received.append(event)
    )

    manager.start("run-1")

    manager.emit_live(
        "run-1",
        "tool_called",
        tool="calculator",
    )

    assert len(received) == 1
    assert received[0].event_type == "tool_called"


def test_invalid_subscriber():
    manager = TraceManager()

    with pytest.raises(TypeError):
        manager.subscribe("invalid")


def test_trace_started_event():
    manager = TraceManager()

    trace = manager.start("run-1")

    assert (
        trace.events[0].event_type
        == "run_started"
    )


def test_complete_lifecycle():
    manager = TraceManager()

    trace = manager.start("run-1")

    manager.emit(
        "run-1",
        "planning",
    )

    manager.emit(
        "run-1",
        "tool_called",
        tool="test",
    )

    manager.finish("run-1")

    assert trace.count() == 4
    assert trace.finished_at is not None
