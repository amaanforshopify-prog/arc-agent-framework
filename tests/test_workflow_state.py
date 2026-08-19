from pathlib import Path

import pytest

from arc import DurableWorkflowState


def test_creation(tmp_path: Path):
    state = DurableWorkflowState(
        db_path=str(tmp_path / "state.db"),
        workflow_id="workflow-1",
    )

    assert state.workflow_id == "workflow-1"
    assert state.data == {}
    assert state.completed_steps == []
    assert state.current_step is None

    state.close()


def test_set_get(tmp_path: Path):
    state = DurableWorkflowState(
        db_path=str(tmp_path / "state.db"),
        workflow_id="workflow",
    )

    state.set("answer", 42)

    assert state.get("answer") == 42
    assert state.get("missing") is None

    state.close()


def test_update(tmp_path: Path):
    state = DurableWorkflowState(
        db_path=str(tmp_path / "state.db"),
        workflow_id="workflow",
    )

    state.update({"a": 1, "b": 2})

    assert state.get("a") == 1
    assert state.get("b") == 2

    state.close()


def test_begin_step(tmp_path: Path):
    state = DurableWorkflowState(
        db_path=str(tmp_path / "state.db"),
        workflow_id="workflow",
    )

    state.begin_step("research")

    assert state.current_step == "research"

    state.close()


def test_mark_completed(tmp_path: Path):
    state = DurableWorkflowState(
        db_path=str(tmp_path / "state.db"),
        workflow_id="workflow",
    )

    state.begin_step("research")
    state.mark_completed("research")

    assert state.is_completed("research")
    assert state.current_step is None

    state.close()


def test_duplicate_completion(tmp_path: Path):
    state = DurableWorkflowState(
        db_path=str(tmp_path / "state.db"),
        workflow_id="workflow",
    )

    state.mark_completed("research")
    state.mark_completed("research")

    assert state.completed_steps == ["research"]

    state.close()


def test_persistence(tmp_path: Path):
    db = tmp_path / "persist.db"

    first = DurableWorkflowState(
        db_path=str(db),
        workflow_id="workflow",
    )

    first.set("status", "running")
    first.metadata["owner"] = "ARC"
    first.mark_completed("step-1")
    first.save()
    first.close()

    second = DurableWorkflowState(
        db_path=str(db),
        workflow_id="workflow",
    )

    assert second.get("status") == "running"
    assert second.is_completed("step-1")
    assert second.metadata["owner"] == "ARC"

    second.close()


def test_snapshot(tmp_path: Path):
    state = DurableWorkflowState(
        db_path=str(tmp_path / "snapshot.db"),
        workflow_id="workflow",
    )

    state.set("value", 100)
    state.mark_completed("step-a")

    snapshot = state.snapshot()

    assert snapshot["workflow_id"] == "workflow"
    assert snapshot["data"]["value"] == 100
    assert snapshot["completed_steps"] == ["step-a"]

    state.close()


def test_reset(tmp_path: Path):
    db = tmp_path / "reset.db"

    state = DurableWorkflowState(
        db_path=str(db),
        workflow_id="workflow",
    )

    state.set("value", 100)
    state.mark_completed("step-a")
    state.save()
    state.reset()

    assert state.data == {}
    assert state.completed_steps == []
    assert state.current_step is None

    state.close()


def test_reset_is_persistent(tmp_path: Path):
    db = tmp_path / "reset.db"

    first = DurableWorkflowState(
        db_path=str(db),
        workflow_id="workflow",
    )

    first.set("value", 100)
    first.save()
    first.reset()
    first.close()

    second = DurableWorkflowState(
        db_path=str(db),
        workflow_id="workflow",
    )

    assert second.data == {}
    assert second.completed_steps == []

    second.close()


def test_invalid_key(tmp_path: Path):
    state = DurableWorkflowState(
        db_path=str(tmp_path / "invalid.db"),
        workflow_id="workflow",
    )

    with pytest.raises(ValueError):
        state.set("", 1)

    state.close()


def test_invalid_step(tmp_path: Path):
    state = DurableWorkflowState(
        db_path=str(tmp_path / "invalid.db"),
        workflow_id="workflow",
    )

    with pytest.raises(ValueError):
        state.begin_step("")

    with pytest.raises(ValueError):
        state.mark_completed("")

    state.close()


def test_workflow_isolation(tmp_path: Path):
    db = tmp_path / "multi.db"

    first = DurableWorkflowState(
        db_path=str(db),
        workflow_id="first",
    )

    second = DurableWorkflowState(
        db_path=str(db),
        workflow_id="second",
    )

    first.set("value", "one")
    first.save()

    second.set("value", "two")
    second.save()

    first.close()
    second.close()

    first = DurableWorkflowState(
        db_path=str(db),
        workflow_id="first",
    )

    second = DurableWorkflowState(
        db_path=str(db),
        workflow_id="second",
    )

    assert first.get("value") == "one"
    assert second.get("value") == "two"

    first.close()
    second.close()
