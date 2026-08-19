from pathlib import Path

from arc import (
    CheckpointedWorkflowEngine,
    ConditionalStep,
    DurableWorkflowState,
    ParallelStep,
)


def test_checkpointed_workflow(tmp_path: Path):
    db = tmp_path / "workflow.db"

    state = DurableWorkflowState(
        db_path=str(db),
        workflow_id="demo",
    )

    engine = CheckpointedWorkflowEngine()

    operations = [
        ConditionalStep(
            name="approve",
            condition=lambda s: True,
            when_true=lambda s: "approved",
        ),
        ConditionalStep(
            name="finalize",
            condition=lambda s: True,
            when_true=lambda s: "done",
        ),
    ]

    result = engine.execute_checkpointed(
        operations,
        state,
    )

    assert result.success
    assert result.outputs == [
        "approved",
        "done",
    ]

    assert state.completed_steps == [
        "approve",
        "finalize",
    ]

    state.close()


def test_checkpoint_resume_skips_completed(
    tmp_path: Path,
):
    db = tmp_path / "resume.db"

    state = DurableWorkflowState(
        db_path=str(db),
        workflow_id="resume",
    )

    state.set(
        "result:step1",
        "already done",
    )
    state.mark_completed("step1")
    state.save()
    state.close()

    state = DurableWorkflowState(
        db_path=str(db),
        workflow_id="resume",
    )

    engine = CheckpointedWorkflowEngine()

    counter = {
        "step1": 0,
        "step2": 0,
    }

    operations = [
        ConditionalStep(
            name="step1",
            condition=lambda s: (
                counter.__setitem__(
                    "step1",
                    counter["step1"] + 1,
                )
                or True
            ),
            when_true=lambda s: "new",
        ),
        ConditionalStep(
            name="step2",
            condition=lambda s: (
                counter.__setitem__(
                    "step2",
                    counter["step2"] + 1,
                )
                or True
            ),
            when_true=lambda s: "second",
        ),
    ]

    result = engine.execute_checkpointed(
        operations,
        state,
    )

    assert result.success
    assert result.resumed
    assert counter["step1"] == 0
    assert counter["step2"] == 1
    assert result.outputs == [
        "already done",
        "second",
    ]

    state.close()


def test_checkpoint_failure_persists(
    tmp_path: Path,
):
    db = tmp_path / "failure.db"

    state = DurableWorkflowState(
        db_path=str(db),
        workflow_id="failure",
    )

    engine = CheckpointedWorkflowEngine()

    def broken(state):
        raise RuntimeError("boom")

    operations = [
        ConditionalStep(
            name="broken",
            condition=broken,
            when_true=lambda s: "never",
        )
    ]

    result = engine.execute_checkpointed(
        operations,
        state,
    )

    assert result.success is False
    assert result.error is not None
    assert state.current_step == "broken"
    assert state.metadata["failed_step"] == "broken"

    state.close()


def test_checkpoint_parallel(
    tmp_path: Path,
):
    db = tmp_path / "parallel.db"

    state = DurableWorkflowState(
        db_path=str(db),
        workflow_id="parallel",
    )

    engine = CheckpointedWorkflowEngine(
        max_workers=3,
    )

    operation = ParallelStep(
        name="parallel",
        tasks={
            "a": lambda s: 1,
            "b": lambda s: 2,
            "c": lambda s: 3,
        },
    )

    result = engine.execute_checkpointed(
        [operation],
        state,
    )

    assert result.success
    assert result.outputs[0] == {
        "a": 1,
        "b": 2,
        "c": 3,
    }

    assert state.is_completed(
        "parallel"
    )

    state.close()
