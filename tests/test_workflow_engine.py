from __future__ import annotations

import time

import pytest

from arc import (
    ConditionalStep,
    ParallelStep,
    WorkflowEngine,
    WorkflowEngineError,
    WorkflowState,
)


def test_workflow_state():
    state = WorkflowState()

    assert state.get("missing") is None
    assert state.has("x") is False

    state.set("x", 10)

    assert state.has("x")
    assert state.get("x") == 10
    assert state.snapshot() == {
        "x": 10,
    }

    state.update(
        {
            "y": 20,
            "z": 30,
        }
    )

    assert state.get("y") == 20
    assert state.get("z") == 30


def test_workflow_state_clear():
    state = WorkflowState(
        data={
            "a": 1,
            "b": 2,
        }
    )

    state.clear()

    assert state.snapshot() == {}


def test_invalid_state_key():
    state = WorkflowState()

    with pytest.raises(ValueError):
        state.set("", 1)


def test_conditional_true_branch():
    engine = WorkflowEngine()
    state = WorkflowState(
        {
            "score": 90
        }
    )

    step = ConditionalStep(
        name="score_check",
        condition=lambda s: (
            s.get("score", 0) >= 80
        ),
        when_true=lambda s: "PASS",
        when_false=lambda s: "FAIL",
    )

    result = engine.run_condition(
        step,
        state,
    )

    assert result.success
    assert result.result == "PASS"


def test_conditional_false_branch():
    engine = WorkflowEngine()
    state = WorkflowState(
        {
            "score": 50
        }
    )

    step = ConditionalStep(
        name="score_check",
        condition=lambda s: (
            s.get("score", 0) >= 80
        ),
        when_true=lambda s: "PASS",
        when_false=lambda s: "FAIL",
    )

    result = engine.run_condition(
        step,
        state,
    )

    assert result.success
    assert result.result == "FAIL"


def test_conditional_without_false():
    engine = WorkflowEngine()

    step = ConditionalStep(
        name="optional",
        condition=lambda s: False,
        when_true=lambda s: "yes",
    )

    result = engine.run_condition(step)

    assert result.success
    assert result.result is None


def test_conditional_failure():
    engine = WorkflowEngine()

    def broken_condition(state):
        raise RuntimeError("condition failed")

    step = ConditionalStep(
        name="broken",
        condition=broken_condition,
        when_true=lambda s: "x",
    )

    result = engine.run_condition(step)

    assert result.success is False
    assert isinstance(
        result.error,
        RuntimeError,
    )


def test_parallel_execution():
    engine = WorkflowEngine(
        max_workers=3
    )

    state = WorkflowState()

    step = ParallelStep(
        name="parallel_math",
        tasks={
            "a": lambda s: 2 + 3,
            "b": lambda s: 4 * 5,
            "c": lambda s: 10 - 7,
        },
    )

    results = engine.run_parallel(
        step,
        state,
    )

    assert results["a"].success
    assert results["b"].success
    assert results["c"].success

    assert results["a"].result == 5
    assert results["b"].result == 20
    assert results["c"].result == 3


def test_parallel_tasks_run_concurrently():
    engine = WorkflowEngine(
        max_workers=3
    )

    started = []

    def task(name):
        def run(state):
            started.append(name)
            time.sleep(0.05)
            return name
        return run

    step = ParallelStep(
        name="parallel",
        tasks={
            "one": task("one"),
            "two": task("two"),
            "three": task("three"),
        },
    )

    results = engine.run_parallel(step)

    assert len(results) == 3
    assert set(started) == {
        "one",
        "two",
        "three",
    }


def test_parallel_failure():
    engine = WorkflowEngine()

    def broken(state):
        raise ValueError("boom")

    step = ParallelStep(
        name="failure",
        tasks={
            "good": lambda s: "ok",
            "bad": broken,
        },
    )

    results = engine.run_parallel(
        step,
        stop_on_error=False,
    )

    assert results["good"].success
    assert results["bad"].success is False
    assert isinstance(
        results["bad"].error,
        ValueError,
    )


def test_execute_mixed_workflow():
    engine = WorkflowEngine()

    state = WorkflowState()

    condition = ConditionalStep(
        name="decision",
        condition=lambda s: True,
        when_true=lambda s: (
            s.set("branch", "A") or "A"
        ),
    )

    parallel = ParallelStep(
        name="parallel",
        tasks={
            "first": lambda s: (
                s.get("branch") + "-1"
            ),
            "second": lambda s: (
                s.get("branch") + "-2"
            ),
        },
    )

    outputs = engine.execute(
        [
            condition,
            parallel,
        ],
        state,
    )

    assert outputs[0] == "A"

    assert outputs[1]["first"].result == "A-1"
    assert outputs[1]["second"].result == "A-2"


def test_execute_stops_on_conditional_error():
    engine = WorkflowEngine()

    step = ConditionalStep(
        name="broken",
        condition=lambda s: (
            1 / 0
        ),
        when_true=lambda s: "x",
    )

    with pytest.raises(
        WorkflowEngineError,
        match="failed",
    ):
        engine.execute(
            [step]
        )


def test_execute_stops_on_parallel_error():
    engine = WorkflowEngine()

    def broken(state):
        raise RuntimeError(
            "parallel failure"
        )

    step = ParallelStep(
        name="broken_parallel",
        tasks={
            "bad": broken,
        },
    )

    with pytest.raises(
        WorkflowEngineError,
        match="failed",
    ):
        engine.execute(
            [step]
        )


def test_invalid_workers():
    with pytest.raises(ValueError):
        WorkflowEngine(
            max_workers=0
        )


def test_invalid_operation():
    engine = WorkflowEngine()

    with pytest.raises(TypeError):
        engine.execute(
            [object()]
        )
