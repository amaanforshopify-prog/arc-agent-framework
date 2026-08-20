from types import SimpleNamespace

from arc import (
    AgentWorkflowEngine,
    AgentWorkflowStep,
    DurableWorkflowState,
)


class FakeModel:
    def __init__(self):
        self.calls = 0

    def generate(self, messages, tools=None):
        self.calls += 1

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="ARC workflow response.",
                        tool_calls=[],
                    )
                )
            ]
        )


def test_agent_workflow_step():
    model = FakeModel()

    engine = AgentWorkflowEngine(
        model=model
    )

    state = DurableWorkflowState(
        db_path="data/test_agent_workflow.db",
        workflow_id="test",
    )

    step = AgentWorkflowStep(
        name="agent_step",
        system_prompt="You are ARC.",
        user_input="Say hello.",
    )

    result = engine.execute_step(
        step,
        state,
    )

    assert result == "ARC workflow response."
    assert state.get(
        "agent_output:agent_step"
    ) == "ARC workflow response."

    state.close()


def test_agent_workflow_checkpoint():
    model = FakeModel()

    engine = AgentWorkflowEngine(
        model=model
    )

    state = DurableWorkflowState(
        db_path="data/test_agent_workflow_checkpoint.db",
        workflow_id="checkpoint",
    )

    steps = [
        AgentWorkflowStep(
            name="step1",
            system_prompt="You are ARC.",
            user_input="First.",
        ),
        AgentWorkflowStep(
            name="step2",
            system_prompt="You are ARC.",
            user_input="Second.",
        ),
    ]

    result = engine.execute_checkpointed(
        steps,
        state,
    )

    assert result.success
    assert result.outputs == [
        "ARC workflow response.",
        "ARC workflow response.",
    ]

    assert state.completed_steps == [
        "step1",
        "step2",
    ]

    state.close()


def test_agent_workflow_resume():
    model = FakeModel()

    engine = AgentWorkflowEngine(
        model=model
    )

    db = "data/test_agent_workflow_resume.db"

    state = DurableWorkflowState(
        db_path=db,
        workflow_id="resume",
    )

    state.set(
        "result:step1",
        "already completed",
    )
    state.mark_completed("step1")
    state.save()
    state.close()

    state = DurableWorkflowState(
        db_path=db,
        workflow_id="resume",
    )

    steps = [
        AgentWorkflowStep(
            name="step1",
            system_prompt="You are ARC.",
            user_input="Should skip.",
        ),
        AgentWorkflowStep(
            name="step2",
            system_prompt="You are ARC.",
            user_input="Continue.",
        ),
    ]

    result = engine.execute_checkpointed(
        steps,
        state,
    )

    assert result.success
    assert result.resumed
    assert result.outputs == [
        "already completed",
        "ARC workflow response.",
    ]

    state.close()
