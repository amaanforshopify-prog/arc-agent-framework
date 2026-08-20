from pathlib import Path
from types import SimpleNamespace

from arc import (
    AgentSpec,
    DurableWorkflowState,
    MultiAgentCoordinator,
)


class FakeModel:
    def generate(
        self,
        messages,
        tools=None,
    ):
        user_messages = [
            message.content
            for message in messages
            if getattr(
                message,
                "role",
                None,
            ) == "user"
        ]

        user = user_messages[-1]

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=f"done:{user}",
                        tool_calls=[],
                    )
                )
            ]
        )


def test_multi_agent_sequential(
    tmp_path: Path,
):
    state = DurableWorkflowState(
        str(tmp_path / "seq.db"),
        "seq",
    )

    coordinator = MultiAgentCoordinator(
        FakeModel()
    )

    agents = [
        AgentSpec(
            "research",
            "You research.",
            "research topic",
        ),
        AgentSpec(
            "writer",
            "You write.",
            "write result",
        ),
    ]

    result = coordinator.run_sequential(
        agents,
        state,
    )

    assert result.success
    assert [
        item.output
        for item in result.results
    ] == [
        "done:research topic",
        "done:write result",
    ]

    assert state.completed_steps == [
        "research",
        "writer",
    ]

    state.close()


def test_multi_agent_parallel(
    tmp_path: Path,
):
    state = DurableWorkflowState(
        str(tmp_path / "parallel.db"),
        "parallel",
    )

    coordinator = MultiAgentCoordinator(
        FakeModel(),
        max_workers=2,
    )

    agents = [
        AgentSpec(
            "a",
            "system",
            "one",
        ),
        AgentSpec(
            "b",
            "system",
            "two",
        ),
    ]

    result = coordinator.run_parallel(
        agents,
        state,
    )

    assert result.success
    assert {
        item.name
        for item in result.results
    } == {
        "a",
        "b",
    }

    assert state.is_completed("a")
    assert state.is_completed("b")

    state.close()


def test_multi_agent_resume(
    tmp_path: Path,
):
    db = str(
        tmp_path / "resume.db"
    )

    state = DurableWorkflowState(
        db,
        "resume",
    )

    state.set(
        "result:a",
        "cached",
    )

    state.mark_completed("a")
    state.save()
    state.close()

    state = DurableWorkflowState(
        db,
        "resume",
    )

    coordinator = MultiAgentCoordinator(
        FakeModel()
    )

    result = coordinator.run_sequential(
        [
            AgentSpec(
                "a",
                "system",
                "skip",
            ),
            AgentSpec(
                "b",
                "system",
                "run",
            ),
        ],
        state,
    )

    assert result.success
    assert result.resumed

    assert (
        result.results[0].output
        == "cached"
    )

    assert (
        result.results[1].output
        == "done:run"
    )

    state.close()
