from arc import ToolIntelligence, ToolPolicy


def add(a: int, b: int) -> int:
    return a + b


def test_allowed_tools_execute():
    intelligence = ToolIntelligence()

    result = intelligence.execute(
        "add",
        add,
        2,
        3,
        policy=ToolPolicy(
            allowed_tools={"add"}
        ),
    )

    assert result.success
    assert result.output == 5


def test_denied_tool_fails():
    intelligence = ToolIntelligence()

    result = intelligence.execute(
        "add",
        add,
        2,
        3,
        policy=ToolPolicy(
            denied_tools={"add"}
        ),
    )

    assert not result.success
    assert "not allowed" in str(
        result.error
    )


def test_confirmation_is_enforced():
    intelligence = ToolIntelligence()

    result = intelligence.execute(
        "add",
        add,
        2,
        3,
        policy=ToolPolicy(
            require_confirmation=True,
            confirmation_callback=lambda name, args: False,
        ),
    )

    assert not result.success
    assert "confirmation" in str(
        result.error
    )


def test_confirmation_can_allow():
    intelligence = ToolIntelligence()

    result = intelligence.execute(
        "add",
        add,
        2,
        3,
        policy=ToolPolicy(
            require_confirmation=True,
            confirmation_callback=lambda name, args: True,
        ),
    )

    assert result.success
    assert result.output == 5


def test_policy_metadata():
    intelligence = ToolIntelligence()

    result = intelligence.execute(
        "add",
        add,
        1,
        1,
        policy=ToolPolicy(
            allowed_tools={"add"}
        ),
    )

    assert result.metadata["policy_allowed"] is True
    assert "confirmation_required" in result.metadata
