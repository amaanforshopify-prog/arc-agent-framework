from arc import ToolPolicy


def test_allowed_tool_list():
    policy = ToolPolicy(
        allowed_tools={"add", "search"}
    )

    assert policy.is_tool_allowed("add")
    assert policy.is_tool_allowed("search")
    assert not policy.is_tool_allowed("delete")


def test_denied_tool_overrides_allowed():
    policy = ToolPolicy(
        allowed_tools={"add", "search"},
        denied_tools={"search"},
    )

    assert policy.is_tool_allowed("add")
    assert not policy.is_tool_allowed("search")


def test_global_deny():
    policy = ToolPolicy(
        allowed=False
    )

    assert not policy.is_tool_allowed("add")


def test_disabled_policy():
    policy = ToolPolicy(
        enabled=False
    )

    assert not policy.is_tool_allowed("add")


def test_confirmation_accept():
    policy = ToolPolicy(
        require_confirmation=True,
        confirmation_callback=lambda name, args: (
            name == "add"
            and args["a"] == 2
        ),
    )

    assert policy.confirm(
        "add",
        {"a": 2},
    )


def test_confirmation_reject():
    policy = ToolPolicy(
        require_confirmation=True,
        confirmation_callback=lambda name, args: False,
    )

    assert not policy.confirm(
        "add",
        {"a": 2},
    )


def test_confirmation_without_callback():
    policy = ToolPolicy(
        require_confirmation=True
    )

    assert not policy.confirm(
        "add",
        {},
    )


def test_policy_validation():
    import pytest

    with pytest.raises(ValueError):
        ToolPolicy(max_attempts=0)

    with pytest.raises(ValueError):
        ToolPolicy(retry_delay=-1)

    with pytest.raises(ValueError):
        ToolPolicy(timeout=0)
