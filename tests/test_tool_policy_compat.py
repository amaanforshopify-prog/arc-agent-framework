from arc import ToolIntelligence, ToolPolicy, tool


@tool()
def add(
    a: int,
    b: int,
) -> int:
    return a + b


def test_allowed_tools_api():
    policy = ToolPolicy(
        allowed_tools={
            "add",
            "search",
        }
    )

    assert policy.is_tool_allowed("add")
    assert policy.is_tool_allowed("search")
    assert not policy.is_tool_allowed("delete")


def test_denied_tools_override_allowed():
    policy = ToolPolicy(
        allowed_tools={
            "add",
            "search",
        },
        denied_tools={
            "search",
        },
    )

    assert policy.is_tool_allowed("add")
    assert not policy.is_tool_allowed("search")


def test_global_deny():
    policy = ToolPolicy(
        allowed=False
    )

    assert not policy.is_tool_allowed(
        "add"
    )


def test_disabled_policy():
    policy = ToolPolicy(
        enabled=False
    )

    assert not policy.is_tool_allowed(
        "add"
    )


def test_confirmation_callback():
    accepted = ToolPolicy(
        require_confirmation=True,
        confirmation_callback=(
            lambda name, args:
            name == "add"
            and args["a"] == 2
        ),
    )

    rejected = ToolPolicy(
        require_confirmation=True,
        confirmation_callback=lambda name, args: False,
    )

    assert not accepted.requires_confirmation_for(
        "add",
        {"a": 2},
    )

    assert rejected.requires_confirmation_for(
        "add",
        {"a": 2},
    )


def test_confirmation_without_callback():
    policy = ToolPolicy(
        require_confirmation=True
    )

    assert policy.requires_confirmation_for(
        "add",
        {},
    )


def test_policy_execution():
    intelligence = ToolIntelligence()

    result = intelligence.execute(
        "add",
        add.function,
        2,
        3,
        policy=ToolPolicy(
            allowed_tools={"add"}
        ),
    )

    assert result.success
    assert result.output == 5


def test_policy_denied_execution():
    intelligence = ToolIntelligence()

    result = intelligence.execute(
        "add",
        add.function,
        2,
        3,
        policy=ToolPolicy(
            denied_tools={"add"}
        ),
    )

    assert result.success is False


def test_confirmation_rejected():
    intelligence = ToolIntelligence()

    result = intelligence.execute(
        "add",
        add.function,
        2,
        3,
        policy=ToolPolicy(
            require_confirmation=True,
            confirmation_callback=lambda name, args: False,
        ),
    )

    assert result.success is False


def test_confirmation_accepted():
    intelligence = ToolIntelligence()

    result = intelligence.execute(
        "add",
        add.function,
        2,
        3,
        policy=ToolPolicy(
            require_confirmation=True,
            confirmation_callback=lambda name, args: True,
        ),
    )

    assert result.success
    assert result.output == 5



