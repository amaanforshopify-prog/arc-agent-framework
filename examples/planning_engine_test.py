from arc import (
    PlanningEngine,
    Tool,
    ToolRegistry,
)


def add(a, b):
    return a + b


def multiply(a, b):
    return a * b


def main():
    print("=" * 60)
    print("ARC PLANNING ENGINE")
    print("=" * 60)

    registry = ToolRegistry()

    registry.register(
        Tool("add", "Add two numbers", add)
    )

    registry.register(
        Tool("multiply", "Multiply two numbers", multiply)
    )

    engine = PlanningEngine(
        tools=registry
    )

    print("\n[1] Creating plan")

    plan = engine.create_plan(
        "Calculate values"
    )

    engine.add_tool_step(
        plan,
        "Add 10 + 20",
        "add",
        10,
        20,
    )

    engine.add_tool_step(
        plan,
        "Multiply 5 × 6",
        "multiply",
        5,
        6,
    )

    for step in plan.steps:
        print(
            f"  Step {step.id}: "
            f"{step.description}"
        )

    print("\n[2] Executing plan")

    result = engine.execute(plan)

    for item in result.results:
        print(
            f"  Step {item.step_id}: "
            f"{item.result}"
        )

    print("\n[3] Final state")

    print(
        f"  Success: {result.success}"
    )

    print(
        f"  Completed: "
        f"{result.completed_steps}"
    )

    print(
        f"  Failed: "
        f"{result.failed_steps}"
    )

    print(
        f"  Progress: "
        f"{plan.progress() * 100:.0f}%"
    )

    assert result.success

    print("\n" + "=" * 60)
    print("PLANNING ENGINE PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
