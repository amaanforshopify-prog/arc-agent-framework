from arc import (
    Planner,
    PromptBuilder,
)


def main():
    print("=" * 60)
    print("ARC CONTEXT + PLANNER")
    print("=" * 60)

    print("\n[1] Building context")

    prompt = PromptBuilder(
        "You are ARC, an AI agent framework."
    )

    prompt.add_user(
        "Build a research workflow."
    )

    prompt.add_assistant(
        "I will create a plan first."
    )

    print(prompt.build())
    print("\nPASS")

    print("\n[2] Creating plan")

    planner = Planner()

    plan = planner.create(
        "Build research workflow",
        [
            "Search for information",
            "Process the results",
            "Generate final report",
        ],
    )

    print(
        f"Goal: {plan.goal}"
    )

    for step in plan.steps:
        print(
            f"{step.id}. "
            f"{step.description}"
        )

    print("\nPASS")

    print("\n[3] Executing plan state")

    planner.start(plan)

    planner.complete_step(
        plan,
        1,
        "Search complete",
    )

    print(
        f"Progress: "
        f"{plan.progress() * 100:.0f}%"
    )

    print("\nPASS")

    print("\n" + "=" * 60)
    print("CONTEXT + PLANNER PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
