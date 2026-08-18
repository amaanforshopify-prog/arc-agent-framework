from arc import (
    Agent,
    AgentRuntime,
    ContextManager,
    Memory,
    PlanningEngine,
)


def main():
    print("=" * 60)
    print("ARC AGENT RUNTIME")
    print("=" * 60)

    agent = Agent(
        name="RuntimeAgent",
        executor=PlanningEngine(),
    )

    memory = Memory()
    context = ContextManager()

    runtime = AgentRuntime(
        agent=agent,
        context=context,
        memory=memory,
    )

    print("\n[1] Running agent...")

    run = runtime.run(
        "Complete a research task",
        [
            "Collect information",
            "Process information",
            "Return final result",
        ],
    )

    print(f"Run ID: {run.run_id}")
    print(f"Success: {run.success}")
    print(f"Result: {run.result}")

    print("\n[2] Runtime state")

    print(f"Runs: {runtime.run_count()}")
    print(f"Context: {context.count()}")
    print(f"Memory used: {run.memory_used}")

    print("\n[3] Plan")

    for step in run.plan.steps:
        print(
            f"  {step.id}. "
            f"{step.description} "
            f"→ {step.status}"
        )

    print("\n[4] Snapshot")
    print(runtime.snapshot())

    print("\n" + "=" * 60)
    print("AGENT RUNTIME PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
