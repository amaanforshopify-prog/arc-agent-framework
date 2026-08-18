from arc import Agent, Executor, Memory, ToolRegistry, tool


@tool()
def add(a, b):
    return a + b


def main():
    registry = ToolRegistry()
    registry.register(add)

    executor = Executor(
        tools=registry
    )

    memory = Memory()

    agent = Agent(
        name="MemoryAgent",
        executor=executor,
        memory=memory,
    )

    print("=" * 60)
    print("ARC AGENT + MEMORY INTEGRATION")
    print("=" * 60)

    print("\n[1] Store manual memory")

    agent.remember(
        "project",
        "ARC AI Agent Framework",
    )

    print(
        f"  Recall: "
        f"{agent.recall('project')}"
    )

    assert (
        agent.recall("project")
        == "ARC AI Agent Framework"
    )

    print("  PASS")

    print("\n[2] Execute and remember result")

    result = agent.run(
        "Calculate 10 + 20",
        "add",
        10,
        20,
        remember_as="last_calculation",
    )

    print(f"  Result: {result.output}")

    assert result.success
    assert (
        agent.recall("last_calculation")
        == 30
    )

    print("  PASS")

    print("\n[3] Agent statistics")

    stats = agent.stats()

    print(f"  Runs: {stats['total_runs']}")
    print(
        f"  Memory: "
        f"{stats['memory_entries']}"
    )

    assert stats["total_runs"] == 1
    assert stats["memory_entries"] == 2

    print("  PASS")

    print("\n[4] Search agent memory")

    results = agent.memory.search("ARC")

    print(f"  Matches: {len(results)}")

    assert len(results) >= 1

    for item in results:
        print(
            f"  - {item.key}: "
            f"{item.value}"
        )

    print("  PASS")

    executor.shutdown()

    print("\n" + "=" * 60)
    print("AGENT + MEMORY INTEGRATION PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
