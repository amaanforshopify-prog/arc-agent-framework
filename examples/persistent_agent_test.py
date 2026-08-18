from arc import Agent
from arc.models import NVIDIAModel
from arc.persistent_memory import PersistentMemory


def main() -> None:
    memory = PersistentMemory(
        "data/persistent_agent.db"
    )

    agent = Agent(
        name="ARC Persistent Agent",
        model=NVIDIAModel(),
        system_prompt=(
            "You are ARC Persistent Agent. "
            "Remember information from previous "
            "conversations and use it when relevant."
        ),
        persistent_memory=memory,
    )

    print("Stored messages before run:", memory.count())

    result = agent.run(
        "My favorite programming language is Python."
    )

    print()
    print("Agent:", result.output)

    print()
    print(
        "Stored messages after run:",
        memory.count(),
    )

    memory.close()


if __name__ == "__main__":
    main()