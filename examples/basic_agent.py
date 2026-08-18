from arc import Agent
from arc.models import NVIDIAModel


def main() -> None:
    agent = Agent(
        name="ARC Assistant",
        model=NVIDIAModel(),
        system_prompt=(
            "You are ARC Assistant, an intelligent AI agent "
            "built with the ARC Agent Framework."
        ),
    )

    result = agent.run(
        "Explain what an AI agent is in three sentences."
    )

    print(f"Agent: {agent.name}")
    print(f"Response: {result.output}")


if __name__ == "__main__":
    main()

    