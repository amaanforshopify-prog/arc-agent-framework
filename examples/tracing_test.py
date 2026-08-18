from arc import Agent
from arc.models import NVIDIAModel
from arc.tools import FunctionTool, ToolRegistry
from arc.tracing import Tracer


def calculate(
    a: float,
    b: float,
    operation: str,
) -> float:
    if operation == "add":
        return a + b

    if operation == "subtract":
        return a - b

    if operation == "multiply":
        return a * b

    if operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero.")

        return a / b

    raise ValueError(f"Unknown operation: {operation}")


def main() -> None:
    calculator = FunctionTool(
        name="calculator",
        description="Perform basic arithmetic calculations.",
        parameters={
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "First number.",
                },
                "b": {
                    "type": "number",
                    "description": "Second number.",
                },
                "operation": {
                    "type": "string",
                    "enum": [
                        "add",
                        "subtract",
                        "multiply",
                        "divide",
                    ],
                    "description": "Arithmetic operation.",
                },
            },
            "required": [
                "a",
                "b",
                "operation",
            ],
        },
        function=calculate,
    )

    registry = ToolRegistry()
    registry.register(calculator)

    tracer = Tracer()

    agent = Agent(
        name="ARC Calculator Agent",
        model=NVIDIAModel(),
        system_prompt=(
            "You are an AI agent. "
            "Use the calculator tool whenever arithmetic "
            "calculation is required."
        ),
        tools=registry,
        tracer=tracer,
    )

    result = agent.run(
        "What is 784 multiplied by 92?"
    )

    print()
    print("========== FINAL RESULT ==========")
    print(result.output)

    print()
    print("========== TRACE SUMMARY ==========")

    for event in tracer.get_events():
        print(
            f"{event.event_type.upper():<8} "
            f"{event.message}"
        )


if __name__ == "__main__":
    main()