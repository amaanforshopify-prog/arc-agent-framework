import json

from arc.models import NVIDIAModel
from arc.tools import FunctionTool, ToolRegistry
from arc.types import Message


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

    raise ValueError(
        f"Unknown operation: {operation}"
    )


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

    model = NVIDIAModel()

    messages = [
        Message(
            role="user",
            content=(
                "Calculate 784 multiplied by 92. "
                "Use the calculator tool."
            ),
        )
    ]

    response = model.generate(
        messages,
        tools=registry.schemas(),
    )

    message = response.choices[0].message

    print("Model:", response.model)
    print("Content:", message.content)
    print("Tool calls:", message.tool_calls)

    if message.tool_calls:
        tool_call = message.tool_calls[0]

        arguments = json.loads(
            tool_call.function.arguments
        )

        print("Tool name:", tool_call.function.name)
        print("Arguments:", arguments)

        result = registry.execute(
            tool_call.function.name,
            **arguments,
        )

        print("Tool result:", result)


if __name__ == "__main__":
    main()