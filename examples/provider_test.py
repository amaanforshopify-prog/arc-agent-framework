from arc import MockProvider, ProviderRegistry


def main():
    print("=" * 60)
    print("ARC LLM PROVIDER SYSTEM")
    print("=" * 60)

    registry = ProviderRegistry()

    print("\n[1] Creating providers")

    mock = MockProvider(
        model="arc-dev-model",
        response="Hello from ARC Provider!"
    )

    registry.register(
        mock,
        name="development"
    )

    print("  PASS")

    print("\n[2] Provider registry")

    print(
        f"  Providers: "
        f"{registry.list()}"
    )

    print(
        f"  Default: "
        f"{registry.default}"
    )

    print("  PASS")

    print("\n[3] Generate response")

    response = registry.generate(
        "What is ARC?"
    )

    print(
        f"  Provider: "
        f"{response.provider}"
    )

    print(
        f"  Model: "
        f"{response.model}"
    )

    print(
        f"  Response: "
        f"{response.content}"
    )

    print("  PASS")

    print("\n[4] Usage")

    print(
        f"  Prompt tokens: "
        f"{response.usage['prompt_tokens']}"
    )

    print(
        f"  Completion tokens: "
        f"{response.usage['completion_tokens']}"
    )

    print("  PASS")

    print("\n[5] Provider switching")

    second = MockProvider(
        model="second-model",
        response="Second provider works!"
    )

    registry.register(
        second,
        name="second"
    )

    registry.set_default("second")

    response = registry.generate(
        "Test"
    )

    print(
        f"  Active: "
        f"{response.model}"
    )

    assert response.model == "second-model"

    print("  PASS")

    print("\n" + "=" * 60)
    print("ARC PROVIDER SYSTEM PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
