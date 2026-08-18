import pytest

from arc import (
    BaseProvider,
    MockProvider,
    ProviderError,
    ProviderRegistry,
    ProviderResponse,
)


def test_mock_provider_creation():
    provider = MockProvider()

    assert provider.name == "mock"
    assert provider.model == "mock-model"


def test_mock_provider_custom_model():
    provider = MockProvider(
        model="test-model"
    )

    assert provider.model == "test-model"


def test_mock_generation():
    provider = MockProvider(
        response="Hello ARC"
    )

    result = provider.generate(
        "Hello"
    )

    assert isinstance(
        result,
        ProviderResponse,
    )

    assert result.success
    assert result.content == "Hello ARC"


def test_mock_provider_name():
    provider = MockProvider()

    result = provider.generate("test")

    assert result.provider == "mock"


def test_mock_provider_model():
    provider = MockProvider(
        model="arc-test"
    )

    result = provider.generate("test")

    assert result.model == "arc-test"


def test_mock_provider_calls():
    provider = MockProvider()

    provider.generate("one")
    provider.generate("two")

    assert provider.calls == 2


def test_mock_empty_prompt():
    provider = MockProvider()

    with pytest.raises(ValueError):
        provider.generate("")


def test_mock_invalid_prompt():
    provider = MockProvider()

    with pytest.raises(TypeError):
        provider.generate(None)


def test_provider_info():
    provider = MockProvider(
        model="test-model"
    )

    info = provider.info()

    assert info["name"] == "mock"
    assert info["model"] == "test-model"


@pytest.mark.asyncio
async def test_mock_async_generation():
    provider = MockProvider(
        response="Async ARC"
    )

    result = await provider.generate_async(
        "Hello"
    )

    assert result.success
    assert result.content == "Async ARC"


def test_registry_creation():
    registry = ProviderRegistry()

    assert registry.count() == 0
    assert registry.default is None


def test_registry_register():
    registry = ProviderRegistry()
    provider = MockProvider()

    registry.register(provider)

    assert registry.count() == 1
    assert registry.has("mock")


def test_registry_get():
    registry = ProviderRegistry()
    provider = MockProvider()

    registry.register(provider)

    assert registry.get() is provider
    assert registry.get("mock") is provider


def test_registry_default():
    registry = ProviderRegistry()

    registry.register(
        MockProvider(
            model="one"
        )
    )

    assert registry.default == "mock"


def test_registry_set_default():
    registry = ProviderRegistry()

    first = MockProvider(
        model="first"
    )

    second = MockProvider(
        model="second"
    )

    registry.register(
        first,
        name="first"
    )

    registry.register(
        second,
        name="second"
    )

    registry.set_default("second")

    assert registry.default == "second"
    assert registry.get().model == "second"


def test_registry_list():
    registry = ProviderRegistry()

    registry.register(
        MockProvider(),
        name="one"
    )

    registry.register(
        MockProvider(),
        name="two"
    )

    assert registry.list() == [
        "one",
        "two",
    ]


def test_registry_unknown():
    registry = ProviderRegistry()

    with pytest.raises(ProviderError):
        registry.get("missing")


def test_registry_duplicate():
    registry = ProviderRegistry()

    registry.register(
        MockProvider()
    )

    with pytest.raises(ProviderError):
        registry.register(
            MockProvider()
        )


def test_registry_invalid_provider():
    registry = ProviderRegistry()

    with pytest.raises(TypeError):
        registry.register("not-provider")


def test_registry_unregister():
    registry = ProviderRegistry()

    registry.register(
        MockProvider()
    )

    removed = registry.unregister(
        "mock"
    )

    assert removed.name == "mock"
    assert registry.count() == 0


def test_registry_clear():
    registry = ProviderRegistry()

    registry.register(
        MockProvider(),
        name="one"
    )

    registry.register(
        MockProvider(),
        name="two"
    )

    registry.clear()

    assert registry.count() == 0
    assert registry.default is None


def test_registry_generate():
    registry = ProviderRegistry()

    registry.register(
        MockProvider(
            response="ARC response"
        )
    )

    result = registry.generate(
        "Hello"
    )

    assert result.content == "ARC response"


@pytest.mark.asyncio
async def test_registry_async_generate():
    registry = ProviderRegistry()

    registry.register(
        MockProvider(
            response="ARC async"
        )
    )

    result = await registry.generate_async(
        "Hello"
    )

    assert result.content == "ARC async"


def test_multiple_providers():
    registry = ProviderRegistry()

    first = MockProvider(
        model="model-a"
    )

    second = MockProvider(
        model="model-b"
    )

    registry.register(
        first,
        name="provider-a"
    )

    registry.register(
        second,
        name="provider-b"
    )

    assert registry.get(
        "provider-a"
    ).model == "model-a"

    assert registry.get(
        "provider-b"
    ).model == "model-b"


def test_provider_response_text():
    response = ProviderResponse(
        content="Hello",
        provider="mock",
        model="test",
    )

    assert response.text == "Hello"


def test_invalid_model():
    with pytest.raises(Exception):
        MockProvider(model="")


def test_provider_metadata():
    provider = MockProvider()

    result = provider.generate("ARC")

    assert result.metadata["mock"] is True
    assert result.metadata["call_number"] == 1


def test_provider_usage():
    provider = MockProvider(
        response="Hello world"
    )

    result = provider.generate(
        "Hello ARC"
    )

    assert "prompt_tokens" in result.usage
    assert "completion_tokens" in result.usage
