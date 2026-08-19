from unittest.mock import MagicMock

from arc.models import NVIDIAModel
from arc.types import Message, ToolCall, ToolFunction


def test_nvidia_model_serializes_tool_calls():
    model = object.__new__(NVIDIAModel)

    message = Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(
                id="call_1",
                function=ToolFunction(
                    name="add",
                    arguments='{"a":1,"b":2}',
                ),
            )
        ],
    )

    payload = model._message_to_api(
        message
    )

    assert payload["role"] == "assistant"
    assert payload["tool_calls"][0]["id"] == "call_1"
    assert (
        payload["tool_calls"][0]["type"]
        == "function"
    )
    assert (
        payload["tool_calls"][0]["function"]["name"]
        == "add"
    )
    assert (
        payload["tool_calls"][0]["function"]["arguments"]
        == '{"a":1,"b":2}'
    )


def test_nvidia_model_serializes_tool_message():
    model = object.__new__(NVIDIAModel)

    message = Message(
        role="tool",
        content="3",
        tool_call_id="call_1",
        name="add",
    )

    payload = model._message_to_api(
        message
    )

    assert payload["role"] == "tool"
    assert payload["content"] == "3"
    assert payload["tool_call_id"] == "call_1"
    assert payload["name"] == "add"


def test_nvidia_model_message_builder():
    model = object.__new__(NVIDIAModel)

    messages = [
        Message(
            role="system",
            content="You are ARC.",
        ),
        Message(
            role="user",
            content="Hello.",
        ),
    ]

    result = model._build_messages(
        messages
    )

    assert result == [
        {
            "role": "system",
            "content": "You are ARC.",
        },
        {
            "role": "user",
            "content": "Hello.",
        },
    ]


def test_nvidia_model_accepts_custom_retry_manager():
    model = object.__new__(NVIDIAModel)

    retry = MagicMock()

    model.retry_manager = retry

    assert model.retry_manager is retry
