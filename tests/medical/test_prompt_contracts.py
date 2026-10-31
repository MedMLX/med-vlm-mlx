"""Prompt ordering and native-template failures without models or downloads."""

from dataclasses import dataclass, field

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mlx_vlm.contracts import JSONObject, JSONValue
from mlx_vlm.prompt_utils import apply_chat_template, get_message_json


@dataclass
class RecordingTemplate:
    chat_template: str | None = "configured"
    messages: list[JSONObject] = field(default_factory=list[JSONObject])
    options: dict[str, JSONValue] = field(default_factory=dict[str, JSONValue])

    def apply_chat_template(
        self,
        messages: list[JSONObject],
        *,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
        **kwargs: JSONValue,
    ) -> str:
        self.messages = messages
        self.options = {
            **kwargs,
            "tokenize": tokenize,
            "generation": add_generation_prompt,
        }
        return "rendered"


@pytest.mark.parametrize("model", ["gemma3"])
@given(st.lists(st.text(max_size=20), min_size=3, max_size=3))
def test_ordered_image_parts_survive_rendering(model: str, texts: list[str]) -> None:
    prompt: list[JSONObject] = [
        {"type": "text", "text": texts[0]},
        {"type": "image"},
        {"type": "text", "text": texts[1]},
        {"type": "image"},
        {"type": "text", "text": texts[2]},
    ]
    message = get_message_json(model, prompt, num_images=2)
    assert message["content"] == "<start_of_image>".join(texts)


def test_tool_turns_and_definitions_are_rejected_before_rendering() -> None:
    processor = RecordingTemplate()
    call: JSONObject = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"function": {"name": "measure", "arguments": "{}"}}],
    }
    with pytest.raises(ValueError, match="Tool calling"):
        apply_chat_template(processor, {"model_type": "gemma3"}, [call])
    with pytest.raises(ValueError, match="Tool calling"):
        apply_chat_template(
            processor, {"model_type": "gemma3"}, "describe", tools=[{"type": "x"}]
        )
    assert processor.messages == []


@pytest.mark.parametrize("error", [AttributeError, ValueError])
def test_native_template_errors_propagate(error: type[Exception]) -> None:
    class BrokenTemplate(RecordingTemplate):
        def apply_chat_template(
            self,
            messages: list[JSONObject],
            *,
            add_generation_prompt: bool = True,
            tokenize: bool = False,
            **kwargs: JSONValue,
        ) -> str:
            raise error("native template failure")

    with pytest.raises(error, match="native template failure"):
        apply_chat_template(BrokenTemplate(), {"model_type": "gemma3"}, "Describe")


def test_plain_prompt_is_selected_only_without_a_configured_template() -> None:
    processor = RecordingTemplate(chat_template=None)
    assert (
        apply_chat_template(processor, {"model_type": "gemma3"}, "Describe")
        == "Describe"
    )
    assert not processor.messages
    assert (
        apply_chat_template(
            processor, {"model_type": "gemma3"}, "Describe", chat_template="override"
        )
        == "rendered"
    )
    assert processor.messages


@pytest.mark.parametrize(
    "message",
    [
        {"role": 7, "content": "text"},
        {"role": "user", "content": [{"type": "text", "text": 7}]},
        {"role": "user", "content": [{"type": "text", "text": False}]},
        {"role": "user", "content": [{"type": "text", "text": 0}]},
    ],
)
def test_invalid_roles_and_text_are_rejected(message: JSONObject) -> None:
    with pytest.raises(TypeError):
        apply_chat_template(None, {"model_type": "gemma3"}, message)
