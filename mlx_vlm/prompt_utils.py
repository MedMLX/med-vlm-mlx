"""Preserve medical image ordering while rendering native chat templates."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from transformers import BatchEncoding, BatchFeature

from mlx_vlm.contracts import (
    InputTensor,
    JSONObject,
    JSONValue,
    ProcessedInputs,
    is_json_object,
)
from mlx_vlm.models.base import BaseModelConfig
from mlx_vlm.models.protocols import GenerationConfigLike

SUPPORTED_PROMPT_MODELS = ("gemma3",)
_VIDEO_PART_TYPES = ("video", "input_video", "video_url")
_AUDIO_PART_TYPES = ("audio", "input_audio")
type ChatTemplateOutput = str | list[int] | list[
    list[int]
] | InputTensor | ProcessedInputs | BatchEncoding | BatchFeature


@runtime_checkable
class TemplateProcessor(Protocol):
    """Chat-template renderer accepting message dictionaries."""

    def apply_chat_template(
        self,
        messages: list[JSONObject],
        *,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
        **kwargs: JSONValue,
    ) -> ChatTemplateOutput:
        """Render messages as text or token IDs using the processor's template."""
        ...


def _part_text(part: JSONObject) -> str:
    text = part.get("text", part.get("content", ""))
    if not isinstance(text, str):
        raise TypeError("Message text parts must contain strings")
    return text


def extract_text_from_content(content: JSONValue) -> str:
    """Extract text without tokenizing image payloads or coercing malformed values."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise TypeError("Message content must be text or a list of content parts")
    return " ".join(
        _part_text(item)
        for item in content
        if isinstance(item, dict)
        and item.get("type") in ("text", "input_text", "output_text")
    ).strip()


def _content_media_count(content: JSONValue, media_types: tuple[str, ...]) -> int:
    if not isinstance(content, list):
        return 0
    return sum(
        isinstance(item, dict) and item.get("type") in media_types for item in content
    )


def reject_unsupported_media(content: JSONValue) -> None:
    """Raise for video or audio parts; MedGemma's processor encodes only images."""
    if _content_media_count(content, _VIDEO_PART_TYPES):
        raise ValueError("Video inputs are outside the medical runtime")
    if _content_media_count(content, _AUDIO_PART_TYPES):
        raise ValueError("Audio inputs are outside the medical runtime")


def reject_tool_message(message: JSONObject) -> None:
    """Raise for tool turns; the MedGemma chat template defines no tool format."""
    if (
        message.get("role") == "tool"
        or message.get("tool_calls")
        or message.get("tool_call_id") is not None
    ):
        raise ValueError("Tool calling is outside the medical runtime")


def normalize_image_content(content: JSONValue) -> str | list[JSONObject]:
    """Keep ordered text/image markers, with image payloads carried separately."""
    if not isinstance(content, list):
        return extract_text_from_content(content)
    reject_unsupported_media(content)
    parts: list[JSONObject] = []
    for item in content:
        if not isinstance(item, dict):
            raise TypeError("Message content parts must be objects")
        kind = item.get("type")
        if kind in ("image", "image_url", "input_image"):
            parts.append({"type": "image"})
        elif kind in ("text", "input_text", "output_text"):
            text = _part_text(item)
            if text:
                parts.append({"type": "text", "text": text})
    if any(part["type"] == "image" for part in parts):
        return parts
    return " ".join(_part_text(part) for part in parts).strip()


class MessageFormatter:
    """Format Gemma image tokens without reordering parts."""

    def __init__(self, model_name: str) -> None:
        """Select the Gemma formatter, rejecting other architectures."""
        self.model_name = model_name.lower()
        if self.model_name not in SUPPORTED_PROMPT_MODELS:
            raise ValueError(f"Unsupported model: {model_name}")

    @staticmethod
    def _content_parts[T](
        prompt: str | list[JSONObject],
        text_builder: Callable[[str], T],
        image_builder: Callable[[int], T],
        include_images: bool,
    ) -> tuple[list[T], int]:
        if not isinstance(prompt, list):
            return [text_builder(extract_text_from_content(prompt))], 0
        parts: list[T] = []
        image_count = 0
        for item in prompt:
            kind = item.get("type")
            if kind in ("image", "image_url", "input_image") and include_images:
                parts.append(image_builder(image_count))
                image_count += 1
            elif kind in ("text", "input_text", "output_text"):
                text = _part_text(item)
                if text:
                    parts.append(text_builder(text))
        return parts, image_count

    def format_message(
        self,
        prompt: str | list[JSONObject],
        role: str = "user",
        skip_image_token: bool = False,
        num_images: int = 1,
    ) -> JSONObject:
        """Insert Gemma image tokens while retaining explicit image positions."""
        content = list[JSONValue](prompt) if isinstance(prompt, list) else prompt
        reject_unsupported_media(content)
        include_images = role == "user" and not skip_image_token
        explicit = _content_media_count(content, ("image", "image_url", "input_image"))
        missing = max(0, num_images - explicit) if include_images else 0
        parts, _ = self._content_parts(
            prompt, str, lambda _: "<start_of_image>", include_images
        )
        return {"role": role, "content": "".join(parts) + "<start_of_image>" * missing}


def get_message_json(
    model_name: str,
    prompt: str | list[JSONObject],
    role: str = "user",
    skip_image_token: bool = False,
    num_images: int = 0,
) -> JSONObject:
    """Build one medical message, leaving image tensors in their side channel."""
    return MessageFormatter(model_name).format_message(
        prompt, role, skip_image_token, num_images
    )


def _marker_token(processor: object, kind: str) -> str:
    token = getattr(processor, f"{kind}_token", None)
    if not isinstance(token, str) or not token:
        token = getattr(getattr(processor, "tokenizer", None), f"{kind}_token", None)
    return token if isinstance(token, str) and token else f"<{kind}>"


def _flatten_content(content: JSONValue, image_token: str) -> str:
    if not isinstance(content, list):
        return extract_text_from_content(content)
    reject_unsupported_media(content)
    parts: list[str] = []
    previous_marker = False
    for item in content:
        if not isinstance(item, dict):
            raise TypeError("Message content parts must be objects")
        kind = item.get("type")
        part = (
            image_token
            if kind in ("image", "image_url", "input_image")
            else _part_text(item)
        )
        if part and previous_marker and part != image_token and not part[0].isspace():
            parts.append(" ")
        parts.append(part)
        previous_marker = part == image_token
    return "".join(parts).strip()


def _plain_prompt(
    processor: object, messages: list[JSONObject], add_generation_prompt: bool
) -> str:
    pairs = [
        (
            _message_role(message),
            _flatten_content(message.get("content"), _marker_token(processor, "image")),
        )
        for message in messages
    ]
    if len(pairs) == 1 and pairs[0][0] == "user":
        return pairs[0][1]
    lines = [
        (
            f"{role.capitalize()}: {content}".rstrip()
            if role in ("system", "user", "assistant")
            else content
        )
        for role, content in pairs
    ]
    if add_generation_prompt and pairs:
        lines.append("Assistant:")
    return "\n".join(lines).strip()


def _template_processor(
    processor: object, override: JSONValue
) -> TemplateProcessor | None:
    for candidate in (processor, getattr(processor, "tokenizer", None)):
        if (
            override is not None
            or getattr(candidate, "chat_template", None) is not None
        ):
            if isinstance(candidate, TemplateProcessor):
                return candidate
    if override is not None or any(
        getattr(candidate, "chat_template", None) is not None
        for candidate in (processor, getattr(processor, "tokenizer", None))
    ):
        raise TypeError(
            "Configured chat templates require an apply_chat_template method"
        )
    return None


def _supports_template_kw(processor: TemplateProcessor, name: str) -> bool:
    try:
        signature = inspect.signature(processor.apply_chat_template)
    except (TypeError, ValueError):
        return False
    return name in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _template_references_kw(
    processor: TemplateProcessor, override: JSONValue, name: str
) -> bool:
    for template in (
        override,
        getattr(processor, "chat_template", None),
        getattr(getattr(processor, "tokenizer", None), "chat_template", None),
    ):
        if isinstance(template, str) and name in template:
            return True
        if is_json_object(template) and any(
            isinstance(value, str) and name in value for value in template.values()
        ):
            return True
    return False


def get_chat_template(
    processor: object,
    messages: list[JSONObject],
    add_generation_prompt: bool,
    tokenize: bool = False,
    **kwargs: JSONValue,
) -> ChatTemplateOutput:
    """Render configured native templates; propagate their errors unchanged."""
    native = _template_processor(processor, kwargs.get("chat_template"))
    if native is None:
        return _plain_prompt(processor, messages, add_generation_prompt)
    options = dict(kwargs)
    if "enable_thinking" not in options and _supports_template_kw(
        native, "enable_thinking"
    ):
        options["enable_thinking"] = False
    if (
        "thinking_mode" not in options
        and options.get("enable_thinking") is True
        and _template_references_kw(
            native, options.get("chat_template"), "thinking_mode"
        )
    ):
        options["thinking_mode"] = "enabled"
    return native.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        **options,
    )


def _message_role(message: JSONObject) -> str:
    role = message.get("role", "user")
    if not isinstance(role, str):
        raise TypeError("Message roles must be strings")
    return role


def _prompt_messages(
    prompt: str | JSONObject | list[JSONValue] | list[JSONObject],
) -> list[JSONObject]:
    items = prompt if isinstance(prompt, list) else [prompt]
    messages: list[JSONObject] = []
    for item in items:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
        elif isinstance(item, dict):
            _message_role(item)
            messages.append(item)
        else:
            raise TypeError("Prompts must contain message objects or strings")
    return messages


def _message_content(message: JSONObject) -> str | list[JSONObject]:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list) or not all(
        isinstance(part, dict) for part in content
    ):
        raise TypeError("Message content must be text or a list of objects")
    return [part for part in content if isinstance(part, dict)]


def _format_messages(
    model_type: str,
    originals: list[JSONObject],
    num_images: int,
) -> list[JSONObject]:
    for message in originals:
        reject_tool_message(message)
    counts = [
        (
            _content_media_count(
                message.get("content"), ("image", "image_url", "input_image")
            )
            if _message_role(message) not in ("system", "assistant")
            else 0
        )
        for message in originals
    ]
    users = [
        i
        for i, message in enumerate(originals)
        if _message_role(message) not in ("system", "assistant")
    ]
    last_user = users[-1] if users else -1
    if last_user >= 0:
        counts[last_user] += max(0, num_images - sum(counts))
    messages: list[JSONObject] = []
    for i, message in enumerate(originals):
        role = _message_role(message)
        messages.append(
            get_message_json(
                model_type,
                _message_content(message),
                role,
                skip_image_token=counts[i] == 0 or role in ("system", "assistant"),
                num_images=counts[i],
            )
        )
    return messages


def apply_chat_template(
    processor: object,
    config: BaseModelConfig | JSONObject | GenerationConfigLike,
    prompt: str | JSONObject | list[JSONValue] | list[JSONObject],
    add_generation_prompt: bool = True,
    return_messages: bool = False,
    num_images: int = 0,
    **kwargs: JSONValue,
) -> list[JSONObject] | ChatTemplateOutput:
    """Render medical messages with native Gemma image tokens.

    Explicit image markers consume side-channel images on their original turn;
    remaining images attach to the last user turn. Video and audio parts and
    tool turns raise ``ValueError``.
    """
    model_type = (
        config.get("model_type")
        if isinstance(config, dict)
        else getattr(config, "model_type", None)
    )
    if not isinstance(model_type, str) or model_type not in SUPPORTED_PROMPT_MODELS:
        raise ValueError(f"Unsupported medical prompt model: {model_type}")
    # Remaining kwargs reach the native template, which would ignore these.
    if "video" in kwargs:
        raise ValueError("Video inputs are outside the medical runtime")
    if "num_audios" in kwargs:
        raise ValueError("Audio inputs are outside the medical runtime")
    if "tools" in kwargs:
        raise ValueError("Tool calling is outside the medical runtime")
    messages = _format_messages(model_type, _prompt_messages(prompt), num_images)
    if return_messages:
        return messages
    tokenize = kwargs.pop("tokenize", False)
    if not isinstance(tokenize, bool):
        raise TypeError("tokenize must be boolean")
    return get_chat_template(
        processor, messages, add_generation_prompt, tokenize=tokenize, **kwargs
    )
