"""Normalize compatible API requests into server generation arguments."""

from typing import List, Optional, Tuple

from pydantic import BaseModel, TypeAdapter

from mlx_vlm.contracts import JSONObject, JSONValue
from mlx_vlm.generate import (
    DEFAULT_REPETITION_CONTEXT_SIZE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from mlx_vlm.prompt_utils import extract_text_from_content, normalize_image_content
from mlx_vlm.server.generation import (
    GenerationArguments,
    get_server_enable_thinking,
    get_server_max_tokens,
    get_server_thinking_budget,
    get_server_thinking_end_token,
    get_server_thinking_start_token,
)
from mlx_vlm.server.runtime import runtime

_JSON_VALUE = TypeAdapter(JSONValue)
_LOGIT_BIAS = TypeAdapter(Optional[dict[int, float]])
_RAW_LOGIT_BIAS = TypeAdapter(dict[str, float] | None)
_OPTIONAL_FLOAT = TypeAdapter(Optional[float])
_OPTIONAL_INT = TypeAdapter(Optional[int])
_OPTIONAL_STRING = TypeAdapter(Optional[str])
_OPTIONAL_BOOL = TypeAdapter(Optional[bool])
_FLOAT = TypeAdapter(float)
_INT = TypeAdapter(int)
_BOOL = TypeAdapter(bool)

_DISABLED_REASONING_EFFORTS = {"none", "off", "disabled", "false", "0"}


def chat_message_to_prompt(message: JSONObject) -> JSONObject:
    """Convert one wire message to the chat-template representation.

    User content lists retain normalized image parts; other content lists are
    reduced to text. Reasoning text is copied when present.
    """
    content = message.get("content")
    if isinstance(content, list):
        content = (
            _JSON_VALUE.validate_python(normalize_image_content(content), strict=True)
            if message["role"] == "user"
            else extract_text_from_content(content)
        )
    msg: JSONObject = {"role": message["role"], "content": content}
    if message.get("reasoning_content") is not None:
        msg["reasoning_content"] = message.get("reasoning_content")
        msg["reasoning"] = message.get("reasoning_content")

    return msg


def normalize_instruction_messages(
    chat_messages: List[JSONObject],
    instructions: Optional[str] = None,
) -> Optional[str]:
    """Move all system/developer content into one leading system message.

    Explicit ``instructions`` come first, followed by nonempty message content
    in conversation order, separated by blank lines. The supplied message list is
    mutated in place; the combined instruction string (or None) is returned.
    """
    instruction_parts = [instructions] if instructions else []
    conversation: list[JSONObject] = []

    for message in chat_messages:
        if message.get("role") in ("system", "developer"):
            content = message.get("content")
            if content:
                instruction_parts.append(str(content))
        else:
            conversation.append(message)

    normalized_instructions = "\n\n".join(instruction_parts) or None
    if normalized_instructions:
        conversation.insert(
            0,
            {"role": "system", "content": normalized_instructions},
        )
    chat_messages[:] = conversation
    return normalized_instructions


def _request_field_is_set(request: BaseModel, field_name: str) -> bool:
    fields_set = getattr(request, "model_fields_set", None)
    if fields_set is not None:
        return field_name in fields_set
    return getattr(request, field_name, None) is not None


def _request_field_or_default(
    request: BaseModel, field_name: str, default: JSONValue, *, config: object = None
) -> JSONValue:
    fields_set = getattr(request, "model_fields_set", None)
    if fields_set is None or field_name in fields_set:
        value = getattr(request, field_name, None)
        if value is not None:
            return _JSON_VALUE.validate_python(value, strict=True)
    if default is not None:
        return default
    return _JSON_VALUE.validate_python(getattr(config, field_name, None), strict=True)


def _reasoning_effort_enabled(
    effort: JSONValue,
) -> Tuple[Optional[bool], Optional[str]]:
    if effort is None:
        return None, None
    normalized = str(effort).strip().lower()
    if not normalized:
        return None, None
    return normalized not in _DISABLED_REASONING_EFFORTS, normalized


def _standard_reasoning_control(
    request: BaseModel,
) -> Tuple[Optional[bool], Optional[str], bool]:
    """Normalize OpenAI reasoning fields into a thinking-mode decision."""
    if _request_field_is_set(request, "reasoning"):
        reasoning = as_plain_dict(getattr(request, "reasoning", None))
        if isinstance(reasoning, dict):
            enabled, effort = _reasoning_effort_enabled(reasoning.get("effort"))
            return (True if enabled is None else enabled), effort, True
        if isinstance(reasoning, bool):
            return reasoning, None, True
        enabled, effort = _reasoning_effort_enabled(reasoning)
        if enabled is not None:
            return enabled, effort, True

    if _request_field_is_set(request, "reasoning_effort"):
        enabled, effort = _reasoning_effort_enabled(
            getattr(request, "reasoning_effort", None)
        )
        if enabled is not None:
            return enabled, effort, True

    return None, None, False


def _model_config_field_or_default(
    processor: object, field_name: str, default: JSONValue
) -> JSONValue:
    config = getattr(processor, "config", None) if processor is not None else None
    if config is None:
        config = runtime.model_cache.for_kind().get("config")
    return _JSON_VALUE.validate_python(
        getattr(config, field_name, default), strict=True
    )


def as_plain_dict(value: object) -> JSONValue:
    """Normalize model-backed wire values and validate their JSON representation."""
    if isinstance(value, BaseModel):
        value = value.model_dump(exclude_none=True)
    return _JSON_VALUE.validate_python(value, strict=True)


def build_gen_args(
    request: BaseModel,
    processor: object = None,
) -> GenerationArguments:
    """Build generation arguments from a compatible API request."""
    config = getattr(processor, "config", None) if processor is not None else None
    max_tokens = getattr(request, "max_tokens", None)
    if max_tokens is None:
        max_tokens = getattr(request, "max_output_tokens", None)
    if max_tokens is None:
        max_tokens = get_server_max_tokens()
    logit_bias = _RAW_LOGIT_BIAS.validate_python(
        getattr(request, "logit_bias", None), strict=True
    )
    if logit_bias is not None:
        parsed_logit_bias = {int(k): v for k, v in logit_bias.items()}
    else:
        parsed_logit_bias = None
    standard_reasoning, reasoning_effort, has_standard_reasoning = (
        _standard_reasoning_control(request)
    )
    server_enable_thinking = get_server_enable_thinking()
    if _request_field_is_set(request, "enable_thinking"):
        enable_thinking = bool(getattr(request, "enable_thinking", False))
        template_reasoning = enable_thinking
    elif has_standard_reasoning:
        enable_thinking = bool(standard_reasoning)
        template_reasoning = enable_thinking
    else:
        enable_thinking = server_enable_thinking
        # Preserve a model template's native default when the server default is
        # off and the request did not express a reasoning preference.
        template_reasoning = True if server_enable_thinking else None
    default_temperature = _model_config_field_or_default(
        processor, "temperature", DEFAULT_TEMPERATURE
    )
    default_top_p = _model_config_field_or_default(processor, "top_p", DEFAULT_TOP_P)
    default_top_k = _model_config_field_or_default(processor, "top_k", 0)
    if _model_config_field_or_default(processor, "do_sample", None) is False:
        default_temperature = 0.0
    args = GenerationArguments(
        max_tokens=_INT.validate_python(max_tokens, strict=True),
        temperature=_FLOAT.validate_python(
            _request_field_or_default(request, "temperature", default_temperature),
            strict=True,
        ),
        top_p=_FLOAT.validate_python(
            _request_field_or_default(request, "top_p", default_top_p), strict=True
        ),
        top_k=_INT.validate_python(
            _request_field_or_default(request, "top_k", default_top_k), strict=True
        ),
        min_p=_FLOAT.validate_python(getattr(request, "min_p", 0.0), strict=True),
        top_n_sigma=_FLOAT.validate_python(
            getattr(request, "top_n_sigma", 0.0), strict=True
        ),
        p_less=_BOOL.validate_python(getattr(request, "p_less", False), strict=True),
        typical_p=_FLOAT.validate_python(
            getattr(request, "typical_p", 1.0), strict=True
        ),
        seed=_OPTIONAL_INT.validate_python(getattr(request, "seed", None), strict=True),
        logprobs=_BOOL.validate_python(
            bool(getattr(request, "logprobs", False)), strict=True
        ),
        repetition_penalty=_OPTIONAL_FLOAT.validate_python(
            getattr(request, "repetition_penalty", None), strict=True
        ),
        repetition_context_size=_OPTIONAL_INT.validate_python(
            _request_field_or_default(
                request,
                "repetition_context_size",
                DEFAULT_REPETITION_CONTEXT_SIZE,
            ),
            strict=True,
        ),
        presence_penalty=_OPTIONAL_FLOAT.validate_python(
            getattr(request, "presence_penalty", None), strict=True
        ),
        presence_context_size=_OPTIONAL_INT.validate_python(
            _request_field_or_default(
                request,
                "presence_context_size",
                DEFAULT_REPETITION_CONTEXT_SIZE,
            ),
            strict=True,
        ),
        frequency_penalty=_OPTIONAL_FLOAT.validate_python(
            getattr(request, "frequency_penalty", None), strict=True
        ),
        frequency_context_size=_OPTIONAL_INT.validate_python(
            _request_field_or_default(
                request,
                "frequency_context_size",
                DEFAULT_REPETITION_CONTEXT_SIZE,
            ),
            strict=True,
        ),
        logit_bias=_LOGIT_BIAS.validate_python(parsed_logit_bias, strict=True),
        enable_thinking=_BOOL.validate_python(enable_thinking, strict=True),
        reasoning=_OPTIONAL_BOOL.validate_python(template_reasoning, strict=True),
        reasoning_effort=_OPTIONAL_STRING.validate_python(
            reasoning_effort, strict=True
        ),
        thinking_budget=_OPTIONAL_INT.validate_python(
            _request_field_or_default(
                request, "thinking_budget", get_server_thinking_budget()
            ),
            strict=True,
        ),
        thinking_start_token=_OPTIONAL_STRING.validate_python(
            _request_field_or_default(
                request,
                "thinking_start_token",
                get_server_thinking_start_token(),
                config=config,
            ),
            strict=True,
        ),
        thinking_end_token=_OPTIONAL_STRING.validate_python(
            _request_field_or_default(
                request,
                "thinking_end_token",
                get_server_thinking_end_token(),
                config=config,
            ),
            strict=True,
        ),
    )
    return args
