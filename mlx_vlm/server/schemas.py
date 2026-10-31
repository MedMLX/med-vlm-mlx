"""Request and response models for the compatible HTTP API."""

import os
from typing import (
    TYPE_CHECKING,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Required, TypeAlias, TypedDict

from mlx_vlm.contracts import JSONValue

if TYPE_CHECKING:
    from mlx_vlm.server.generation import GenerationMetrics

from mlx_vlm.generate import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from mlx_vlm.generate.ar import normalize_resize_shape


def get_server_max_tokens():
    """Read the default output limit from ``MLX_VLM_MAX_TOKENS``."""
    return int(os.environ.get("MLX_VLM_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))


_TOOL_REQUEST_FIELDS = ("tools", "tool_choice", "tool_parser")
_TOOL_ITEM_TYPES = frozenset(
    {
        "function_call",
        "function_call_output",
        "shell_call",
        "shell_call_output",
        "apply_patch_call",
        "apply_patch_call_output",
        "tool_use",
        "tool_result",
    }
)
# Removed features. Request models accept unknown fields, so these must fail
# loudly instead of being dropped and changing what the caller asked for.
_REMOVED_REQUEST_FIELDS = {
    "context_management": "context compaction is outside the medical runtime",
    "adapter_path": "adapter loading is outside the medical runtime",
}
_VIDEO_PART_TYPES = frozenset({"video", "input_video", "video_url"})
_AUDIO_PART_TYPES = frozenset({"audio", "input_audio"})


def _reject_unsupported_items(value: object) -> None:
    if isinstance(value, list):
        for item in cast(list[object], value):
            _reject_unsupported_items(item)
        return
    if not isinstance(value, dict):
        return
    item = cast(dict[str, object], value)
    kind = item.get("type")
    if kind in _VIDEO_PART_TYPES:
        raise ValueError("Video inputs are outside the medical runtime")
    if kind in _AUDIO_PART_TYPES:
        raise ValueError("Audio inputs are outside the medical runtime")
    if (
        kind in _TOOL_ITEM_TYPES
        or item.get("role") == "tool"
        or item.get("tool_calls")
        or item.get("tool_call_id") is not None
    ):
        raise ValueError("Tool calling is outside the medical runtime")
    _reject_unsupported_items(item.get("content"))


def _reject_unsupported_inputs(cls: type[BaseModel], data: object) -> object:
    """Reject tool calling, video, audio and removed request options.

    Request models accept unknown fields, so without this check these inputs
    would be dropped silently instead of failing validation.
    """
    if not isinstance(data, dict):
        return data
    request = cast(dict[str, object], data)
    for field in _TOOL_REQUEST_FIELDS:
        if request.get(field) is not None:
            raise ValueError(
                f"{field} is unsupported: tool calling is outside the medical runtime"
            )
    for field, reason in _REMOVED_REQUEST_FIELDS.items():
        if request.get(field) is not None:
            raise ValueError(f"{field} is unsupported: {reason}")
    response_format = request.get("response_format")
    if response_format is not None and response_format != {"type": "text"}:
        raise ValueError(
            "response_format is unsupported: structured output is outside the "
            "medical runtime"
        )
    for field in ("messages", "input"):
        _reject_unsupported_items(request.get(field))
    return request


class FlexibleBaseModel(BaseModel):
    """Base model that ignores/accepts any unknown OpenAI SDK fields."""

    model_config = ConfigDict(extra="allow")


# OpenAI API Models


class ResponseInputTextParam(TypedDict, total=False):
    """Text content part of a Responses API input message."""

    text: Required[str]
    type: Required[
        Literal["input_text", "text"]
    ]  # The type of the input item. Always `input_text`.


class ResponseInputImageParam(TypedDict, total=False):
    """Image content part; file IDs are declared but rejected by the server."""

    detail: Literal["high", "low", "auto"]
    """The detail level of the image to be sent to the model.

    One of `high`, `low`, or `auto`. Validation preserves omission of this key.
    """
    type: Required[
        Literal["input_image"]
    ]  # The type of the input item. Always `input_image`.
    image_url: Optional[str]
    file_id: Optional[str]
    """A file reference. This server currently rejects file IDs."""


class ImageUrl(TypedDict, total=False):
    """URL wrapper used by chat-style image parts."""

    url: Required[str]


class ResponseImageUrlParam(TypedDict, total=False):
    """Chat-style image part carrying a required URL object."""

    type: Required[
        Literal["image_url"]
    ]  # The type of the input item. Always`image_url`.
    image_url: Required[ImageUrl]


ResizeShapeInput: TypeAlias = Union[Tuple[int], Tuple[int, int]]

ResponseInputContentParam: TypeAlias = Union[
    ResponseInputTextParam,
    ResponseInputImageParam,
    ResponseImageUrlParam,
]

ResponseInputMessageContentListParam: TypeAlias = List[ResponseInputContentParam]


class ResponseOutputText(TypedDict, total=False):
    """Text item of an assistant response."""

    text: Required[str]
    type: Required[
        Literal["output_text"]
    ]  # The type of the output item. Always `output_text`


ResponseOutputMessageContentList: TypeAlias = List[ResponseOutputText]


class ChatMessage(FlexibleBaseModel):
    """One chat message; ``content`` is text or a list of typed parts."""

    role: Literal["user", "assistant", "system", "developer"] = Field(
        ...,
        description="Role of the message sender.",
    )
    content: Union[
        str,
        None,
        ResponseInputMessageContentListParam,
        ResponseOutputMessageContentList,
    ] = Field(default=None, description="Content of the message.")
    reasoning_content: Optional[str] = Field(
        default=None,
        description="Thinking/reasoning content (when thinking is enabled).",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description=(
            "Deprecated alias for reasoning_content, kept for backward compatibility."
        ),
    )

    @model_validator(mode="after")
    def sync_reasoning_aliases(self):
        """Mirror ``reasoning`` and ``reasoning_content`` onto each other."""
        if self.reasoning_content is None and self.reasoning is not None:
            self.reasoning_content = self.reasoning
        elif self.reasoning is None and self.reasoning_content is not None:
            self.reasoning = self.reasoning_content
        return self


class PromptTokensDetails(BaseModel):
    """Prompt token breakdown reported by compatible APIs."""

    cached_tokens: int = 0


class GenerationTimings(BaseModel):
    """Per-request timing breakdown."""

    prompt_n: int
    cache_n: int
    predicted_n: int
    prompt_ms: float
    prompt_per_token_ms: float
    prompt_per_second: float
    predicted_ms: float
    predicted_per_token_ms: float
    predicted_per_second: float
    peak_memory: float = 0.0

    @staticmethod
    def _derive_gen_tps(token_times: List[float]) -> Optional[float]:
        if len(token_times) < 2:
            return None
        elapsed = token_times[-1] - token_times[0]
        return (len(token_times) - 1) / elapsed if elapsed > 0 else None

    @classmethod
    def from_metrics(
        cls,
        metrics: "GenerationMetrics",
        prompt_tokens: int,
        output_tokens: int,
    ) -> "GenerationTimings":
        """Build timings from one request's generation metrics.

        ``prompt_n`` excludes cached prompt tokens; rates are zero when the
        metric needed to derive them is missing or zero.
        """
        generation_tps = getattr(metrics, "rate", None)
        if generation_tps is None:
            generation_tps = metrics.generation_tps or cls._derive_gen_tps(
                metrics.token_times
            )
        cached_tokens = metrics.cached_tokens
        prompt_n = max(0, int(prompt_tokens) - int(cached_tokens))
        prompt_s = prompt_tokens / metrics.prompt_tps if metrics.prompt_tps else 0.0
        prompt_ms = prompt_s * 1000.0
        predicted_ms = (
            output_tokens / generation_tps * 1000.0 if generation_tps else 0.0
        )
        return cls(
            prompt_n=prompt_n,
            cache_n=int(cached_tokens),
            predicted_n=int(output_tokens),
            prompt_ms=prompt_ms,
            prompt_per_token_ms=(prompt_ms / prompt_n) if prompt_n else 0.0,
            prompt_per_second=(prompt_n / prompt_s) if prompt_s else 0.0,
            predicted_ms=predicted_ms,
            predicted_per_token_ms=(
                predicted_ms / output_tokens if output_tokens else 0.0
            ),
            predicted_per_second=float(generation_tps or 0.0),
            peak_memory=float(metrics.peak_memory or 0.0),
        )


class StreamingTimings(BaseModel):
    """Timing data available while a response is still streaming."""

    predicted_per_second: Optional[float] = None


class VLMRequest(FlexibleBaseModel):
    """Sampling, media and thinking options shared by text requests."""

    model: str = Field(
        ...,
        description="The path to the local model directory or Hugging Face repo.",
    )
    max_tokens: int = Field(
        default_factory=get_server_max_tokens,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        DEFAULT_TEMPERATURE, description="Temperature for sampling."
    )
    top_p: float = Field(DEFAULT_TOP_P, description="Top-p sampling.")
    top_k: int = Field(0, description="Top-k sampling.")
    min_p: float = Field(0.0, description="Min-p sampling.")
    top_n_sigma: float = Field(
        0.0, description="Top-nσ sampling. 0 disables; typical ~0.5-2.0."
    )
    p_less: bool = Field(
        False, description="Hyperparameter-free p-less sampling (on/off)."
    )
    typical_p: float = Field(
        1.0, description="Locally typical sampling. 1.0 disables; typical ~0.2-0.95."
    )
    seed: int = Field(DEFAULT_SEED, description="Seed for random generation.")
    repetition_penalty: Optional[float] = Field(
        default=None, description="Repetition penalty."
    )
    repetition_context_size: Optional[int] = Field(
        default=None, description="Repetition penalty context size."
    )
    presence_penalty: Optional[float] = Field(
        default=None, description="Presence penalty."
    )
    presence_context_size: Optional[int] = Field(
        default=None, description="Presence penalty context size."
    )
    frequency_penalty: Optional[float] = Field(
        default=None, description="Frequency penalty."
    )
    frequency_context_size: Optional[int] = Field(
        default=None, description="Frequency penalty context size."
    )
    logit_bias: Optional[JSONValue] = Field(
        default=None, description="Logit bias dict."
    )
    enable_thinking: Optional[bool] = Field(
        default=None,
        description=(
            "Override server thinking mode for this request. If omitted, the "
            "server default set by --enable-thinking is used."
        ),
    )
    reasoning: Optional[JSONValue] = Field(
        default=None,
        description="OpenAI-compatible reasoning configuration.",
    )
    reasoning_effort: Optional[str] = Field(
        default=None,
        description="OpenAI-compatible reasoning effort.",
    )
    thinking_budget: Optional[int] = Field(
        default=None, description="Max thinking tokens."
    )
    thinking_start_token: Optional[str] = Field(
        default=None, description="Thinking start token."
    )
    thinking_end_token: Optional[str] = Field(
        default=None, description="Thinking end token."
    )
    logprobs: Optional[bool] = Field(
        default=None,
        description="Return log-probabilities for each output token.",
    )
    top_logprobs: Optional[int] = Field(
        default=None,
        description=(
            "Number of most-likely tokens to return at each position "
            "(0-20). Requires logprobs=true. The server-side cap is set by "
            "the TOP_LOGPROBS_K env var; values above the cap are clamped."
        ),
    )
    resize_shape: Optional[ResizeShapeInput] = Field(
        default=None,
        description="Resize shape for the image. Provide one integer for square or two for (height, width).",
    )

    @field_validator("resize_shape", mode="before")
    @classmethod
    def normalize_resize_shape_field(cls, value: object) -> tuple[int, int] | None:
        """Normalize one or two sizes to an ``(height, width)`` tuple."""
        return normalize_resize_shape(value)


class GenerationRequest(VLMRequest):
    """
    Inherits from VLMRequest and adds additional fields for the generation request.
    """

    stream: bool = Field(
        False, description="Whether to stream the response chunk by chunk."
    )


class UsageStats(BaseModel):
    """OpenAI-compatible usage statistics for chat completions. Throughput and
    memory metrics live in `GenerationTimings` (sibling `timings` field on the
    response) to keep this object spec-clean."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: PromptTokensDetails = Field(
        default_factory=PromptTokensDetails
    )

    @classmethod
    def from_metrics(
        cls, metrics: "GenerationMetrics", prompt_tokens: int, completion_tokens: int
    ) -> "UsageStats":
        """Build usage counts, including cached prompt tokens, from metrics."""
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=PromptTokensDetails(
                cached_tokens=metrics.cached_tokens
            ),
        )


class StreamOptions(BaseModel):
    """Streaming options accepted by chat completions."""

    include_usage: bool = False


class ChatRequest(GenerationRequest):
    """Chat completion request with messages and unsupported-feature checks."""

    messages: List[ChatMessage]
    stream_options: Optional[StreamOptions] = None
    _reject_unsupported = model_validator(mode="before")(_reject_unsupported_inputs)


class TopLogprob(BaseModel):
    """One alternative token at a log-probability position."""

    token: str
    logprob: float
    bytes: Optional[List[int]] = None


class ChatLogprobContent(BaseModel):
    """Log probability of one generated token and its alternatives."""

    token: str
    logprob: float
    bytes: Optional[List[int]] = None
    top_logprobs: List[TopLogprob] = []


class ChatLogprobs(BaseModel):
    """Log probabilities for a whole completion."""

    content: List[ChatLogprobContent] = []


class ChatChoice(BaseModel):
    """One non-streamed completion choice."""

    index: int = 0
    finish_reason: str = "stop"
    message: ChatMessage
    logprobs: Optional[ChatLogprobs] = None


class ChatResponse(BaseModel):
    """Non-streamed ``chat.completion`` response."""

    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: List[ChatChoice] = []
    usage: Optional[UsageStats] = None
    timings: Optional[GenerationTimings] = None


class ChatStreamChoice(BaseModel):
    """One streamed choice delta."""

    index: int = 0
    finish_reason: Optional[str] = None
    delta: ChatMessage
    logprobs: Optional[ChatLogprobs] = None


class ChatStreamChunk(BaseModel):
    """One ``chat.completion.chunk`` stream event."""

    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: List[ChatStreamChoice] = []
    usage: Optional[UsageStats] = None
    timings: Optional[Union[GenerationTimings, StreamingTimings]] = None


# Models for Anthropic-compatible /v1/messages endpoint


class ModelInfo(BaseModel):
    """One entry in the model listing."""

    id: str
    object: str
    created: int
    loaded: bool = Field(
        default=False, description="Whether the model is loaded in this server process."
    )


class ModelsResponse(BaseModel):
    """Model listing response."""

    object: Literal["list"]
    data: List[ModelInfo]
