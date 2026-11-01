"""State owned by model caches and the server's generation worker."""

from __future__ import annotations

from queue import Queue
from threading import Lock
from typing import TYPE_CHECKING, TypedDict

from mlx_vlm.contracts import PromptInputs
from mlx_vlm.generate.types import ProcessorLike, TokenizerLike
from mlx_vlm.tokenizer_utils import ServerTokenStreamer

if TYPE_CHECKING:
    from mlx_vlm.models.protocols import GenerationConfigLike, GenerationModelLike
    from mlx_vlm.server.generation import (
        GenerationContext,
        ResponseGenerator,
        StreamingToken,
    )
    from mlx_vlm.vision_cache import VisionFeatureCache


class ModelResources(TypedDict, total=False):
    """Resources owned by one independently unloadable model cache group."""

    cache_key: tuple[str, str, str]
    model_path: str
    model_kind: str
    model: GenerationModelLike
    processor: ProcessorLike | TokenizerLike
    config: GenerationConfigLike
    vision_cache: VisionFeatureCache
    response_generator: ResponseGenerator | None
    generation_lock: Lock


type GenerationEvent = GenerationContext | StreamingToken | Exception | None


class RequestLogState(TypedDict, total=False):
    """Timing and token counters tracked while a request is in flight."""

    request_id: str
    queued_at: float
    prefill_started_at: float
    prefill_processed: int
    generated_tokens: int
    decode_started_at: float | None
    last_token_at: float | None
    decode_first_chunk_tokens: int


class ActiveGeneration(RequestLogState):
    """Worker-side state of one active streamed generation.

    ``rqueue`` carries generation events from the worker thread; ``streamer``
    converts tokens to text deltas and ``gen_kwargs`` holds the prepared prompt
    inputs until prefill completes.
    """

    rqueue: Queue[GenerationEvent]
    streamer: ServerTokenStreamer
    gen_kwargs: PromptInputs | None
    prompt_tps: float | None
    cached_tokens: int


class MetricsEnvelope(TypedDict, total=False):
    """One completed request's metrics record.

    Token counts and rates are per request; ``ttft_s`` is time to first token
    and the ``sliding_decode_tok_s_*`` fields summarize early or late decode
    windows.
    """

    timestamp_unix: float
    endpoint: str
    model: str
    stream: bool
    backend: str
    prompt_tokens: int
    completion_tokens: int
    generated_tokens: int
    reasoning_tokens: int
    total_tokens: int
    prompt_eval_time_s: float | None
    prefill_tok_s: float | None
    ttft_s: float | None
    decode_elapsed_s: float | None
    request_elapsed_s: float
    request_tok_s: float
    decode_tok_s: float | None
    sliding_decode_tok_s_first_32: float | None
    sliding_decode_tok_s_first_64: float | None
    sliding_decode_tok_s_last_32: float | None
    sliding_decode_tok_s_last_64: float | None
    peak_memory_gb: float | None
    finish_reason: str | None
    image_count: int
    thinking_enabled: bool


class RequestFailure(TypedDict):
    """One failed request, with the endpoint and model that rejected it."""

    timestamp_unix: float
    endpoint: str
    model: str
    stream: bool
    error: str


class MetricsSummary(TypedDict):
    """Aggregate counters since server start."""

    uptime_s: float
    requests_started: int
    requests_completed: int
    requests_failed: int
    streaming_requests: int
    in_flight: int
    prompt_tokens_total: int
    completion_tokens_total: int
    generated_tokens_total: int
    avg_request_time_s: float
    avg_request_tok_s: float
    avg_decode_tok_s: float
    last_request_at: float | None
    last_error: RequestFailure | None


class MetricsSnapshot(TypedDict):
    """Latest and recent request records plus aggregate counters."""

    latest: MetricsEnvelope | None
    recent: list[MetricsEnvelope]
    summary: MetricsSummary
