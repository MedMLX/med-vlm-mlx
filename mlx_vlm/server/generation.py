"""Continuous-batching generation worker and request metrics for the HTTP server."""

from __future__ import annotations

import gc
import logging
import os
import time
from collections import deque
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass, field
from queue import Empty as QueueEmpty
from queue import Queue
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import mlx.core as mx
import numpy as np
from fastapi import HTTPException
from typing_extensions import ReadOnly, TypedDict

from mlx_vlm._stream_cleanup import clear_mlx_streams
from mlx_vlm.contracts import (
    ImageSource,
    JSONValue,
    ProcessedInputs,
    ProcessorValue,
    PromptInputs,
    array_int_list,
    is_pixel_values,
)
from mlx_vlm.generate import (
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_QUANTIZED_KV_START,
    DEFAULT_REPETITION_CONTEXT_SIZE,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_END_TOKEN,
    DEFAULT_THINKING_START_TOKEN,
    DEFAULT_TOP_P,
    BatchGenerator,
)
from mlx_vlm.generate.ar import PromptProgress
from mlx_vlm.generate.common import GenerationResult
from mlx_vlm.generate.types import (
    GenerateTextKwargs,
    ProcessorLike,
    TokenEncoder,
    TokenizerLike,
    resolve_tokenizer,
)
from mlx_vlm.models.protocols import (
    GenerationConfigLike,
    GenerationModelLike,
    ModelArgument,
    require_generation_model,
)
from mlx_vlm.sample_utils import (
    apply_min_p,
    apply_p_less,
    apply_top_k,
    apply_top_n_sigma,
    apply_top_p,
    apply_typical_p,
    clamp_temperature,
    make_logits_processors,
    top_p_sampling,
)
from mlx_vlm.server.runtime import runtime
from mlx_vlm.server.state_types import (
    ActiveGeneration,
    GenerationEvent,
    MetricsEnvelope,
    MetricsSnapshot,
    RequestFailure,
    RequestLogState,
)
from mlx_vlm.tokenizer_utils import ServerTokenStreamer, make_streaming_detokenizer
from mlx_vlm.utils import (
    ThinkingBudgetCriteria,
    load,
    prepare_inputs,
    resolve_eos_token_ids,
)

if TYPE_CHECKING:
    from mlx_vlm.vision_cache import VisionFeatureCache

logger = logging.getLogger("mlx_vlm.server")

DEFAULT_LOG_PROGRESS_INTERVAL = 10
DEFAULT_ENABLE_THINKING = False
METRICS_HISTORY_LIMIT = 100
METRICS_RECENT_LIMIT = 32


class PromptTooLongError(ValueError):
    """Raised when a request exceeds the configured server context budget."""


def _notify_queues(queues: Iterable[Queue[GenerationEvent]], *items: GenerationEvent):
    for queue in queues:
        for item in items:
            try:
                queue.put(item)
            except Exception:
                break


def get_prefill_step_size():
    """Return the prefill chunk size configured by ``PREFILL_STEP_SIZE``."""
    return int(os.environ.get("PREFILL_STEP_SIZE", str(DEFAULT_PREFILL_STEP_SIZE)))


def get_max_num_seqs():
    """Max sequences allowed in the running batch at once (None = unbounded)."""
    raw = os.environ.get("MLX_VLM_MAX_NUM_SEQS", "")
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n > 0 else None


def get_server_max_tokens():
    """Return the default output limit configured by ``MLX_VLM_MAX_TOKENS``."""
    return int(os.environ.get("MLX_VLM_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))


def get_token_queue_timeout():
    """Return the live token queue timeout in seconds, or ``None`` when disabled."""
    return runtime.config.token_queue_timeout


def get_log_progress_interval():
    """Number of decoded tokens between INFO progress messages (0 disables)."""
    raw = os.environ.get(
        "MLX_VLM_LOG_PROGRESS_INTERVAL", str(DEFAULT_LOG_PROGRESS_INTERVAL)
    )
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "Invalid MLX_VLM_LOG_PROGRESS_INTERVAL=%r; falling back to %d.",
            raw,
            DEFAULT_LOG_PROGRESS_INTERVAL,
        )
        return DEFAULT_LOG_PROGRESS_INTERVAL


def _position_seed(seed: int, row_id: int, position: int) -> int:
    x = (int(seed) ^ 0x9E3779B9) & 0xFFFFFFFF
    x = (x + (int(row_id) + 1) * 0x85EBCA6B) & 0xFFFFFFFF
    x = (x ^ ((int(position) + 1) * 0xC2B2AE35)) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    return int(x & 0xFFFFFFFF)


def _position_keys(seed: int, row_ids: List[int], positions: List[int]) -> mx.array:
    return mx.stack(
        [
            mx.random.key(_position_seed(seed, row, pos))
            for row, pos in zip(row_ids, positions)
        ]
    )


class _PositionedTargetSampler:
    """Server sampler with stateless target draws for ragged verification."""

    def __init__(
        self,
        *,
        temperature: float,
        top_p: float,
        top_k: int = 0,
        seed: Optional[int],
        min_p: float = 0.0,
        top_n_sigma: float = 0.0,
        p_less: bool = False,
        typical_p: float = 1.0,
    ):
        self.temperature = clamp_temperature(float(temperature))
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.min_p = float(min_p)
        self.top_n_sigma = float(top_n_sigma)
        self.p_less = p_less
        self.typical_p = float(typical_p)
        self.seed = DEFAULT_SEED if seed is None else int(seed)

    def _apply_top_k(self, logprobs: mx.array) -> mx.array:
        return apply_top_k(logprobs, self.top_k) if self.top_k > 0 else logprobs

    def __call__(self, logprobs: mx.array) -> mx.array:
        if self._uses_filters():
            logprobs = self._filter(logprobs)
            return mx.random.categorical(logprobs * (1 / self.temperature))
        logprobs = self._apply_top_k(logprobs)
        if self.top_p > 0 and self.top_p < 1.0:
            return top_p_sampling(logprobs, self.top_p, self.temperature)
        return mx.random.categorical(logprobs * (1 / self.temperature))

    def sample_target(
        self,
        logprobs: mx.array,
        *,
        row_ids: List[int],
        positions: List[int],
    ) -> mx.array:
        if logprobs.shape[0] != len(row_ids) or len(row_ids) != len(positions):
            raise ValueError("row_ids and positions must match logprobs batch size.")
        if self._uses_filters():
            logprobs = self._filter(logprobs)
            keys = _position_keys(self.seed, row_ids, positions)
            return mx.vmap(self._sample_one, in_axes=(0, 0))(logprobs, keys)
        logprobs = self._apply_top_k(logprobs)
        keys = _position_keys(self.seed, row_ids, positions)
        if self.top_p > 0 and self.top_p < 1.0:
            return mx.vmap(self._sample_top_p_one, in_axes=(0, 0))(logprobs, keys)
        return mx.vmap(self._sample_one, in_axes=(0, 0))(logprobs, keys)

    def _sample_one(self, logprobs: mx.array, key: mx.array) -> mx.array:
        return mx.random.categorical(logprobs * (1 / self.temperature), key=key)

    def _uses_filters(self) -> bool:
        return bool(
            self.min_p or self.top_n_sigma > 0 or self.p_less or self.typical_p < 1
        )

    def _filter(self, logprobs: mx.array) -> mx.array:
        # Match make_sampler's filter order before drawing with the position key.
        if self.top_n_sigma > 0:
            logprobs = apply_top_n_sigma(logprobs, self.top_n_sigma)
        if self.p_less:
            logprobs = apply_p_less(logprobs, self.temperature)
        if 0 < self.typical_p < 1:
            logprobs = apply_typical_p(logprobs, self.typical_p)
        if 0 < self.top_p < 1:
            logprobs = apply_top_p(logprobs, self.top_p)
        if self.min_p:
            logprobs = apply_min_p(logprobs, self.min_p)
        return self._apply_top_k(logprobs)

    def _sample_top_p_one(self, logprobs: mx.array, key: mx.array) -> mx.array:
        return top_p_sampling(logprobs, self.top_p, self.temperature, key=key)


def get_server_enable_thinking():
    """Read the default thinking mode from ``MLX_VLM_ENABLE_THINKING``."""
    raw = os.environ.get("MLX_VLM_ENABLE_THINKING")
    if raw is None:
        return DEFAULT_ENABLE_THINKING
    return raw.lower() in ("1", "true", "yes", "on")


def get_server_thinking_budget():
    """Read the default thinking budget in tokens, if configured."""
    raw = os.environ.get("MLX_VLM_THINKING_BUDGET")
    return None if raw is None else int(raw)


def get_server_thinking_start_token():
    """Read the configured thinking start marker, if any."""
    return os.environ.get("MLX_VLM_THINKING_START_TOKEN")


def get_server_thinking_end_token():
    """Read the configured thinking end marker, if any."""
    return os.environ.get("MLX_VLM_THINKING_END_TOKEN")


def get_expert_cache_gb():
    """Read ``EXPERT_CACHE_GB``; warn and return ``None`` when invalid."""
    raw = os.environ.get("EXPERT_CACHE_GB", "")
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid EXPERT_CACHE_GB=%r; ignoring.", raw)
        return None


def get_quantized_kv_bits():
    """Return the KV bit width, where zero or unset disables quantization."""
    kv_bits = float(os.environ.get("KV_BITS", 0))
    return kv_bits or None


def get_kv_group_size():
    """Return the KV quantization group size from ``KV_GROUP_SIZE``."""
    return int(os.environ.get("KV_GROUP_SIZE", str(DEFAULT_KV_GROUP_SIZE)))


def get_configured_context_limit():
    """Return the context limit in tokens, preferring the live runtime setting."""
    if runtime.config.max_kv_size is not None:
        return runtime.config.max_kv_size or None
    max_kv_tokens = int(os.environ.get("MAX_KV_SIZE", 0))
    return max_kv_tokens or None


def count_prompt_tokens(raw_inputs: ProcessedInputs) -> int:
    """Count elements in array input IDs, or entries in a list/string.

    Array padding and every batch row count toward the total; this helper does
    not inspect an attention mask. For a list it returns the outer length, not
    a recursive token count, and for a string it returns the character count.
    """
    input_ids = raw_inputs["input_ids"]
    if isinstance(input_ids, (mx.array, np.ndarray)):
        return int(input_ids.size)
    if isinstance(input_ids, (list, str)):
        return len(input_ids)
    raise TypeError(f"object of type '{type(input_ids).__name__}' has no len()")


def _check_configured_context_budget(prompt_tokens: int, max_tokens: int):
    """Reject a request exceeding the configured KV limit, allowing equality."""
    context_limit = get_configured_context_limit()
    requested_tokens = prompt_tokens + max(0, int(max_tokens or 0))
    if context_limit is not None and requested_tokens > context_limit:
        raise PromptTooLongError(
            "Request needs "
            f"{requested_tokens} context tokens "
            f"({prompt_tokens} prompt + {max_tokens} max generation), "
            f"but MAX_KV_SIZE is {context_limit}."
        )


def get_quantized_kv_start():
    """Return the token offset where KV quantization begins."""
    return int(os.environ.get("QUANTIZED_KV_START", str(DEFAULT_QUANTIZED_KV_START)))


def get_top_logprobs_k():
    """Max per-token top_logprobs honored by the server (0 = disabled).

    Set via TOP_LOGPROBS_K env var. OpenAI caps this at 20. When 0, requests
    with top_logprobs>0 still succeed but the top_logprobs list stays empty.
    """
    k = int(os.environ.get("TOP_LOGPROBS_K", 0))
    return max(0, min(k, 20))


def _token_window_rate(token_times: List[float], window: int) -> Optional[float]:
    if len(token_times) < 2:
        return None
    subset = token_times[-window:]
    if len(subset) < 2:
        return None
    elapsed = subset[-1] - subset[0]
    if elapsed <= 0:
        return None
    return (len(subset) - 1) / elapsed


def _token_window_rate_first(token_times: List[float], window: int) -> Optional[float]:
    if len(token_times) < 2:
        return None
    subset = token_times[:window]
    if len(subset) < 2:
        return None
    elapsed = subset[-1] - subset[0]
    if elapsed <= 0:
        return None
    return (len(subset) - 1) / elapsed


class ServerMetricsStore:
    """Keep process-local lifetime counters and a bounded recent-success history.

    Failed requests update the failure counters and last error but are not added
    to the recent-success deque. Snapshots copy the stored top-level mappings;
    reset starts a new accounting interval.
    """

    def __init__(self, history_limit: int = METRICS_HISTORY_LIMIT):
        """Create counters with a bounded recent-success history."""
        self.history_limit = history_limit
        self.reset()

    def reset(self):
        """Reset counters and history for a new accounting interval."""
        self.started_at = time.time()
        self._lock = Lock()
        self._latest: Optional[MetricsEnvelope] = None
        self._recent: deque[MetricsEnvelope] = deque(maxlen=self.history_limit)
        self._requests_started = 0
        self._requests_completed = 0
        self._requests_failed = 0
        self._streaming_requests = 0
        self._in_flight = 0
        self._prompt_tokens_total = 0
        self._completion_tokens_total = 0
        self._generated_tokens_total = 0
        self._request_time_total_s = 0.0
        self._decode_time_total_s = 0.0
        self._last_request_at: Optional[float] = None
        self._last_error: Optional[RequestFailure] = None

    def begin_request(self, *, endpoint: str, model: str, stream: bool):
        """Count a request start and take an in-flight slot."""
        with self._lock:
            self._requests_started += 1
            self._in_flight += 1
            if stream:
                self._streaming_requests += 1
            in_flight = self._in_flight
        logger.info(
            "Request started: endpoint=%s model=%s stream=%s in_flight=%d",
            endpoint,
            model,
            stream,
            in_flight,
        )

    def record_success(self, envelope: MetricsEnvelope):
        """Record one completed request and its token and time totals."""
        payload: MetricsEnvelope = {**envelope}
        with self._lock:
            self._requests_completed += 1
            self._in_flight = max(0, self._in_flight - 1)
            self._latest = payload
            self._recent.append(payload)
            self._last_request_at = payload.get("timestamp_unix")
            self._prompt_tokens_total += int(payload.get("prompt_tokens") or 0)
            self._completion_tokens_total += int(payload.get("completion_tokens") or 0)
            self._generated_tokens_total += int(payload.get("generated_tokens") or 0)
            self._request_time_total_s += float(payload.get("request_elapsed_s") or 0.0)
            self._decode_time_total_s += float(payload.get("decode_elapsed_s") or 0.0)
            in_flight = self._in_flight
        logger.info(
            "Request completed: endpoint=%s model=%s stream=%s backend=%s "
            "prompt_tokens=%d generated_tokens=%d elapsed=%.3fs "
            "prefill=%.1f tok/s decode=%.1f tok/s finish_reason=%s in_flight=%d",
            payload.get("endpoint"),
            payload.get("model"),
            payload.get("stream"),
            payload.get("backend"),
            int(payload.get("prompt_tokens") or 0),
            int(payload.get("generated_tokens") or 0),
            float(payload.get("request_elapsed_s") or 0.0),
            float(payload.get("prefill_tok_s") or 0.0),
            float(payload.get("decode_tok_s") or 0.0),
            payload.get("finish_reason"),
            in_flight,
        )

    def record_failure(self, *, endpoint: str, model: str, stream: bool, error: str):
        """Record one failed request and keep it as the last error."""
        with self._lock:
            self._requests_failed += 1
            self._in_flight = max(0, self._in_flight - 1)
            self._last_request_at = time.time()
            self._last_error = {
                "timestamp_unix": self._last_request_at,
                "endpoint": endpoint,
                "model": model,
                "stream": bool(stream),
                "error": error,
            }
            in_flight = self._in_flight
        logger.warning(
            "Request failed: endpoint=%s model=%s stream=%s error=%s in_flight=%d",
            endpoint,
            model,
            stream,
            error,
            in_flight,
        )

    def snapshot(self) -> MetricsSnapshot:
        """Return copied latest and recent records plus aggregate counters."""
        with self._lock:
            latest: MetricsEnvelope | None = (
                {**self._latest} if self._latest is not None else None
            )
            recent: list[MetricsEnvelope] = [
                {**item} for item in list(self._recent)[-METRICS_RECENT_LIMIT:]
            ]
            requests_completed = self._requests_completed
            avg_request_time = (
                self._request_time_total_s / requests_completed
                if requests_completed > 0
                else 0.0
            )
            avg_request_tok_s = (
                self._completion_tokens_total / self._request_time_total_s
                if self._request_time_total_s > 0
                else 0.0
            )
            avg_decode_tok_s = (
                self._generated_tokens_total / self._decode_time_total_s
                if self._decode_time_total_s > 0
                else 0.0
            )
            last_error: RequestFailure | None = (
                {**self._last_error} if self._last_error is not None else None
            )
            last_request_at = self._last_request_at
            return {
                "latest": latest,
                "recent": recent,
                "summary": {
                    "uptime_s": max(0.0, time.time() - self.started_at),
                    "requests_started": self._requests_started,
                    "requests_completed": requests_completed,
                    "requests_failed": self._requests_failed,
                    "streaming_requests": self._streaming_requests,
                    "in_flight": self._in_flight,
                    "prompt_tokens_total": self._prompt_tokens_total,
                    "completion_tokens_total": self._completion_tokens_total,
                    "generated_tokens_total": self._generated_tokens_total,
                    "avg_request_time_s": avg_request_time,
                    "avg_request_tok_s": avg_request_tok_s,
                    "avg_decode_tok_s": avg_decode_tok_s,
                    "last_request_at": last_request_at,
                    "last_error": last_error,
                },
            }


def _prompt_eval_time_from_tps(
    prompt_tokens: int, prompt_tps: Optional[float]
) -> Optional[float]:
    if prompt_tokens <= 0 or prompt_tps is None or prompt_tps <= 0:
        return None
    return prompt_tokens / prompt_tps


def _decode_elapsed_from_metrics(
    token_times: List[float],
    generated_tokens: int,
    generation_tps: Optional[float],
) -> Optional[float]:
    if len(token_times) >= 2:
        elapsed = token_times[-1] - token_times[0]
        if elapsed > 0:
            return elapsed
    if generated_tokens > 0 and generation_tps is not None and generation_tps > 0:
        return generated_tokens / generation_tps
    return None


def build_metrics_envelope(
    *,
    endpoint: str,
    model: str,
    stream: bool,
    backend: str,
    prompt_tokens: int,
    completion_tokens: int,
    generated_tokens: int,
    request_elapsed_s: float,
    request_started_s: float,
    token_times: Optional[List[float]] = None,
    prompt_tps: Optional[float] = None,
    generation_tps: Optional[float] = None,
    peak_memory_gb: Optional[float] = None,
    finish_reason: Optional[str] = None,
    image_count: int = 0,
    thinking_enabled: bool = False,
) -> MetricsEnvelope:
    """Assemble one request's metrics record.

    ``ttft_s`` is measured from ``request_started_s`` to the first token;
    ``reasoning_tokens`` is the part of ``generated_tokens`` outside
    ``completion_tokens``. Decode rates use measured token times and fall back
    to the reported generation rate.
    """
    token_times = token_times or []
    ttft_s = max(0.0, token_times[0] - request_started_s) if token_times else None
    decode_elapsed_s = _decode_elapsed_from_metrics(
        token_times, generated_tokens, generation_tps
    )
    decode_tok_s = None
    if generation_tps is not None and generation_tps > 0:
        decode_tok_s = generation_tps
    elif decode_elapsed_s is not None and decode_elapsed_s > 0 and generated_tokens > 0:
        decode_tok_s = generated_tokens / decode_elapsed_s
    prompt_eval_time_s = _prompt_eval_time_from_tps(prompt_tokens, prompt_tps)
    request_tok_s = (
        completion_tokens / request_elapsed_s if request_elapsed_s > 0 else 0.0
    )
    return {
        "timestamp_unix": time.time(),
        "endpoint": endpoint,
        "model": model,
        "stream": bool(stream),
        "backend": backend,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "generated_tokens": int(generated_tokens),
        "reasoning_tokens": max(0, int(generated_tokens) - int(completion_tokens)),
        "total_tokens": int(prompt_tokens) + int(completion_tokens),
        "prompt_eval_time_s": prompt_eval_time_s,
        "prefill_tok_s": prompt_tps,
        "ttft_s": ttft_s,
        "decode_elapsed_s": decode_elapsed_s,
        "request_elapsed_s": request_elapsed_s,
        "request_tok_s": request_tok_s,
        "decode_tok_s": decode_tok_s,
        "sliding_decode_tok_s_first_32": _token_window_rate_first(token_times, 32),
        "sliding_decode_tok_s_first_64": _token_window_rate_first(token_times, 64),
        "sliding_decode_tok_s_last_32": _token_window_rate(token_times, 32),
        "sliding_decode_tok_s_last_64": _token_window_rate(token_times, 64),
        "peak_memory_gb": peak_memory_gb,
        "finish_reason": finish_reason,
        "image_count": int(image_count),
        "thinking_enabled": bool(thinking_enabled),
    }


def load_model_resources(model_path: str):
    """
    Loads model, processor, and config based on paths.
    Handles potential loading errors.
    """
    try:
        logger.info("Loading model: %s", model_path)
        # Use the load function from utils.py which handles path resolution and loading
        trust_remote_code = (
            os.environ.get("MLX_TRUST_REMOTE_CODE", "false").lower() == "true"
        )
        model, processor = load(
            model_path,
            trust_remote_code=trust_remote_code,
            expert_cache_gb=get_expert_cache_gb(),
            max_kv_size=get_configured_context_limit(),
        )
        model = require_generation_model(model)
        config = model.config
        # The registered MLX model and processor expose dynamic child attributes.
        # Prove their callable surfaces before narrowing this loader boundary.
        assert callable(getattr(model, "get_input_embeddings", None))
        assert callable(getattr(model, "language_model", None))
        assert isinstance(getattr(config, "model_type", None), str)
        if not isinstance(processor, ProcessorLike):
            processor = resolve_tokenizer(processor)
        logger.info("Model and processor loaded successfully.")
        return model, processor, config
    except Exception as e:
        logger.exception("Error loading model %s: %s", model_path, e)
        raise HTTPException(status_code=400, detail=f"Failed to load model: {e}")


# =============================================================================
# ResponseGenerator - Concurrent Request Handling with Threaded Batching
# =============================================================================


class ServerTemplateKwargs(TypedDict, total=False, extra_items=ReadOnly[JSONValue]):
    """Chat-template options derived from a generation request."""

    enable_thinking: bool
    reasoning: bool
    reasoning_effort: str
    reasoning_strength: str
    thinking_budget: int
    thinking_start_token: str
    thinking_end_token: str


@dataclass
class GenerationArguments:
    """Arguments for a generation request."""

    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = 0
    min_p: float = 0.0
    top_n_sigma: float = 0.0
    p_less: bool = False
    typical_p: float = 1.0
    seed: Optional[int] = None
    logprobs: bool = False
    repetition_penalty: Optional[float] = None
    repetition_context_size: Optional[int] = DEFAULT_REPETITION_CONTEXT_SIZE
    presence_penalty: Optional[float] = None
    presence_context_size: Optional[int] = DEFAULT_REPETITION_CONTEXT_SIZE
    frequency_penalty: Optional[float] = None
    frequency_context_size: Optional[int] = DEFAULT_REPETITION_CONTEXT_SIZE
    logit_bias: Optional[dict[int, float]] = None
    enable_thinking: bool = DEFAULT_ENABLE_THINKING
    reasoning: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    thinking_budget: Optional[int] = None
    thinking_start_token: Optional[str] = None
    thinking_end_token: Optional[str] = None
    skip_special_tokens: bool = True

    def __post_init__(self):
        """Clamp the sampling temperature to the supported minimum."""
        self.temperature = clamp_temperature(self.temperature)

    def batch_sampling_key(self) -> tuple[float | int | bool | None, ...]:
        """Settings shared by the continuous batch sampler and logprob calculation."""
        return (
            self.temperature,
            self.top_p,
            self.top_k,
            self.min_p,
            self.top_n_sigma,
            self.p_less,
            self.typical_p,
            self.seed,
            self.logprobs,
        )

    def to_generate_kwargs(self) -> GenerateTextKwargs:
        """Convert to kwargs dict for generate()/stream_generate()."""
        kw: GenerateTextKwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "top_n_sigma": self.top_n_sigma,
            "p_less": self.p_less,
            "typical_p": self.typical_p,
            "enable_thinking": self.enable_thinking,
        }
        if self.seed is not None:
            kw["seed"] = self.seed
        if self.repetition_penalty is not None:
            kw["repetition_penalty"] = self.repetition_penalty
        if self.repetition_context_size is not None:
            kw["repetition_context_size"] = self.repetition_context_size
        if self.presence_penalty is not None:
            kw["presence_penalty"] = self.presence_penalty
        if self.presence_context_size is not None:
            kw["presence_context_size"] = self.presence_context_size
        if self.frequency_penalty is not None:
            kw["frequency_penalty"] = self.frequency_penalty
        if self.frequency_context_size is not None:
            kw["frequency_context_size"] = self.frequency_context_size
        if self.logit_bias is not None:
            kw["logit_bias"] = self.logit_bias
        if self.thinking_budget is not None:
            kw["thinking_budget"] = self.thinking_budget
        if self.thinking_start_token is not None:
            kw["thinking_start_token"] = self.thinking_start_token
        if self.thinking_end_token is not None:
            kw["thinking_end_token"] = self.thinking_end_token
        return kw

    def to_template_kwargs(self) -> ServerTemplateKwargs:
        """Convert to kwargs for apply_chat_template()."""
        kw: ServerTemplateKwargs = {"enable_thinking": self.enable_thinking}
        if self.reasoning is not None:
            kw["reasoning"] = self.reasoning
        if self.reasoning_effort is not None:
            kw["reasoning_effort"] = self.reasoning_effort
            # Alias for chat templates that read reasoning_strength instead.
            kw["reasoning_strength"] = self.reasoning_effort
        if self.thinking_budget is not None:
            kw["thinking_budget"] = self.thinking_budget
        if self.thinking_start_token is not None:
            kw["thinking_start_token"] = self.thinking_start_token
        if self.thinking_end_token is not None:
            kw["thinking_end_token"] = self.thinking_end_token
        return kw


@dataclass
class GenerationContext:
    """Context returned when a request is queued."""

    uid: int
    prompt_tokens: int


@dataclass
class QueuedGenerationRequest:
    """Preprocessed generation request waiting for the GPU worker."""

    rqueue: Queue[GenerationEvent]
    raw_inputs: ProcessedInputs
    prompt_tokens: int
    args: GenerationArguments
    thinking_budget_criteria: Optional[ThinkingBudgetCriteria] = None
    images: Optional[list[ImageSource]] = None
    request_id: Optional[str] = None
    queued_at: float = field(default_factory=time.perf_counter)


@dataclass
class GenerationMetrics:
    """Accumulate generation counts and timing while consuming output chunks.

    A chunk can carry an explicit token count or a cumulative generation count;
    zero-token chunks do not create timing samples. ``rate`` uses tokens after
    the first nonempty chunk divided by the time between the first and last
    nonempty chunks, and is unavailable with fewer than two timestamps.
    """

    token_times: List[float] = field(default_factory=list[float])
    peak_memory: float = 0.0
    cached_tokens: int = 0
    prompt_tps: Optional[float] = None
    generation_tps: Optional[float] = None
    last_chunk_at: Optional[float] = None
    last_chunk_rate: Optional[float] = None
    generated_tokens: int = 0
    first_chunk_tokens: int = 0

    def record_chunk(self, chunk: StreamingToken | GenerationResult) -> Optional[float]:
        """Record one output chunk and return its instantaneous token rate."""
        now = getattr(chunk, "emitted_at", None) or time.perf_counter()
        token_count = getattr(chunk, "token_count", None)
        if token_count is None:
            generation_tokens = getattr(chunk, "generation_tokens", None)
            token_count = (
                max(0, int(generation_tokens) - self.generated_tokens)
                if generation_tokens is not None
                else 1
            )
        emitted_tokens = max(0, int(token_count or 0))
        self.last_chunk_rate = None
        if emitted_tokens > 0:
            if not self.token_times:
                self.first_chunk_tokens = emitted_tokens
            self.generated_tokens += emitted_tokens
            if self.last_chunk_at is not None and now > self.last_chunk_at:
                self.last_chunk_rate = emitted_tokens / (now - self.last_chunk_at)
            self.last_chunk_at = now
            self.token_times.append(now)
        self.record_result(chunk)
        return self.last_chunk_rate

    @property
    def rate(self) -> Optional[float]:
        """Return tokens per second after the first chunk, or ``None``."""
        if len(self.token_times) < 2:
            return None
        elapsed = self.token_times[-1] - self.token_times[0]
        measured_tokens = self.generated_tokens - self.first_chunk_tokens
        return measured_tokens / elapsed if elapsed > 0 else None

    def record_result(self, result: StreamingToken | GenerationResult) -> None:
        """Keep the largest peak memory and cached-token count, latest rates."""
        self.peak_memory = max(
            self.peak_memory, float(getattr(result, "peak_memory", 0.0) or 0.0)
        )
        prompt_tps = getattr(result, "prompt_tps", None)
        if prompt_tps is not None:
            self.prompt_tps = prompt_tps
        generation_tps = getattr(result, "generation_tps", None)
        if generation_tps is not None:
            self.generation_tps = generation_tps
        cached_tokens = getattr(result, "cached_tokens", None)
        if cached_tokens is not None:
            self.cached_tokens = max(self.cached_tokens, int(cached_tokens))


@dataclass
class StreamingToken:
    """A single token response during streaming generation.

    Diffusion models stream block-by-block: one StreamingToken per denoised
    block, with ``token_count`` carrying the number of tokens in the block.
    """

    text: str
    token: int
    logprobs: float | None
    finish_reason: Optional[str]
    peak_memory: float = 0.0
    prompt_tps: Optional[float] = None
    generation_tps: Optional[float] = None
    top_logprobs: Optional[List[Tuple[int, float]]] = None
    cached_tokens: int = 0
    token_count: int = 1
    emitted_at: Optional[float] = None


class _TokenIterator:
    """Closeable iterator over queued tokens for one generation request.

    close() cancels unfinished requests and is safe while another thread is
    blocked in __next__ waiting for the next token. A queue timeout requests
    cancellation and raises RuntimeError. A finish-reason token marks the
    iterator ended, so closing it afterward does not request cancellation.
    """

    def __init__(
        self,
        rqueue: Queue[GenerationEvent],
        uid: int,
        cancel_fn: Callable[[int], None],
        queue_timeout: float | None,
    ):
        self._rqueue = rqueue
        self._uid = uid
        self._cancel_fn = cancel_fn
        self._queue_timeout = queue_timeout
        self._ended = False
        self._closed = False
        self._lock = Lock()

    def __iter__(self):
        return self

    def __next__(self) -> StreamingToken:
        if self._ended:
            raise StopIteration
        try:
            item = self._rqueue.get(timeout=self._queue_timeout)
        except QueueEmpty as exc:
            # Consumer is stalled or upstream is wedged — treat as cancel.
            self.close()
            label = (
                "without a timeout"
                if self._queue_timeout is None
                else f"for {self._queue_timeout:g}s"
            )
            raise RuntimeError(
                "Timed out waiting "
                f"{label} for the next generated token. "
                "Increase MLX_VLM_TOKEN_QUEUE_TIMEOUT for long "
                "prefills, or reduce the prompt size."
            ) from exc
        if item is None:
            self._ended = True
            raise StopIteration
        if isinstance(item, Exception):
            self._ended = True
            raise item
        if not isinstance(item, StreamingToken):
            raise RuntimeError(
                "Generation queue emitted context after token streaming began"
            )
        if item.finish_reason:
            self._ended = True
        return item

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if not self._ended:
            self._cancel_fn(self._uid)

    def __del__(self):
        # Mirror generator semantics: implicit close on GC.
        try:
            self.close()
        except Exception:
            pass


class ResponseGenerator:
    """
    Continuous batching for concurrent requests via a single GPU thread.

    A dedicated thread owns all GPU work (BatchGenerator). FastAPI async
    handlers submit requests to a queue and read tokens back from
    per-request queues. Multiple requests are batched together for
    higher throughput — same pattern as mlx-lm's server.
    """

    def __init__(
        self,
        model_path: str,
        vision_cache: "VisionFeatureCache | None" = None,
        kv_bits: float | None = None,
        kv_group_size: int = DEFAULT_KV_GROUP_SIZE,
        quantized_kv_start: int = DEFAULT_QUANTIZED_KV_START,
        top_logprobs_k: int = 0,
        prefill_step_size: Optional[int] = None,
    ):
        """Configure the worker and start the daemon thread that loads the model.

        ``prefill_step_size=None`` defers to ``PREFILL_STEP_SIZE``. Request and
        cancellation queues, the readiness event and tokenizer lock are shared
        with the worker thread.
        """
        self.model_path = model_path
        self.model: GenerationModelLike | None = None
        self.processor: ProcessorLike | TokenizerLike | None = None
        self.config: GenerationConfigLike | None = None
        self.stop_tokens: set[int] = set()
        self.vision_cache = vision_cache
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.quantized_kv_start = quantized_kv_start
        self.top_logprobs_k = top_logprobs_k
        self.prefill_step_size = (
            get_prefill_step_size()
            if prefill_step_size is None
            else int(prefill_step_size)
        )
        self.tokenizer: TokenizerLike | None = None
        self.requests: Queue[QueuedGenerationRequest | None] = Queue()
        self._deferred_request: QueuedGenerationRequest | None = None
        self._stop = False
        self._requests_lock = Lock()
        self._ready = Event()
        self._load_error: Optional[Exception] = None
        self._cancelled: set[int] = set()
        self._cancel_lock = Lock()
        self._tokenizer_lock = Lock()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def _effective_prefill_step_size(self) -> int:
        prefill_step_size = getattr(self, "prefill_step_size", None)
        return (
            get_prefill_step_size()
            if prefill_step_size is None
            else int(prefill_step_size)
        )

    def stop_and_join(self, timeout: float = 10.0):
        """Request worker shutdown and wait up to ``timeout`` seconds.

        A worker still draining after the timeout continues in the background.
        """
        with self._requests_lock:
            if not self._stop:
                self._stop = True
                self.requests.put(None)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.info(
                "Generation thread still draining in-flight requests; "
                "letting it finish in the background."
            )

    def wait_until_ready(self, timeout: Optional[float] = None):
        """Block until the worker loads its model and raise any load error.

        Returns the model, processor and config, or raises ``RuntimeError`` if
        the wait times out or resources are missing.
        """
        if not self._ready.wait(timeout):
            raise RuntimeError("Timed out waiting for generation thread to load model.")
        if self._load_error is not None:
            raise self._load_error
        if self.model is None or self.processor is None or self.config is None:
            raise RuntimeError("Generation worker became ready without model resources")
        return self.model, self.processor, self.config

    def _cancel(self, uid: int):
        with self._cancel_lock:
            self._cancelled.add(uid)

    def _drain_cancellations(self) -> set[int]:
        with self._cancel_lock:
            pending, self._cancelled = self._cancelled, set()
            return pending

    def _initialize_model(self):
        model, processor, config = load_model_resources(self.model_path)

        stop_tokens = set(
            resolve_eos_token_ids(
                getattr(config, "eos_token_id", None),
                getattr(processor, "tokenizer", processor),
            )
        )
        stop_tokens.update(getattr(processor, "additional_eos_token_ids", ()))

        if os.environ.get("MLX_VLM_DRAFT_MODEL") or os.environ.get(
            "MLX_VLM_DRAFT_KIND"
        ):
            raise ValueError(
                "External speculative drafters are outside the medical runtime"
            )

        self.model = model
        self.processor = processor
        self.config = config
        self.stop_tokens = stop_tokens
        self.tokenizer = resolve_tokenizer(processor)

    def generate(
        self,
        prompt: str,
        images: Optional[list[ImageSource]] = None,
        args: Optional[GenerationArguments] = None,
    ) -> Tuple[GenerationContext, "_TokenIterator"]:
        """Prepare and enqueue a request, then wait for its worker context.

        The returned iterator can cancel an unfinished request when closed. The
        token queue timeout applies after the context is returned; the initial
        wait for that context has no timeout in this method. Admission after
        shutdown starts raises RuntimeError instead of queuing behind the stop marker.
        """
        self.wait_until_ready()
        args = args or GenerationArguments(max_tokens=get_server_max_tokens())
        rqueue: Queue[GenerationEvent] = Queue()
        request_started_at = time.perf_counter()

        # CPU preprocessing and thinking-token resolution share tokenizer state.
        # Keep both on the caller side and serialize them so the GPU worker never
        # races request threads through the mutable fast-tokenizer backend.
        tokenizer_lock = getattr(self, "_tokenizer_lock", None)
        with tokenizer_lock if tokenizer_lock is not None else nullcontext():
            raw_inputs = self.cpu_preprocess(prompt, images)
            request_ids = raw_inputs.get("input_ids")
            if args.thinking_budget is not None:
                assert isinstance(request_ids, mx.array)
            thinking_budget_criteria = self._make_thinking_budget_criteria(
                args, request_ids
            )
        prompt_tokens = count_prompt_tokens(raw_inputs)
        _check_configured_context_budget(prompt_tokens, args.max_tokens)

        request_id = f"{id(rqueue):x}"
        queued_request = QueuedGenerationRequest(
            rqueue=rqueue,
            raw_inputs=raw_inputs,
            prompt_tokens=prompt_tokens,
            args=args,
            thinking_budget_criteria=thinking_budget_criteria,
            images=images,
            request_id=request_id,
            queued_at=request_started_at,
        )
        self._enqueue_request(queued_request)
        logger.info(
            "Generation queued: request=%s prompt_tokens=%d max_tokens=%d " "images=%d",
            request_id,
            prompt_tokens,
            args.max_tokens,
            len(images or []),
        )
        # Block until the GPU thread sends back the context
        ctx = rqueue.get()
        if isinstance(ctx, Exception):
            raise ctx
        # The worker sends exactly one context before streaming any tokens.
        assert isinstance(ctx, GenerationContext)

        return ctx, _TokenIterator(
            rqueue, ctx.uid, self._cancel, get_token_queue_timeout()
        )

    def _enqueue_request(self, request: QueuedGenerationRequest) -> None:
        """Order request admission atomically with the shutdown sentinel."""
        with self._requests_lock:
            if self._stop:
                raise RuntimeError("Generation worker is shutting down")
            self.requests.put(request)

    def cpu_preprocess(
        self,
        prompt: str,
        images: list[ImageSource] | None = None,
    ) -> ProcessedInputs:
        """CPU-only: tokenize text, load/resize images. Thread-safe."""
        assert self.model is not None
        assert self.processor is not None
        add_special_tokens = (
            getattr(self.processor, "chat_template", None) is None
            if self.model.config.model_type
            in ["gemma3", "gemma3n", "gemma4", "gemma4_unified"]
            else True
        )
        image_token_index = getattr(self.model.config, "image_token_index", None)
        return prepare_inputs(
            self.processor,
            images=images,
            prompts=prompt,
            image_token_index=image_token_index,
            add_special_tokens=add_special_tokens,
        )

    # -- internals --

    @staticmethod
    def _request_log_id(request: QueuedGenerationRequest) -> str:
        if request.request_id is None:
            request.request_id = f"{id(request.rqueue):x}"
        return request.request_id

    def _log_prefill_started(
        self, request: QueuedGenerationRequest, *, backend: str
    ) -> RequestLogState:
        request_id = self._request_log_id(request)
        now = time.perf_counter()
        logger.info(
            "Prefill started: request=%s backend=%s prompt_tokens=%d " "images=%d",
            request_id,
            backend,
            request.prompt_tokens,
            len(request.images or []),
        )
        return {
            "request_id": request_id,
            "queued_at": request.queued_at,
            "prefill_started_at": now,
            "prefill_processed": -1,
            "generated_tokens": 0,
            "decode_started_at": None,
            "last_token_at": None,
        }

    def _log_prefill_progress(
        self, batch_gen: BatchGenerator, active: dict[int, ActiveGeneration]
    ) -> None:
        """Report each chunked-prefill step without changing generation output."""
        prompt_batch = getattr(batch_gen, "_prompt_batch", None)
        if prompt_batch is None:
            return
        processed_columns = int(
            getattr(prompt_batch, "_processed_prompt_columns", 0) or 0
        )
        remaining = getattr(prompt_batch, "_inputs_embeds", None)
        remaining_columns = (
            int(remaining.shape[1])
            if isinstance(remaining, mx.array) and remaining.ndim >= 2
            else 0
        )
        total_columns = processed_columns + remaining_columns
        uids = list(getattr(prompt_batch, "uids", []))
        suffix_lengths = list(getattr(prompt_batch, "_suffix_lens", []))
        cached_tokens = list(getattr(prompt_batch, "_cached_tokens_per_row", []))
        left_padding = list(getattr(prompt_batch, "_left_padding_per_row", []))
        right_padding = getattr(prompt_batch, "_right_pad_per_row", None)

        for index, uid in enumerate(uids):
            info = active.get(uid)
            if info is None:
                continue
            suffix_len = (
                int(suffix_lengths[index])
                if index < len(suffix_lengths)
                else total_columns
            )
            cached = int(cached_tokens[index]) if index < len(cached_tokens) else 0
            if right_padding is not None:
                suffix_processed = min(suffix_len, processed_columns)
            else:
                pad = int(left_padding[index]) if index < len(left_padding) else 0
                suffix_processed = min(suffix_len, max(0, processed_columns - pad))
            processed = cached + suffix_processed
            total = cached + suffix_len
            if processed <= int(info.get("prefill_processed", -1)):
                continue
            info["prefill_processed"] = processed
            percent = 100.0 * processed / total if total > 0 else 100.0
            logger.info(
                "Prefill progress: request=%s tokens=%d/%d (%.1f%%)",
                info.get("request_id", uid),
                processed,
                total,
                percent,
            )

    @staticmethod
    def _log_prefill_completed(
        uid: int, info: RequestLogState, prompt_response: PromptProgress
    ) -> None:
        prompt_tokens = int(getattr(prompt_response, "prompt_tokens", 0) or 0)
        cached_tokens = int(getattr(prompt_response, "cached_tokens", 0) or 0)
        prompt_tps = float(getattr(prompt_response, "prompt_tps", 0.0) or 0.0)
        prompt_time = float(getattr(prompt_response, "prompt_time", 0.0) or 0.0)
        if prompt_time <= 0 and prompt_tps > 0:
            prompt_time = prompt_tokens / prompt_tps
        info["prefill_processed"] = prompt_tokens
        logger.info(
            "Prefill completed: request=%s prompt_tokens=%d cached_tokens=%d "
            "elapsed=%.3fs rate=%.1f tok/s",
            info.get("request_id", uid),
            prompt_tokens,
            cached_tokens,
            prompt_time,
            prompt_tps,
        )

    @staticmethod
    def _log_decode_progress(
        uid: int,
        info: RequestLogState,
        *,
        token: int,
        text: str,
        finish_reason: Optional[str],
        token_count: int = 1,
    ) -> float:
        now = time.perf_counter()
        previous_tokens = int(info.get("generated_tokens", 0) or 0)
        emitted_tokens = max(0, int(token_count or 0))
        generated_tokens = previous_tokens + emitted_tokens
        info["generated_tokens"] = generated_tokens
        request_id = info.get("request_id", uid)

        previous_token_at = info.get("last_token_at")
        token_rate = None
        if emitted_tokens > 0:
            if previous_token_at is not None and now > previous_token_at:
                token_rate = emitted_tokens / (now - previous_token_at)
            info["last_token_at"] = now
        progress_rate_text = "n/a" if token_rate is None else f"{token_rate:.1f} tok/s"

        decode_started_at = info.get("decode_started_at")
        if decode_started_at is None:
            decode_started_at = now
            info["decode_started_at"] = now
            queued_at = float(info.get("queued_at", now) or now)
            logger.info(
                "Decode started: request=%s time_to_first_token=%.3fs",
                request_id,
                max(0.0, now - queued_at),
            )

        elapsed = max(0.0, now - decode_started_at)
        first_chunk_tokens = int(info.get("decode_first_chunk_tokens", 0) or 0)
        if first_chunk_tokens == 0:
            first_chunk_tokens = emitted_tokens
            info["decode_first_chunk_tokens"] = emitted_tokens
        measured_tokens = max(0, generated_tokens - first_chunk_tokens)
        rate = measured_tokens / elapsed if elapsed > 0 else 0.0
        interval = get_log_progress_interval()
        crossed_interval = interval > 0 and (
            generated_tokens // interval > previous_tokens // interval
        )
        debug_enabled = logger.isEnabledFor(logging.DEBUG)
        if debug_enabled:
            logger.debug(
                "Decode progress: request=%s generated_tokens=%d elapsed=%.3fs "
                "rate=%s token_number=%d token_id=%s text=%r",
                request_id,
                generated_tokens,
                elapsed,
                progress_rate_text,
                generated_tokens,
                token,
                text,
            )

        if finish_reason is not None:
            logger.info(
                "Decode completed: request=%s generated_tokens=%d elapsed=%.3fs "
                "rate=%.1f tok/s finish_reason=%s",
                request_id,
                generated_tokens,
                elapsed,
                rate,
                finish_reason,
            )
        elif crossed_interval and not debug_enabled:
            logger.info(
                "Decode progress: request=%s generated_tokens=%d elapsed=%.3fs "
                "rate=%s",
                request_id,
                generated_tokens,
                elapsed,
                progress_rate_text,
            )
        return now

    def _make_sampler(
        self, args: GenerationArguments
    ) -> Optional[Callable[[mx.array], mx.array]]:
        if args.temperature == 0:
            return None
        return _PositionedTargetSampler(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            min_p=args.min_p,
            top_n_sigma=args.top_n_sigma,
            p_less=args.p_less,
            typical_p=args.typical_p,
        )

    def _make_logits_processors(
        self, args: GenerationArguments, input_ids: Optional[mx.array] = None
    ) -> List[Callable[[mx.array, mx.array], mx.array]]:
        processors = make_logits_processors(
            args.logit_bias,
            args.repetition_penalty,
            args.repetition_context_size,
            args.presence_penalty,
            args.presence_context_size,
            args.frequency_penalty,
            args.frequency_context_size,
        )
        return processors

    def _thinking_token_ids(self, args: GenerationArguments) -> Tuple[int, int]:
        assert self.tokenizer is not None
        tokenizer = self.tokenizer
        if not isinstance(tokenizer, TokenEncoder):
            raise TypeError("Thinking token detection requires a text encoder")
        thinking_start_token = args.thinking_start_token or DEFAULT_THINKING_START_TOKEN
        thinking_end_token = args.thinking_end_token or DEFAULT_THINKING_END_TOKEN
        thinking_start_token_id = tokenizer.encode(
            thinking_start_token, add_special_tokens=False
        )[-1]
        thinking_end_token_id = tokenizer.encode(
            thinking_end_token, add_special_tokens=False
        )[-1]
        return thinking_start_token_id, thinking_end_token_id

    def _prompt_has_open_thinking(
        self, args: GenerationArguments, input_ids: mx.array
    ) -> bool:
        if not args.enable_thinking:
            return False
        thinking_start_token_id, thinking_end_token_id = self._thinking_token_ids(args)
        tokens = array_int_list(input_ids.flatten())
        try:
            last_start = len(tokens) - 1 - tokens[::-1].index(thinking_start_token_id)
        except ValueError:
            return False
        try:
            last_end = len(tokens) - 1 - tokens[::-1].index(thinking_end_token_id)
        except ValueError:
            last_end = -1
        return last_start > last_end

    def _make_thinking_budget_criteria(
        self, args: GenerationArguments, input_ids: ProcessorValue
    ) -> Optional[ThinkingBudgetCriteria]:
        if args.thinking_budget is None:
            return None
        assert isinstance(input_ids, mx.array)
        tokenizer = self.tokenizer
        if not isinstance(tokenizer, TokenEncoder):
            raise TypeError("Thinking budgets require a tokenizer encode method")
        thinking_start_token = args.thinking_start_token or DEFAULT_THINKING_START_TOKEN
        thinking_end_token = args.thinking_end_token or DEFAULT_THINKING_END_TOKEN
        prompt_preopens_thinking = self._prompt_has_open_thinking(args, input_ids)
        return ThinkingBudgetCriteria(
            tokenizer=tokenizer,
            thinking_budget=args.thinking_budget,
            thinking_end_token=thinking_end_token,
            thinking_start_token=thinking_start_token,
            enable_thinking=args.enable_thinking,
            prompt_preopens_thinking=prompt_preopens_thinking,
        )

    def _gpu_embed(
        self,
        raw_inputs: ProcessedInputs,
        images: list[ImageSource] | None = None,
    ) -> Tuple[mx.array, PromptInputs]:
        """GPU-only: run vision encoder if needed. Must run on GPU thread."""
        assert self.model is not None
        input_ids = raw_inputs.get("input_ids")
        pixel_values = raw_inputs.get("pixel_values")
        mask = raw_inputs.get("attention_mask")
        assert isinstance(input_ids, mx.array)
        assert mask is None or isinstance(mask, mx.array)
        assert is_pixel_values(pixel_values)
        data_kwargs: dict[str, ModelArgument] = {
            k: v
            for k, v in raw_inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }
        # Pass vision cache for image feature caching
        if (
            pixel_values is not None
            and self.vision_cache is not None
            and images is not None
        ):
            data_kwargs["vision_cache"] = self.vision_cache
            data_kwargs["_image_key"] = images

        # Always call get_input_embeddings — BatchGenerator requires inputs_embeds
        embed = self.model.get_input_embeddings(
            input_ids, pixel_values, mask=mask, **data_kwargs
        )
        # Remove cache kwargs before passing to BatchGenerator
        data_kwargs.pop("vision_cache", None)
        data_kwargs.pop("_image_key", None)
        gen_kwargs: PromptInputs = {
            **data_kwargs,
            **{k: v for k, v in embed.to_dict().items() if v is not None},
        }
        return input_ids, gen_kwargs

    def _collect_pending_requests(
        self,
        *,
        active: bool,
        capacity: Optional[int] = None,
        idle_timeout: float = 0.1,
        coalesce_s: float = 0.0,
        sampling_key: tuple[float | int | bool | None, ...] | None = None,
    ):
        """Collect the first queued request, then drain immediately available peers.

        When ``capacity`` is set, admit at most ``capacity`` new requests and leave
        the rest queued (backpressure), so the running batch never exceeds
        ``--max-num-seqs`` concurrent sequences.
        """
        pending: list[QueuedGenerationRequest] = []
        should_stop = False

        def append_item(item: QueuedGenerationRequest | None):
            nonlocal should_stop, sampling_key
            if item is None:
                if self._stop and not pending:
                    should_stop = True
                return
            key = item.args.batch_sampling_key()
            if sampling_key is not None and key != sampling_key:
                self._deferred_request = item
                return
            sampling_key = key
            pending.append(item)

        def _has_room():
            return capacity is None or len(pending) < capacity

        try:
            if self._deferred_request is not None and _has_room():
                item, self._deferred_request = self._deferred_request, None
                append_item(item)
            elif active:
                if _has_room():
                    append_item(self.requests.get_nowait())
            else:
                append_item(self.requests.get(timeout=idle_timeout))
        except QueueEmpty:
            pass

        if pending and coalesce_s > 0:
            time.sleep(coalesce_s)

        while not should_stop and self._deferred_request is None and _has_room():
            try:
                append_item(self.requests.get_nowait())
            except QueueEmpty:
                break

        return pending, should_stop

    def _run(self):
        try:
            self._run_impl()
        finally:
            clear_mlx_streams()

    def _run_impl(self):
        """Single GPU thread: owns BatchGenerator, runs tight next() loop."""
        try:
            self._initialize_model()
        except Exception as e:
            self._load_error = e
            self._ready.set()
            logger.exception("Error loading model in generation thread: %s", e)
            return

        self._ready.set()
        assert self.model is not None
        assert self.processor is not None

        generation_stream = mx.default_stream(mx.default_device())

        batch_gen = None
        # uid -> {rqueue, tokens, gen_kwargs}
        active: dict[int, ActiveGeneration] = {}
        sampling_key: tuple[float | int | bool | None, ...] | None = None
        max_num_seqs = get_max_num_seqs()

        while not (
            self._stop
            and not active
            and self.requests.empty()
            and self._deferred_request is None
        ):
            new_items: list[QueuedGenerationRequest] = []
            try:
                # Poll the request queue — non-blocking when generating, short
                # blocking wait when idle so we don't spin.
                active_batch = bool(active)
                if not active_batch:
                    sampling_key = None
                capacity = (
                    None if max_num_seqs is None else max(0, max_num_seqs - len(active))
                )
                new_items, should_stop = self._collect_pending_requests(
                    active=active_batch,
                    capacity=capacity,
                    sampling_key=sampling_key,
                )
                if should_stop and not active:
                    break

                # Drop abandoned requests before doing more work.
                cancelled = self._drain_cancellations()
                if cancelled and batch_gen is not None:
                    for uid in cancelled:
                        if uid in active:
                            batch_gen.remove(uid)
                            info = active.pop(uid)
                            logger.info(
                                "Generation cancelled: request=%s generated_tokens=%d",
                                info.get("request_id", uid),
                                int(info.get("generated_tokens", 0) or 0),
                            )
                            try:
                                info["rqueue"].put(None)
                            except Exception:
                                pass

                if new_items and batch_gen is not None and not active:
                    if not batch_gen.has_work:
                        batch_gen.close()
                        batch_gen = None

                for request in new_items:
                    rqueue = request.rqueue
                    raw_inputs = request.raw_inputs
                    prompt_tokens = request.prompt_tokens
                    args = request.args
                    images = request.images
                    log_state = self._log_prefill_started(
                        request, backend="continuous_batching"
                    )
                    if batch_gen is None:
                        sampling_key = args.batch_sampling_key()
                        batch_gen = BatchGenerator(
                            self.model.language_model,
                            self.processor,
                            stop_tokens=self.stop_tokens,
                            sampler=self._make_sampler(args),
                            kv_bits=self.kv_bits,
                            kv_group_size=self.kv_group_size,
                            quantized_kv_start=self.quantized_kv_start,
                            compute_logprobs=bool(args.logprobs),
                            top_logprobs_k=self.top_logprobs_k if args.logprobs else 0,
                            stream=generation_stream,
                            greedy_sampling=args.temperature == 0,
                            prefill_step_size=self._effective_prefill_step_size(),
                        )

                    # Vision encoder runs on the GPU thread; text tokenization
                    # already happened on the caller thread.
                    input_ids, gen_kwargs = self._gpu_embed(raw_inputs, images)
                    has_embeds = bool(gen_kwargs.get("inputs_embeds") is not None)
                    # Drain pending text-only prompts before inserting an
                    # embed-bearing request — multi-row PromptProcessingBatch
                    # admission expects all rows to carry inputs_embeds.
                    if has_embeds and any(
                        not (s[3] and s[3].get("inputs_embeds") is not None)
                        for s in batch_gen.unprocessed_prompts
                    ):
                        self._flush(batch_gen, active)

                    try:
                        thinking_budget_criteria = request.thinking_budget_criteria
                        (uid,) = batch_gen.insert(
                            [array_int_list(input_ids.squeeze(0))],
                            max_tokens=args.max_tokens,
                            prompt_kwargs=[gen_kwargs],
                            logits_processors=[
                                self._make_logits_processors(args, input_ids)
                            ],
                            thinking_budget_criteria=[thinking_budget_criteria],
                        )
                    except Exception as e:
                        rqueue.put(e)
                        continue

                    rqueue.put(GenerationContext(uid=uid, prompt_tokens=prompt_tokens))
                    active[uid] = {
                        "rqueue": rqueue,
                        "streamer": ServerTokenStreamer(
                            self.tokenizer,
                            make_streaming_detokenizer(self.processor),
                        ),
                        "gen_kwargs": gen_kwargs if has_embeds else None,
                        "prompt_tps": None,
                        "cached_tokens": 0,
                        **log_state,
                    }

                if not active or batch_gen is None:
                    continue

                self._step(batch_gen, active)

            except Exception as e:
                logger.exception("Error in generation thread")
                error_queues = {
                    id(info["rqueue"]): info["rqueue"] for info in active.values()
                }
                error_queues.update(
                    {id(request.rqueue): request.rqueue for request in new_items}
                )
                _notify_queues(error_queues.values(), e, None)
                active.clear()
                batch_gen = None
                mx.clear_cache()
                gc.collect()

        if batch_gen is not None and callable(getattr(batch_gen, "close", None)):
            batch_gen.close()

    def _step(
        self,
        batch_gen: BatchGenerator,
        active: dict[int, ActiveGeneration],
        gen_kwargs: PromptInputs | None = None,
    ):
        """One batch generation step: prefill + decode."""
        kwargs = gen_kwargs or {}
        prompt_responses, responses = batch_gen.next(**kwargs)
        self._log_prefill_progress(batch_gen, active)
        for prompt_response in prompt_responses:
            if prompt_response.uid in active:
                info = active[prompt_response.uid]
                info["prompt_tps"] = prompt_response.prompt_tps
                info["cached_tokens"] = getattr(prompt_response, "cached_tokens", 0)
                self._log_prefill_completed(prompt_response.uid, info, prompt_response)
        if not responses:
            return

        for r in responses:
            if r.uid not in active:
                continue

            info = active[r.uid]
            rqueue = info["rqueue"]

            tok = r.token
            token_count = 0 if tok is None else 1
            if tok is None:
                text = info["streamer"].finalize()
                tok = 0
            else:
                text = self._stream_text(info, tok, r.finish_reason)

            lp = r.token_logprob

            emitted_at = self._log_decode_progress(
                r.uid,
                info,
                token=tok,
                text=text,
                finish_reason=r.finish_reason,
                token_count=token_count,
            )

            rqueue.put(
                StreamingToken(
                    text=text,
                    token=tok,
                    logprobs=lp,
                    finish_reason=r.finish_reason,
                    peak_memory=mx.get_peak_memory() / 1e9 if r.finish_reason else 0,
                    prompt_tps=info.get("prompt_tps"),
                    top_logprobs=getattr(r, "top_logprobs", None),
                    cached_tokens=info.get("cached_tokens", 0),
                    token_count=token_count,
                    emitted_at=emitted_at,
                )
            )

            if r.finish_reason is not None:
                rqueue.put(None)
                del active[r.uid]

    def _stream_text(
        self, info: ActiveGeneration, token: int, finish_reason: Optional[str]
    ) -> str:
        """Convert one generated token into a streaming text segment."""
        return info["streamer"].advance(token, finish_reason)

    def _flush(self, batch_gen: BatchGenerator, active: dict[int, ActiveGeneration]):
        """Drain all pending text-only prompts before inserting an image request."""
        while batch_gen.has_pending_prompts:
            self._step(batch_gen, active)

    def validate_context_budget(
        self,
        prompt: str,
        images: Optional[list[ImageSource]] = None,
        args: Optional[GenerationArguments] = None,
    ):
        """Validate request size before opening a streaming response."""
        if get_configured_context_limit() is None:
            return
        self.wait_until_ready()
        args = args or GenerationArguments(max_tokens=get_server_max_tokens())
        raw_inputs = self.cpu_preprocess(prompt, images)
        _check_configured_context_budget(
            count_prompt_tokens(raw_inputs), args.max_tokens
        )
