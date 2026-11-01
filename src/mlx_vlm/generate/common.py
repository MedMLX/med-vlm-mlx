"""Generation defaults, result records and shared stream helpers."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import List, Optional, Protocol, cast, runtime_checkable

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce

from mlx_vlm.contracts import CacheComponent, CacheEntry, ModelArgument
from mlx_vlm.kv_quant import kv_quant_config
from mlx_vlm.models.cache import RotatingKVCache
from mlx_vlm.models.protocols import GenerationModelLike, LanguageModelLike

DEFAULT_KV_GROUP_SIZE = 64
DEFAULT_QUANTIZED_KV_START = 5000

logger = logging.getLogger("mlx_vlm.generate")

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 0
DEFAULT_MIN_P = 0.0
DEFAULT_REPETITION_CONTEXT_SIZE = 20
DEFAULT_PREFILL_STEP_SIZE = 2048
DEFAULT_COMPLETION_BATCH_SIZE = 32
DEFAULT_PREFILL_BATCH_SIZE = 8

# A stream on the default device just for generation
generation_stream = mx.new_thread_local_stream(mx.default_device())


class _ChunkedPrefillPolicy(Protocol):
    def __call__(
        self,
        *,
        input_ids: mx.array | None,
        inputs_embeds: mx.array | None,
        prompt_cache: Sequence[CacheComponent] | None,
        prefill_kwargs: Mapping[str, ModelArgument],
    ) -> object: ...


@runtime_checkable
class _QuantizableCache(Protocol):
    offset: int

    def to_quantized(self, *, group_size: int, bits: int) -> CacheEntry: ...


def _policy_enabled(policy: object) -> bool:
    return bool(getattr(policy, "enabled", policy))


def default_prefill_step_size_for_offload(
    model: object,
    prefill_step_size: Optional[int],
    default: int,
) -> Optional[int]:
    """A one-shot lazy prefill pins every touched expert in a single graph
    until the final eval, defeating ``ExpertStore``'s LRU budget regardless
    of ``prefill_step_size``'s own chunked-prefill benefits elsewhere -- so
    an explicit ``None`` (chunking off) against an expert-offload model is a
    footgun, not a valid choice."""
    if (
        prefill_step_size is None
        and getattr(model, "moe_offload_store", None) is not None
    ):
        logger.warning(
            "prefill_step_size=None with an expert-offload model: falling "
            "back to prefill_step_size=%d instead of pinning every touched "
            "expert resident until the final eval.",
            default,
        )
        return default
    return prefill_step_size


def chunked_prefill_enabled(
    model: object,
    *,
    input_ids: mx.array | None = None,
    inputs_embeds: mx.array | None = None,
    prompt_cache: Sequence[CacheComponent] | None = None,
    prefill_kwargs: Mapping[str, ModelArgument] | None = None,
) -> bool:
    """Return whether prompt prefill may be split into ``prefill_step_size`` chunks.

    The first object with a callable ``chunked_prefill_policy`` (the model, then
    its language model) decides. Gemma3 uses this to keep prompts that contain
    image blocks whole, because a chunk boundary inside a bidirectional image
    block leaves its earlier queries without the later keys of that block.
    Otherwise a truthy ``no_chunked_prefill`` attribute disables chunking, and
    the default is to chunk.
    """
    prefill_kwargs = prefill_kwargs or {}
    candidates = [model]
    language_model = getattr(model, "language_model", None)
    if language_model is not None and language_model is not model:
        candidates.append(language_model)

    for candidate in candidates:
        policy: object = getattr(candidate, "chunked_prefill_policy", None)
        if callable(policy):
            return _policy_enabled(
                cast(_ChunkedPrefillPolicy, policy)(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    prompt_cache=prompt_cache,
                    prefill_kwargs=prefill_kwargs,
                )
            )

    if any(getattr(candidate, "no_chunked_prefill", False) for candidate in candidates):
        return False

    return True


def maybe_quantize_kv_cache(
    prompt_cache: list[CacheEntry],
    quantized_kv_start: int,
    kv_group_size: int,
    kv_bits: float | None,
):
    """Quantize eligible cache entries in place for single-stream generation.

    ``kv_bits=None`` leaves the cache unchanged. Every layer exposing
    ``to_quantized`` converts to uniform quantization once its offset reaches
    ``quantized_kv_start``; fractional ``kv_bits`` raise. Gemma3 sliding-window
    layers keep their rotating caches unquantized, as in batch generation and
    upstream MLX-VLM's other KV schemes, because rotating caches have no
    quantized layout.
    """
    config = kv_quant_config(kv_bits, kv_group_size)
    if config is None:
        return

    for index, layer_cache in enumerate(prompt_cache):
        if isinstance(layer_cache, RotatingKVCache):
            continue
        if hasattr(layer_cache, "to_quantized"):
            if not isinstance(layer_cache, _QuantizableCache):
                raise TypeError("A quantizable cache must expose an integer offset")
            if layer_cache.offset >= quantized_kv_start:
                prompt_cache[index] = layer_cache.to_quantized(
                    group_size=config["group_size"],
                    bits=config["bits"],
                )


@contextlib.contextmanager
def wired_limit(
    model: nn.Module | GenerationModelLike | LanguageModelLike,
    streams: Optional[List[mx.Stream | mx.ThreadLocalStream]] = None,
):
    """Temporarily set the wired memory limit for generation."""
    if not mx.metal.is_available():
        yield
        return

    def parameter_bytes(total: int, leaf: object) -> int:
        return total + leaf.nbytes if isinstance(leaf, mx.array) else total

    model_bytes = tree_reduce(parameter_bytes, model, 0)
    assert isinstance(model_bytes, int)
    max_rec_size = mx.device_info()["max_recommended_working_set_size"]
    assert isinstance(max_rec_size, int)  # Native MLX reports this byte count as int.
    if model_bytes > 0.9 * max_rec_size:
        model_mb = model_bytes // 2**20
        max_rec_mb = max_rec_size // 2**20
        print(
            f"[WARNING] Generating with a model that requires {model_mb} MB "
            f"which is close to the maximum recommended size of {max_rec_mb} "
            "MB. This can be slow. See the documentation for possible work-arounds: "
            "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
        )
    old_limit = mx.set_wired_limit(max_rec_size)
    try:
        yield
    finally:
        if streams is not None:
            for stream in streams:
                mx.synchronize(stream)
        else:
            mx.synchronize()
        mx.set_wired_limit(old_limit)


@dataclass
class GenerationResult:
    """Text and measurements from a streaming event or completed generation.

    In ``stream_generate``, ``text`` is the latest decoded segment and token
    counts are cumulative; ``generate`` concatenates those segments. The final
    stream event carries ``finish_reason`` and generated ``token_ids`` when
    available. Prompt and generation rates are tokens per second; ``peak_memory``
    is MLX's reported peak in decimal GB, not an isolated per-request allocation.
    ``cached_tokens`` counts the reused prompt prefix.
    """

    text: str = ""
    token: Optional[int] = None
    logprobs: mx.array | List[float] | None = None
    prompt_tokens: int = 0
    generation_tokens: int = 0
    total_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory: float = 0.0
    cached_tokens: int = 0
    finish_reason: Optional[str] = None
    text_already_printed: bool = False
    token_ids: Optional[List[int]] = None


class PromptCacheState:
    """Holds KV cache and token history across conversation turns."""

    def __init__(self):
        """Start without cached tokens or cache entries."""
        self.cache: Optional[List[CacheEntry]] = None
        self.token_ids: Optional[List[int]] = None

    def find_prefix_length(self, new_ids: list[int]) -> int:
        """Return the number of leading tokens that match the cached ids."""
        if self.token_ids is None:
            return 0
        max_len = min(len(self.token_ids), len(new_ids))
        for i in range(max_len):
            if self.token_ids[i] != new_ids[i]:
                return i
        return max_len

    def update(self, token_ids: list[int], kv_cache: list[CacheEntry]):
        """Store the full token sequence and corresponding KV cache."""
        self.token_ids = list(token_ids)
        self.cache = kv_cache
