"""Model configuration, embedding and cache-aware attention interfaces."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Optional,
    Protocol,
    Self,
    TypeIs,
    runtime_checkable,
)

import mlx.core as mx
import numpy as np

from mlx_vlm.contracts import (
    CacheComponent,
    JSONValue,
    KVBuffer,
    ProcessedInputs,
    QuantizedTensor,
    is_json_object,
    is_json_value,
    numpy_to_mlx,
)
from mlx_vlm.models.cache import (
    BatchQuantizedKVCache,
    QuantizedKVCache,
    create_causal_mask,
)


class ChatTemplated(Protocol):
    """Tokenizer exposing an optional chat template."""

    chat_template: str | dict[str, str] | None


def load_chat_template[T: ChatTemplated](tokenizer: T, model_path: str | Path) -> T:
    """Apply a chat template from the model directory to *tokenizer*."""
    import json
    from pathlib import Path

    model_dir = Path(model_path)
    chat_template_json = model_dir / "chat_template.json"
    chat_template_jinja = model_dir / "chat_template.jinja"

    if chat_template_json.exists():
        template_data: object = json.loads(chat_template_json.read_text())
        if not is_json_object(template_data) or not isinstance(
            template_data.get("chat_template"), str
        ):
            raise ValueError("chat_template.json requires a string chat_template")
        template = template_data["chat_template"]
        if not isinstance(template, str):
            raise ValueError("chat_template must be a string")
        tokenizer.chat_template = template
    elif chat_template_jinja.exists():
        tokenizer.chat_template = chat_template_jinja.read_text()

    return tokenizer


def to_mlx(data: ProcessedInputs) -> ProcessedInputs:
    """Convert all array-like values in a processor output dict to mx.array."""
    result: ProcessedInputs = {}
    for key, value in data.items():
        if value is None or isinstance(value, mx.array):
            result[key] = value
        elif isinstance(value, np.ndarray):
            result[key] = numpy_to_mlx(value)
        elif isinstance(value, list):
            try:
                result[key] = numpy_to_mlx(np.array(value))
            except (ValueError, TypeError):
                result[key] = value
        else:
            result[key] = value
    return result


@dataclass
class LanguageModelOutput:
    """Logits of shape ``(batch, tokens, vocab)`` from a decoder forward pass."""

    logits: mx.array


@dataclass
class InputEmbeddingsFeatures:
    """Prompt embeddings plus the optional four-dimensional attention mask.

    ``attention_mask_4d`` carries explicit image-block visibility, for example
    a bidirectional mask over an image's patch tokens, and is ``None`` when the
    language model should build its own causal mask.
    """

    inputs_embeds: mx.array
    attention_mask_4d: Optional[mx.array] = None

    def to_dict(self) -> dict[str, mx.array | None]:
        """Return both fields, preserving ``None`` for the mask."""
        return {
            "inputs_embeds": self.inputs_embeds,
            "attention_mask_4d": self.attention_mask_4d,
        }


type ConfigValue = JSONValue | BaseModelConfig


@dataclass
class BaseModelConfig:
    """Dataclass base for checkpoint configs with JSON-safe dictionary output."""

    @classmethod
    def from_dict(cls, params: Mapping[str, object]) -> Self:
        """Build a config from the keys its dataclass fields accept."""
        if not params:
            return cls()
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

    def to_dict(self) -> dict[str, ConfigValue]:
        """Expose populated config fields, retaining nested config instances."""
        result: dict[str, ConfigValue] = {}
        for key, raw in vars(self).items():
            value: object = raw
            if value is None:
                continue
            if not isinstance(value, BaseModelConfig) and not is_json_value(value):
                raise TypeError(f"Unsupported model configuration field: {key}")
            result[key] = value
        return result


def _is_quantized_tensor(value: KVBuffer) -> TypeIs[QuantizedTensor]:
    """Distinguish uniform quantization tuples from dense arrays."""
    return isinstance(value, tuple)


def kv_sequence_length(keys: KVBuffer) -> int:
    """Read sequence length from dense or uniform-quantized keys."""
    if _is_quantized_tensor(keys):
        return keys[0].shape[-2]
    return keys.shape[-2]


@runtime_checkable
class MaskCache(Protocol):
    """Cache that builds the attention mask for its stored layout."""

    def make_mask(
        self, N: int, *, return_array: bool = False, window_size: int | None = None
    ) -> mx.array | str | None:
        """Return a mask array, ``"causal"``, or ``None`` for ``N`` new tokens."""
        ...


def create_attention_mask(
    h: mx.array,
    cache: CacheComponent | mx.array | None = None,
    window_size: Optional[int] = None,
    return_array: bool = False,
) -> mx.array | str | None:
    """Build the mask for ``h.shape[1]`` query tokens.

    A cache providing ``make_mask`` decides the layout; otherwise single-token
    steps return ``None``, and wider steps return ``"causal"`` unless an array
    is requested or a sliding window applies.
    """
    N = h.shape[1]
    if (
        cache is not None
        and not isinstance(cache, mx.array)
        and isinstance(cache, MaskCache)
    ):
        return cache.make_mask(N, return_array=return_array, window_size=window_size)
    if N == 1:
        return None
    if return_array or (window_size and N > window_size):
        return create_causal_mask(N, window_size=window_size)
    return "causal"


def align_attention_mask_to_scores(
    mask: mx.array | str | None, scores: mx.array
) -> mx.array | str | None:
    """Broadcast an attention mask onto *scores* without mis-aligning batch vs heads.

    Quantized GQA expands scores to 5D ``(B, n_kv_heads, n_repeats, L, K)`` while
    batch left-pad masks are typically 4D ``(B, 1, L, K)``. Right-aligning those
    shapes aliases ``B`` with ``n_kv_heads`` and crashes for multi-row batches
    (upstream MLX-VLM issue #1567). Insert singleton dims *before* the last two
    axes until ranks match so the layout is ``(B, 1, ..., 1, L, K)``.
    """
    if mask is None or isinstance(mask, str):
        return mask

    # Grow rank by inserting size-1 axes immediately before (L, K).
    while mask.ndim < scores.ndim:
        insert_at = max(mask.ndim - 2, 0)
        mask = mx.expand_dims(mask, axis=insert_at)
    return mask


def quantized_scaled_dot_product_attention(
    queries: mx.array,
    q_keys: tuple[mx.array, mx.array, mx.array],
    q_values: tuple[mx.array, mx.array, mx.array],
    scale: float,
    mask: mx.array | str | None,
    group_size: int = 64,
    bits: int = 8,
) -> mx.array:
    """Apply attention directly against uniformly quantized K/V tuples.

    ``queries`` is ``(batch, query_heads, L, head_dim)`` and each packed tuple
    holds ``(values, scales, biases)`` with ``n_kv_heads`` at axis ``-3``.
    Grouped-query heads are expanded to ``(batch, n_kv_heads, repeats, L, D)``
    for the quantized matmuls; the result is flattened back to query heads.
    """
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    queries *= scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = (
            mx.expand_dims(q_keys[0], axis=-3),
            mx.expand_dims(q_keys[1], axis=-3),
            mx.expand_dims(q_keys[2], axis=-3),
        )
        q_values = (
            mx.expand_dims(q_values[0], axis=-3),
            mx.expand_dims(q_values[1], axis=-3),
            mx.expand_dims(q_values[2], axis=-3),
        )

    scores = mx.quantized_matmul(
        queries, *q_keys, transpose=True, group_size=group_size, bits=bits
    )
    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
        aligned_mask = align_attention_mask_to_scores(mask, scores)
        assert isinstance(aligned_mask, mx.array)
        mask = aligned_mask
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores += mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(
        scores, *q_values, transpose=False, group_size=group_size, bits=bits
    )

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))

    return out


def scaled_dot_product_attention(
    queries: mx.array,
    keys: KVBuffer,
    values: KVBuffer,
    cache: CacheComponent | None,
    scale: float,
    mask: mx.array | str | None,
    sinks: Optional[mx.array] = None,
) -> mx.array:
    """Apply cache-aware attention with an array mask or MLX causal sentinel."""
    if isinstance(cache, (QuantizedKVCache, BatchQuantizedKVCache)):
        if not _is_quantized_tensor(keys) or not _is_quantized_tensor(values):
            raise TypeError("Uniform-quantized cache requires packed K/V tuples")
        if sinks is not None:
            raise ValueError("Quantized SDPA does not support attention sinks.")
        return quantized_scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            group_size=cache.group_size,
            bits=cache.bits,
        )

    if not isinstance(keys, mx.array) or not isinstance(values, mx.array):
        raise TypeError("Dense attention requires array K/V state")
    return mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=scale,
        mask=mask,
        sinks=sinks,
    )


def check_array_shape(arr: mx.array) -> bool:
    """Report whether a convolution weight already uses MLX's channels-last layout.

    For four-dimensional weights the layout is accepted when the output
    channels are at least as wide as the square kernel; three-dimensional
    weights are accepted when the second axis is at least the third.
    """
    shape = arr.shape

    # Check if the shape has 4 dimensions
    if len(shape) == 4:
        out_channels, kH, KW, _ = shape
        # Check if out_channels is the largest, and kH and KW are the same
        if (out_channels >= kH) and (out_channels >= KW) and (kH == KW):
            return True
        else:
            return False
    # Check if the shape has 3 dimensions
    elif len(shape) == 3:
        _, kW, out_channels = shape
        # Check if out_channels is the largest
        if kW >= out_channels:
            return True
        else:
            return False
    else:
        return False
