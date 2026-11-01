"""Gemma3 text decoder with image-aware masks and global/sliding-window caches."""

from collections.abc import Mapping, Sequence
from functools import partial
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.contracts import AttentionCache, ModelArgument
from mlx_vlm.models.base import (
    LanguageModelOutput,
    create_attention_mask,
    kv_sequence_length,
    scaled_dot_product_attention,
)
from mlx_vlm.models.cache import BatchRotatingKVCache, KVCache, RotatingKVCache
from mlx_vlm.models.gemma3.attention import cached_attention_mask, prompt_position
from mlx_vlm.models.gemma3.config import TextConfig
from mlx_vlm.models.rope_utils import initialize_rope


class RMSNorm(nn.Module):
    """Gemma RMS norm: normalized input is scaled by ``1 + weight``.

    The weight starts at one until checkpoint values are loaded.
    """

    def __init__(self, dims: int, eps: float = 1e-5):
        """Create ones-weighted normalization over ``dims`` channels."""
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array):
        """Normalize the last axis and scale by ``1 + weight``."""
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class Attention(nn.Module):
    """Grouped-query attention for one Gemma3 layer.

    Layers are sliding-window except every ``sliding_window_pattern``-th layer,
    which is global. Sliding layers use the local RoPE base and no RoPE scaling;
    global layers use the global base and ``rope_scaling``.
    """

    def __init__(self, config: TextConfig, layer_idx: int):
        """Create projections, head norms and RoPE for layer ``layer_idx``.

        Layer ``layer_idx`` is sliding-window unless it is every
        ``sliding_window_pattern``-th layer; sliding layers use the local RoPE
        base without scaling, global layers the global base with
        ``rope_scaling``. Projections have no biases.
        """
        super().__init__()

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.repeats = n_heads // n_kv_heads
        self.head_dim = head_dim = config.head_dim
        self.layer_idx = layer_idx

        self.scale = config.query_pre_attn_scalar**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = RMSNorm(dims=head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(dims=head_dim, eps=config.rms_norm_eps)
        self.is_sliding = (layer_idx + 1) % config.sliding_window_pattern != 0
        self.window_size = config.sliding_window if self.is_sliding else None

        self.rope = initialize_rope(
            dims=head_dim,
            traditional=config.rope_traditional,
            base=(
                config.rope_local_base_freq
                if self.is_sliding
                else config.rope_global_base_freq
            ),
            scaling_config=None if self.is_sliding else config.rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: AttentionCache | None = None,
        prompt_mask: mx.array | None = None,
    ) -> mx.array:
        """Attend over ``x`` of shape ``(batch, length, hidden)``.

        Args:
            x: Normalized hidden states.
            mask: Mask or MLX ``"causal"`` sentinel, used only when
                ``prompt_mask`` is ``None``. An array is trimmed to the last
                ``key_len`` columns.
            cache: Layer KV cache. RoPE positions start at ``cache.offset``.
            prompt_mask: Boolean ``(batch, 1, prompt, prompt)`` image/padding
                mask. When given it replaces ``mask``: the slice for this step
                is built with ``cached_attention_mask`` and this layer's window.

        Raises:
            ValueError: If ``prompt_mask`` is combined with a rotating cache that
                has ``keep > 0``, because the mask assumes a pure sliding window.
        """
        if (
            prompt_mask is not None
            and isinstance(cache, RotatingKVCache)
            and cache.keep
        ):
            raise ValueError(
                "Gemma image attention requires a rotating cache with keep=0"
            )
        B, L, _ = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = mx.transpose(
            mx.reshape(queries, (B, L, self.n_heads, -1)), (0, 2, 1, 3)
        )

        keys = mx.transpose(mx.reshape(keys, (B, L, self.n_kv_heads, -1)), (0, 2, 1, 3))
        values = mx.transpose(
            mx.reshape(values, (B, L, self.n_kv_heads, -1)), (0, 2, 1, 3)
        )

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        # RoPE excludes each row's left padding; the prompt mask uses shared
        # padded columns. A batched cache therefore supplies two coordinates.
        offset = prompt_position(cache)
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        if prompt_mask is not None:
            # A full rotating cache returns keys in ring-buffer order during
            # single-token decode, so roll the mask columns to match.
            rotation = (
                cache.position
                if isinstance(cache, (RotatingKVCache, BatchRotatingKVCache))
                and L == 1
                and kv_sequence_length(keys) == cache.max_size
                else 0
            )
            mask = cached_attention_mask(
                prompt_mask,
                offset,
                L,
                kv_sequence_length(keys),
                self.window_size,
                rotation,
            )
        elif isinstance(mask, mx.array):
            key_len = kv_sequence_length(keys)
            if mask.shape[-1] != key_len:
                mask = mask[..., -key_len:]

        output = scaled_dot_product_attention(
            queries, keys, values, cache, scale=self.scale, mask=mask
        )
        output = mx.reshape(mx.transpose(output, (0, 2, 1, 3)), (B, L, -1))
        return self.o_proj(output)


class MLP(nn.Module):
    """Gemma3 gated feed-forward block."""

    def __init__(self, dim: int, hidden_dim: int):
        """Create the gate, up and down projections without biases."""
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply the gated tanh-approximate GELU feed-forward."""
        # Gemma uses the tanh-approximate GELU (``gelu_pytorch_tanh`` in
        # Transformers, the default of ``jax.nn.gelu``), not the exact erf form.
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


@partial(mx.compile, shapeless=True)
def clip_residual(x: mx.array, y: mx.array | None = None):
    """Return ``x`` or ``x + y``, saturated to the float16 range for float16 input.

    Float16 overflows to infinity above 65504, which large activations can
    exceed. For float16 the sum is formed in float32 and clipped to the finite
    range before casting back. Other dtypes pass through unchanged.
    """
    bound = mx.finfo(mx.float16).max
    if y is None:
        if x.dtype == mx.float16:
            return mx.clip(x.astype(mx.float32), -bound, bound).astype(mx.float16)
        else:
            return x

    if x.dtype != mx.float16:
        return x + y

    return mx.clip(x.astype(mx.float32) + y.astype(mx.float32), -bound, bound).astype(
        mx.float16
    )


class TransformerBlock(nn.Module):
    """One Gemma3 decoder layer: attention, MLP and four RMS norms."""

    def __init__(self, config: TextConfig, layer_idx: int):
        """Build attention, MLP and normalization layers for ``layer_idx``."""
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | RotatingKVCache | AttentionCache | None = None,
        prompt_mask: mx.array | None = None,
    ) -> mx.array:
        """Apply the attention and MLP residual blocks to ``x``.

        ``x`` has shape ``(batch, length, hidden)``. Inputs and residual sums
        are saturated to the finite float16 range before each addition.
        """
        # Clip inputs and residual sums to the finite float16 range. The sums
        # below are formed before the one-argument clip, in their input dtype.
        x = clip_residual(x)

        # Self-attention block
        r = self.self_attn(self.input_layernorm(x), mask, cache, prompt_mask)
        h = self.post_attention_layernorm(r)

        # Add residual connection with overflow protection for float16
        h = clip_residual(x + h)

        # MLP block
        r = self.mlp(self.pre_feedforward_layernorm(h))
        out = self.post_feedforward_layernorm(r)

        # Add residual connection with overflow protection for float16
        out = clip_residual(h + out)

        return out


class Gemma3Model(nn.Module):
    """Gemma3 decoder stack: embedding, transformer blocks and final norm.

    ``scale_inputs_embeds`` controls whether caller-supplied ``inputs_embeds``
    are multiplied by ``sqrt(hidden_size)`` (rounded through bfloat16). Token
    embeddings looked up from ``inputs`` are always scaled.
    """

    def __init__(self, config: TextConfig, scale_inputs_embeds: bool = True):
        """Create token embeddings, ``num_hidden_layers`` blocks and the final norm."""
        super().__init__()
        self.config = config
        self.scale_inputs_embeds = scale_inputs_embeds
        self.vocab_size = config.vocab_size
        self.window_size = config.sliding_window
        self.sliding_window_pattern = config.sliding_window_pattern
        self.num_hidden_layers = config.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TransformerBlock(config=config, layer_idx=layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: mx.array | None = None,
        mask: mx.array | None = None,
        cache: Sequence[AttentionCache | None] | None = None,
        attention_mask_4d: mx.array | None = None,
    ):
        """Run the decoder and return normalized hidden states.

        With neither mask supplied, per-layer causal (global) or sliding-window
        masks are derived from the caches.

        Args:
            inputs: Token IDs, ``(batch, length)``; used only for the embedding
                lookup when ``inputs_embeds`` is ``None``.
            inputs_embeds: Optional embeddings, ``(batch, length, hidden)``;
                scaled here only when ``scale_inputs_embeds`` is enabled.
            mask: Explicit attention mask shared by all layers.
            cache: One cache per layer; ``None`` entries mean no cache. The
                sequence must match the layer count.
            attention_mask_4d: Boolean ``(batch, 1, prompt, prompt)`` prompt
                mask from ``image_attention_mask``; each layer slices it for the
                current cache window.

        Returns:
            Hidden states of shape ``(batch, length, hidden)``.

        Raises:
            ValueError: If the cache count differs from the layer count, or both
                ``mask`` and ``attention_mask_4d`` are given.
                When ``inputs_embeds`` is supplied, also raised if the prompt
                mask does not cover the cached prompt or this step's keys end
                inside a bidirectional image block.
        """
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError("Gemma cache count must match the layer count")
        embedded_tokens = inputs_embeds is None
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        if embedded_tokens or self.scale_inputs_embeds:
            h = h * mx.array(self.config.hidden_size**0.5, mx.bfloat16).astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        if attention_mask_4d is not None and inputs_embeds is not None:
            # A True entry right of this step's last key means a query here would
            # need a later key of its image block, which does not exist yet:
            # the step boundary splits that block.
            offset = prompt_position(cache[0] if cache else None)
            end = offset + h.shape[1]
            if end > attention_mask_4d.shape[-1]:
                raise ValueError(
                    "Gemma prefill mask must cover the complete cached prompt"
                )
            if bool(mx.any(attention_mask_4d[:, :, offset:end, end:]).item()):
                raise ValueError(
                    "Gemma prefill boundary splits a bidirectional image block"
                )
        if mask is not None and attention_mask_4d is not None:
            raise ValueError("Provide either mask or attention_mask_4d, not both")
        global_mask = sliding_window_mask = None
        if mask is None and attention_mask_4d is None:
            global_cache = next(
                (
                    c
                    for layer, c in zip(self.layers, cache)
                    if not layer.self_attn.is_sliding
                ),
                None,
            )
            local_cache = next(
                (
                    c
                    for layer, c in zip(self.layers, cache)
                    if layer.self_attn.is_sliding
                ),
                None,
            )
            global_mask = create_attention_mask(h, global_cache)
            sliding_window_mask = create_attention_mask(
                h, local_cache, window_size=self.window_size
            )

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            is_global = (
                i % self.sliding_window_pattern == self.sliding_window_pattern - 1
            )

            local_mask = mask
            if mask is None and attention_mask_4d is None and is_global:
                local_mask = global_mask
            elif mask is None and attention_mask_4d is None:
                local_mask = sliding_window_mask

            h = layer(h, local_mask, c, attention_mask_4d)

        return self.norm(h)


class LanguageModel(nn.Module):
    """Gemma3 decoder stack plus its vocabulary projection."""

    def __init__(self, config: TextConfig):
        """Create the decoder and a bias-free output projection."""
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Gemma3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def chunked_prefill_policy(
        self,
        *,
        prefill_kwargs: Mapping[str, ModelArgument] | None = None,
        **kwargs: object,
    ) -> bool:
        """Keep prepared bidirectional image masks intact in decoder-only batches.

        The HTTP batcher receives this decoder without the outer vision model's
        policy. Future-visible entries identify image blocks in its prepared mask.
        Text-only causal masks can still use the requested prefill chunk size.
        """
        mask = (prefill_kwargs or {}).get("attention_mask_4d")
        if mask is not None and not isinstance(mask, mx.array):
            raise TypeError("Gemma prompt attention must be an MLX array")
        has_image_blocks = isinstance(mask, mx.array) and bool(
            mx.any(mx.triu(mask, k=1)).item()
        )
        return not has_image_blocks

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: mx.array | None = None,
        cache: Sequence[AttentionCache | None] | None = None,
        attention_mask_4d: mx.array | None = None,
        **kwargs: ModelArgument,
    ):
        """Run the decoder and project hidden states to vocabulary logits."""
        out = self.model(
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            attention_mask_4d=attention_mask_4d,
        )
        out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Supply an omitted tied head without replacing a saved output head.

        The loader passes the full multimodal checkpoint namespace here.
        Checking an unprefixed key would overwrite an existing native head
        with the input embedding on every load.
        """
        sanitized = {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }
        head_key = "language_model.lm_head.weight"
        if head_key not in sanitized:
            sanitized[head_key] = sanitized["language_model.model.embed_tokens.weight"]
        return sanitized

    @property
    def layers(self):
        """Return the decoder blocks in execution order."""
        return self.model.layers

    @property
    def head_dim(self):
        """Return the per-head width from the text config."""
        return self.config.head_dim

    @property
    def n_kv_heads(self):
        """Return the number of key/value heads from the text config."""
        return self.config.num_key_value_heads

    def make_cache(self):
        """Return per-layer caches: ``KVCache`` (global) or ``RotatingKVCache``.

        Sliding layers get a rotating cache of ``sliding_window`` entries with
        ``keep=0``, which the image attention mask requires because it assumes a
        pure sliding window.
        """
        caches: list[KVCache | RotatingKVCache] = []
        for i in range(self.config.num_hidden_layers):
            if (
                i % self.config.sliding_window_pattern
                == self.config.sliding_window_pattern - 1
            ):
                caches.append(KVCache())
            else:
                caches.append(
                    RotatingKVCache(
                        max_size=self.config.sliding_window,
                        keep=0,
                    )
                )
        return caches
