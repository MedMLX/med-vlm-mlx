"""Attention KV caches for dense, rotating and quantized decode layouts.

Every cache stores keys and values as ``(batch, heads, tokens, head_dim)``
buffers, where key and value widths may differ. ``offset`` counts logical tokens
(per row in batched caches) while allocated storage can be longer; ``state``
exposes only the logical prefix so snapshots stay independent of the allocation
step.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import ceil
from typing import List, Optional, Protocol, Self, cast

import mlx.core as mx

from mlx_vlm.contracts import (
    CacheComponent,
    CacheFactory,
    CacheModel,
    KVBuffer,
    QuantizedTensor,
    StateTree,
    StateView,
    array_depends,
    parse_int,
)

type ArrayPairState = tuple[mx.array | None, mx.array | None]
type PackedPairState = tuple[QuantizedTensor | None, QuantizedTensor | None]


def _map_packed(
    operation: Callable[[mx.array], mx.array], value: QuantizedTensor
) -> QuantizedTensor:
    """Transform a packed triple without losing its three tensor roles."""
    return operation(value[0]), operation(value[1]), operation(value[2])


def _state_items(value: StateTree, count: int | None = None) -> Sequence[StateTree]:
    """Validate the serialized sequence before unpacking native cache state."""
    if not isinstance(value, (list, tuple)):
        raise TypeError("Cache state must be a list or tuple")
    if count is not None and len(value) != count:
        raise ValueError(f"Cache state requires {count} items; received {len(value)}")
    return value


def _array_pair_state(value: StateTree) -> ArrayPairState:
    """Read the optional key/value tensors used by dense caches."""
    keys, values = _state_items(value, 2)
    if keys is not None and not isinstance(keys, mx.array):
        raise TypeError("Cache keys must be an MLX array or None")
    if values is not None and not isinstance(values, mx.array):
        raise TypeError("Cache values must be an MLX array or None")
    if (keys is None) != (values is None):
        raise ValueError("Cache keys and values must both be present or both be None")
    return keys, values


def _packed_tensor_state(value: StateTree) -> QuantizedTensor | None:
    if value is None:
        return None
    packed, scales, biases = _state_items(value, 3)
    if not all(isinstance(part, mx.array) for part in (packed, scales, biases)):
        raise TypeError("Quantized cache state requires packed, scale and bias arrays")
    return cast(QuantizedTensor, (packed, scales, biases))


def _packed_pair_state(value: StateTree) -> PackedPairState:
    keys, values = _state_items(value, 2)
    packed_keys, packed_values = (
        _packed_tensor_state(keys),
        _packed_tensor_state(values),
    )
    if (packed_keys is None) != (packed_values is None):
        raise ValueError("Cache keys and values must both be present or both be None")
    return packed_keys, packed_values


def _metadata_ints(value: StateTree, count: int) -> list[int]:
    return [parse_int(item) for item in _state_items(value, count)]


class _KVMemoryView(Protocol):
    @property
    def keys(self) -> KVBuffer | None: ...

    @property
    def values(self) -> KVBuffer | None: ...


class _WindowedKVMemoryView(_KVMemoryView, Protocol):
    @property
    def max_size(self) -> int: ...


def should_quantize_kv_layer(layer_idx: int, num_layers: int) -> bool:
    """Whether layer ``layer_idx`` should use a quantized KV cache.

    Live batch generation (``_make_cache``) uses this policy. Live stream
    quantization can quantize every eligible layer, so callers cannot assume
    the policies are identical.

    For deep stacks (``num_layers > 2``) the final layer stays unquantized,
    a policy inherited from upstream MLX-VLM because that layer is sensitive
    to quantization. Shallow stacks (``num_layers <= 2``) quantize every layer
    when kv-bits is on.
    """
    if num_layers <= 2:
        return True
    return layer_idx < num_layers - 1


def create_causal_mask(
    N: int,
    offset: int = 0,
    window_size: Optional[int] = None,
    right_padding: Optional[mx.array] = None,
    left_padding: Optional[mx.array] = None,
) -> mx.array:
    """Build a causal mask of shape ``[N, offset + N]``.

    ``window_size`` limits each query to the preceding window, ``right_padding``
    hides trailing cache columns per row and ``left_padding`` hides leading
    padding columns per row. The mask is boolean, with ``True`` marking allowed
    key positions.
    """
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    if right_padding is not None:
        mask = mask & (rinds < mx.expand_dims((offset + N) - right_padding, (1, 2, 3)))
    if left_padding is not None:
        mask = mask & (mx.expand_dims(left_padding, (1, 2, 3)) <= rinds)
    return mask


def make_prompt_cache(
    model: CacheModel,
    max_kv_size: Optional[int] = None,
) -> list[CacheComponent]:
    """
    Construct the model's cache for use in generation.

    This function will defer the cache construction to the model if it has a
    ``make_cache`` method, otherwise it will make a default KV cache.
    Native MLX model factories supply generation cache entries with state;
    their dynamic module lookup does not expose this return type in MLX stubs.

    Args:
        model (nn.Module): The language model.
        max_kv_size (Optional[int]): If provided and the model does not have a
            ``make_cache`` method, a ``RotatingKVCache`` is used with a maximum
            size of ``max_kv_size``
    """
    if isinstance(model, CacheFactory):
        return model.make_cache()

    num_layers = len(model.layers)
    if max_kv_size is not None:
        return [
            RotatingKVCache(max_size=max_kv_size, keep=4) for _ in range(num_layers)
        ]
    else:
        return [KVCache() for _ in range(num_layers)]


def create_attention_mask(
    N: int, offset: int, return_array: bool, window_size: Optional[int]
) -> mx.array | str | None:
    """Choose the mask form a cache forwards to an attention layer.

    Returns ``None`` for single-token decoding without a window, ``"causal"``
    when the attention layer may build its own causal mask, and an explicit
    array when a window or an array result is required.
    """
    if window_size is not None:
        return create_causal_mask(N, offset, window_size=window_size)
    elif N == 1:
        return None
    elif return_array:
        return create_causal_mask(N, offset, window_size=window_size)
    else:
        return "causal"


@dataclass(frozen=True)
class CacheMemory:
    """One row's allocation policy, measured without copying cache tensors."""

    source_bytes: int = 0
    fixed_bytes: int = 0
    bytes_per_token: float = 0
    step: int = 1
    window_size: Optional[int] = None
    fallback: bool = False
    min_capacity: int = 0

    def footprint(self, tokens: int, chunk_size: Optional[int] = None) -> int:
        """Return allocated bytes for ``tokens``, rounded up to ``step``.

        A sliding window caps capacity at the window plus one chunk, and
        ``min_capacity`` forces a floor for buffers kept from a source cache.
        """
        if tokens <= 0:
            return 0
        if self.window_size is not None and chunk_size is not None:
            tokens = min(tokens, self.window_size - 1 + chunk_size)
        tokens = max(tokens, self.min_capacity)
        step = max(1, self.step)
        capacity = ((tokens + step - 1) // step) * step
        return self.fixed_bytes + ceil(capacity * self.bytes_per_token)


def cache_nbytes(value: object, seen: Optional[set[int]] = None) -> int:
    """Account cache buffers without evaluating or cloning their contents."""
    if value is None:
        return 0
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, dict):
        return sum(
            cache_nbytes(v, seen) for v in cast(dict[object, object], value).values()
        )
    if isinstance(value, (list, tuple)):
        return sum(cache_nbytes(v, seen) for v in cast(Sequence[object], value))
    try:
        size: object = getattr(value, "nbytes", None)
        if isinstance(size, int):
            return size
    except (AttributeError, NotImplementedError):
        pass
    return cache_nbytes(getattr(value, "state", None), seen) + cache_nbytes(
        getattr(value, "meta_state", None), seen
    )


def _kv_memory_profile(c: _KVMemoryView, token_count: int) -> CacheMemory:
    if c.keys is None:
        return CacheMemory()
    keys = c.keys[0] if isinstance(c.keys, tuple) else c.keys
    capacity = keys.shape[2]
    size = cache_nbytes(c.keys) + cache_nbytes(c.values)
    return CacheMemory(
        source_bytes=size,
        bytes_per_token=size / capacity if capacity else 0,
        step=parse_int(getattr(c, "step", 1)),
    )


def _windowed_memory_profile(c: _WindowedKVMemoryView, token_count: int) -> CacheMemory:
    return replace(_kv_memory_profile(c, token_count), window_size=c.max_size)


class BaseCache:
    """Common cache contract: state round trip, trimming and memory accounting."""

    def memory_profile(self, token_count: int) -> CacheMemory | None:
        """Describe standard KV buffers; other layouts override this method."""
        if not hasattr(self, "keys") or not hasattr(self, "values"):
            return None
        # Native subclasses with these attributes own dense or packed KV buffers.
        profile = _kv_memory_profile(cast(_KVMemoryView, self), token_count)
        try:
            if self.nbytes > profile.source_bytes:
                return None  # Auxiliary state needs its own profile.
        except (AttributeError, NotImplementedError):
            pass
        return profile

    @property
    def state(self) -> StateView:
        """Return the empty state of a cache that stores nothing."""
        return []

    @state.setter
    def state(self, v: StateTree, /) -> None:
        """Accept only an empty state."""
        if v is not None and v:
            raise ValueError("This cache has no state but a state was set.")

    @property
    def meta_state(self) -> StateView:
        """Return the empty metadata of a cache without configuration."""
        return ""

    @meta_state.setter
    def meta_state(self, v: StateTree, /) -> None:
        """Accept only empty metadata."""
        if v is not None and v:
            raise ValueError("This cache has no meta_state but a meta_state was set.")

    def is_trimmable(self) -> bool:
        """Report whether tokens can be removed from the end."""
        return False

    def size(self) -> int:
        """
        Return the size (i.e. sequence length) of the cache.

        Not every cache is required to implement this, in which case the size
        will always be 0 (though the cache may not be empty).
        """
        return 0

    @property
    def nbytes(self) -> int:
        """Return the size of this cache in bytes"""
        raise NotImplementedError("Cache sub-class must implement nbytes")

    def empty(self) -> bool:
        """
        Return if the cache is empty or not.
        """
        raise NotImplementedError("Cache sub-class must implement this.")

    @classmethod
    def from_state(cls, state: StateTree, meta_state: StateTree) -> Self:
        """Rebuild a cache from snapshots without calling ``__init__``."""
        # Create an instance of cls without calling __init__
        obj = cls.__new__(cls)
        obj.state = state
        obj.meta_state = meta_state
        return obj


class QuantizedKVCache(BaseCache):
    """Append uniformly quantized K/V in packed MLX format.

    Inputs are floating ``(batch, heads, tokens, width)`` arrays. ``state``
    holds packed values, scales and biases up to the logical ``offset``;
    ``meta_state`` holds offset, group size and bit width.
    """

    step = 256

    def __init__(self, group_size: int = 64, bits: int = 8) -> None:
        """Create an empty cache packing keys and values at ``bits`` per value."""
        self.keys: QuantizedTensor | None = None
        self.values: QuantizedTensor | None = None
        self.offset: int = 0
        self.group_size = group_size
        self.bits = bits

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[QuantizedTensor, QuantizedTensor]:
        """Quantize one step of keys and values and append it to the buffers.

        Storage grows in ``step`` increments; the returned packed triples are
        sliced to the logical offset.
        """
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self.offset

        if self.keys is None or (prev + num_steps) > self.keys[0].shape[-2]:
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            shape = (B, n_kv_heads, new_steps)

            def init_quant(dim: int) -> QuantizedTensor:
                return (
                    mx.zeros((*shape, dim * self.bits // 32), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                )

            def expand_quant(x: mx.array) -> mx.array:
                new_x = mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)
                return mx.concatenate([x, new_x], axis=-2)

            if self.keys is not None:
                assert self.values is not None
                assert self.values is not None
                if prev % self.step != 0:
                    self.keys = _map_packed(lambda x: x[..., :prev, :], self.keys)
                    self.values = _map_packed(lambda x: x[..., :prev, :], self.values)

                self.keys = _map_packed(expand_quant, self.keys)
                self.values = _map_packed(expand_quant, self.values)
            else:
                self.keys, self.values = init_quant(k_head_dim), init_quant(v_head_dim)

        self.offset += num_steps

        q_keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        q_values = mx.quantize(values, group_size=self.group_size, bits=self.bits)
        assert self.keys is not None and self.values is not None
        for i in range(len(self.keys)):
            self.keys[i][..., prev : self.offset, :] = q_keys[i]
            self.values[i][..., prev : self.offset, :] = q_values[i]

        return (
            _map_packed(lambda x: x[..., : self.offset, :], self.keys),
            _map_packed(lambda x: x[..., : self.offset, :], self.values),
        )

    @property
    def state(self) -> PackedPairState:
        """Return packed keys and values sliced to the logical offset."""
        if self.keys is None or self.offset == self.keys[0].shape[2]:
            return self.keys, self.values
        else:
            assert self.values is not None
            return (
                _map_packed(lambda x: x[..., : self.offset, :], self.keys),
                _map_packed(lambda x: x[..., : self.offset, :], self.values),
            )

    @state.setter
    def state(self, v: StateTree, /) -> None:
        self.keys, self.values = _packed_pair_state(v)

    @property
    def meta_state(self):
        """Return offset, group size and bit width as text."""
        return tuple(map(str, (self.offset, self.group_size, self.bits)))

    @meta_state.setter
    def meta_state(self, v: StateTree, /) -> None:
        self.offset, self.group_size, self.bits = _metadata_ints(v, 3)

    def size(self) -> int:
        """Return the number of cached tokens."""
        return self.offset

    def is_trimmable(self) -> bool:
        """Report that quantized tokens can be trimmed."""
        return True

    def trim(self, n: int) -> int:
        """Drop up to ``n`` trailing tokens, returning the number removed."""
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(
        self, N: int, return_array: bool = False, window_size: int | None = None
    ) -> mx.array | str | None:
        """Build the mask for ``N`` new tokens against this cache."""
        return create_attention_mask(
            N, offset=self.offset, return_array=return_array, window_size=window_size
        )

    def empty(self) -> bool:
        """Report whether no keys have been stored."""
        return self.keys is None

    @property
    def nbytes(self) -> int:
        """Return bytes held by packed keys and values, including spare capacity."""
        return cache_nbytes((self.keys, self.values))


class KVCache(BaseCache):
    """Append attention keys and values in 256-token allocation increments.

    Inputs have shape ``(batch, heads, tokens, head_dim)``; key and value head
    widths can differ. ``offset`` is the logical token count, while allocated
    buffers can be longer. ``state`` and ``update_and_fetch`` expose only the
    logical prefix; ``nbytes`` includes unused capacity.
    """

    step = 256

    def __init__(self) -> None:
        """Create an empty dense cache."""
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset: int = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Append one step of keys and values and return the logical prefix.

        Storage grows in ``step``-token increments; the returned arrays are
        sliced to ``offset`` so attention never reads spare capacity.
        """
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                assert self.values is not None
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        assert self.keys is not None and self.values is not None
        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def size(self) -> int:
        """Return the number of cached tokens."""
        return self.offset

    @property
    def state(self) -> tuple[mx.array | None, mx.array | None]:
        """Return keys and values sliced to the logical offset."""
        if self.keys is None or self.offset == self.keys.shape[2]:
            return self.keys, self.values
        else:
            assert self.values is not None
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )

    @state.setter
    def state(self, v: StateTree, /) -> None:
        self.keys, self.values = _array_pair_state(v)
        self.offset = 0 if self.keys is None else self.keys.shape[2]

    def is_trimmable(self) -> bool:
        """Report that dense KV entries can be trimmed."""
        return True

    def trim(self, n: int) -> int:
        """Drop up to ``n`` trailing tokens, returning the number removed."""
        n = min(self.offset, n)
        self.offset -= n
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4) -> QuantizedKVCache:
        """Return a quantized copy of the populated cache.

        The logical offset is preserved and every allocated token, including
        spare capacity, is quantized.
        """
        quant_cache = QuantizedKVCache(group_size=group_size, bits=bits)
        quant_cache.offset = self.offset
        if self.keys is not None:
            assert self.values is not None
            quant_cache.keys = mx.quantize(self.keys, group_size=group_size, bits=bits)
            quant_cache.values = mx.quantize(
                self.values, group_size=group_size, bits=bits
            )
        return quant_cache

    def make_mask(
        self, N: int, return_array: bool = False, window_size: int | None = None
    ) -> mx.array | str | None:
        """Build the mask for ``N`` new tokens against this cache."""
        return create_attention_mask(
            N, offset=self.offset, return_array=return_array, window_size=window_size
        )

    @classmethod
    def merge(cls, caches: Sequence[KVCache]) -> BatchKVCache:
        """Merge single-row caches into one left-padded batch cache."""
        return BatchKVCache.merge(caches)

    def empty(self) -> bool:
        """Report whether no keys have been stored."""
        return self.keys is None

    @property
    def nbytes(self) -> int:
        """Return bytes held by the key and value buffers, including spare capacity."""
        if self.keys is None:
            return 0
        assert self.values is not None
        return self.keys.nbytes + self.values.nbytes


class RotatingKVCache(BaseCache):
    """Keep a fixed prefix and a rotating window of attention keys and values.

    ``offset`` counts all tokens processed and ``position`` is a buffer cursor,
    so they diverge after rotation. Single-token updates may return ring-buffer
    order; ``temporal_order`` reorders it. Chunked updates can temporarily
    exceed ``max_size`` to preserve each query's window. A rotated cache cannot
    be trimmed back to an arbitrary earlier prefix.
    """

    def memory_profile(self, token_count: int) -> CacheMemory:
        """Profile the windowed KV buffers for allocation accounting."""
        return _windowed_memory_profile(self, token_count)

    step = 256

    def __init__(self, max_size: int, keep: int = 0) -> None:
        """Create an empty cache with ``keep`` fixed prefix rows per layer.

        The rotating window holds ``max_size`` tokens; ``keep`` rows are never
        overwritten, so they stay available to every query.
        """
        self.keep: int = keep
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset: int = 0
        self.max_size: int = max_size
        self.position: int = 0

    def _trim(
        self, trim_size: int, v: mx.array, append: mx.array | None = None
    ) -> mx.array:
        to_cat = []
        if trim_size > 0:
            to_cat = [v[..., : self.keep, :], v[..., trim_size + self.keep :, :]]
        else:
            to_cat = [v]
        if append is not None:
            to_cat.append(append)
        return mx.concatenate(to_cat, axis=2)

    def temporal_order(self, v: mx.array) -> mx.array:
        """
        Rearrange the cache into temporal order, slicing off the end if unused.
        """
        if self.position == v.shape[2]:
            return v
        elif self.position < self.offset:
            return mx.concatenate(
                [
                    v[..., : self.keep, :],
                    v[..., self.position :, :],
                    v[..., self.keep : self.position, :],
                ],
                axis=2,
            )
        else:
            return v[..., : self.position, :]

    def _update_concat(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            assert self.values is not None
            # Put the keys/values in temporal order to
            # preserve context
            self.keys = self.temporal_order(self.keys)
            self.values = self.temporal_order(self.values)
            self.position = self.keys.shape[2]

            # The largest size is self.max_size + S - 1 to ensure
            # every token gets at least self.max_size context
            trim_size = self.position - self.max_size + 1
            self.keys = self._trim(trim_size, self.keys, keys)
            self.values = self._trim(trim_size, self.values, values)
        assert self.keys is not None and self.values is not None
        self.offset += keys.shape[2]
        self.position = self.keys.shape[2]
        return self.keys, self.values

    def _update_in_place(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        # May not have hit the max size yet, so potentially
        # keep growing the cache
        B, n_kv_heads, S, k_head_dim = keys.shape
        prev = self.offset
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            v_head_dim = values.shape[3]
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                assert self.values is not None
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self.position = prev

        assert self.keys is not None and self.values is not None
        # Trim if needed
        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self.position = self.max_size

        # Rotate
        if self.position == self.max_size:
            self.position = self.keep

        # Assign
        self.keys[..., self.position : self.position + S, :] = keys
        self.values[..., self.position : self.position + S, :] = values
        self.offset += S
        self.position += S

        # If the buffer is not full, slice off the end
        if self.offset < self.max_size:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        return self.keys, self.values

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Append keys and values, rotating the window once it is full.

        Single-token steps may return ring-buffer order for the attention
        layer; multi-token steps concatenate in temporal order.
        """
        if keys.shape[2] == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

    def size(self) -> int:
        """Return the number of cached tokens within the window."""
        return min(self.offset, self.max_size)

    @property
    def state(self) -> tuple[mx.array | None, mx.array | None]:
        """Return ring-buffer keys and values without reordering them."""
        if self.keys is not None and self.offset < self.keys.shape[2]:
            assert self.values is not None
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        else:
            return self.keys, self.values

    @state.setter
    def state(self, v: StateTree, /) -> None:
        self.keys, self.values = _array_pair_state(v)

    @property
    def meta_state(self):
        """Return keep count, window size, offset and buffer cursor as text."""
        return tuple(map(str, (self.keep, self.max_size, self.offset, self.position)))

    @meta_state.setter
    def meta_state(self, v: StateTree, /) -> None:
        self.keep, self.max_size, self.offset, self.position = _metadata_ints(v, 4)

    def is_trimmable(self) -> bool:
        """Report whether the cache has not rotated yet."""
        return self.offset < self.max_size

    def trim(self, n: int) -> int:
        """Drop up to ``n`` trailing tokens from the offset and cursor."""
        n = min(self.offset, n)
        self.offset -= n
        self.position -= n
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4) -> QuantizedKVCache:
        """Reject quantization; rotating caches have no quantized layout."""
        raise NotImplementedError("RotatingKVCache Quantization NYI")

    def make_mask(
        self, N: int, window_size: Optional[int] = None, return_array: bool = False
    ):
        """Build the mask for ``N`` new tokens, unrolling a full window."""
        if N > 1:
            window_size = window_size or self.max_size
            offset = min(self.max_size - 1, self.offset)
            if offset + N > window_size or return_array:
                return create_causal_mask(N, offset, window_size=window_size)
            else:
                return "causal"
        else:
            if window_size is None:
                return None
            # May need a mask for when window_size < max_size
            if self.offset >= window_size and self.max_size > window_size:
                idx = self.position
                if idx >= self.max_size:
                    idx = 0
                if self.offset < self.max_size:
                    mask_size = self.offset + 1
                else:
                    mask_size = self.max_size
                mask = mx.arange(mask_size) >= (mask_size - window_size)
                mask = mx.roll(mask, idx + 1)
                return mask

    @classmethod
    def merge(cls, caches: Sequence[RotatingKVCache]) -> BatchRotatingKVCache:
        """Merge rotating caches into one left-padded batch cache."""
        return BatchRotatingKVCache.merge(caches)

    def empty(self) -> bool:
        """Report whether no keys have been stored."""
        return self.keys is None

    @property
    def nbytes(self) -> int:
        """Return bytes held by the key and value buffers."""
        if self.keys is None:
            return 0
        assert self.values is not None
        return self.keys.nbytes + self.values.nbytes


def dynamic_roll(x: mx.array, shifts: mx.array, axis: int) -> mx.array:
    """Roll each row of ``x`` along ``axis`` by its own shift."""
    n = x.shape[axis]
    expand_shifts = (...,) + (None,) * (x.ndim - axis)
    expand_indices = expand_shifts[:-1]
    idx = (mx.arange(n)[expand_indices] - shifts[expand_shifts]) % n
    rolled = mx.take_along_axis(x, idx, axis=axis)
    return rolled


class BatchKVCache(BaseCache):
    """Batched dense KV cache over left-padded rows.

    Each row tracks its own logical ``offset`` (negative while only padding is
    cached) while ``position`` counts shared buffer columns. ``keys`` and
    ``values`` are padded on the left so rows share one attention mask.
    """

    left_padding: mx.array
    step = 256

    def __init__(self, left_padding: List[int]):
        """
        The BatchKV cache expects inputs to be left-padded.

        E.g. the following prompts:

            [1, 3, 5]
            [7]
            [2, 6, 8, 9]

        Should be padded like so:

            [0, 1, 3, 5]
            [0, 0, 0, 7]
            [2, 6, 8, 9]

        And ``left_padding`` specifies the amount of padding for each.
        In this case, ``left_padding = [1, 3, 0]``.
        """
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.left_padding = mx.array(list(left_padding))
        self.offset: mx.array = mx.array([-pad for pad in left_padding])
        self.position: int = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Append one step for every row and return the shared column prefix."""
        prev = self.position
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                assert self.values is not None
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        assert self.keys is not None and self.values is not None
        self.offset += keys.shape[2]
        self.position += keys.shape[2]
        self.keys[..., prev : self.position, :] = keys
        self.values[..., prev : self.position, :] = values
        return self.keys[..., : self.position, :], self.values[..., : self.position, :]

    @property
    def state(self) -> tuple[mx.array | None, mx.array | None, mx.array, mx.array]:
        """Return keys, values, per-row offsets and left padding."""
        k, v = self.keys, self.values
        if k is not None and self.position < k.shape[2]:
            assert v is not None
            k = k[..., : self.position, :]
            v = v[..., : self.position, :]
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, v: StateTree) -> None:
        keys, values, offset, padding = _state_items(v, 4)
        self.keys, self.values = _array_pair_state((keys, values))
        if not isinstance(offset, mx.array) or not isinstance(padding, mx.array):
            raise TypeError("Batch cache offset and left padding must be MLX arrays")
        self.offset, self.left_padding = offset, padding
        self.position = 0 if self.keys is None else self.keys.shape[2]

    def is_trimmable(self) -> bool:
        """Report that batch rows can be trimmed."""
        return True

    def trim(self, n: int) -> int:
        """Drop up to ``n`` trailing columns from every row."""
        n = min(self.position, n)
        self.position -= n
        self.offset -= n
        return n

    def make_mask(
        self,
        N: int,
        return_array: bool = False,
        *,
        window_size: int | None = None,
        right_padding: mx.array | None = None,
    ) -> mx.array:
        """Build the four-dimensional mask for ``N`` new tokens with row padding."""
        return create_causal_mask(
            N,
            offset=self.position,
            left_padding=self.left_padding,
            window_size=window_size,
            right_padding=right_padding,
        )

    def filter(self, batch_indices: mx.array) -> None:
        """
        In-place filter to keep just the given indices in the cache.
        """
        if self.keys is not None:
            assert self.values is not None
            self.keys = self.keys[batch_indices]
            self.values = self.values[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

        # Shift left to reduce padding
        min_left_pad = parse_int(self.left_padding.min().item())
        if min_left_pad > 0:
            if self.keys is not None:
                assert self.values is not None
                self.keys = self.keys[..., min_left_pad:, :]
                self.values = self.values[..., min_left_pad:, :]
            self.position -= min_left_pad
            self.left_padding -= min_left_pad

    def extend(self, other: BatchKVCache) -> None:
        """
        In-place extend this cache with the other cache.
        """
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        max_idx = max(self.position, other.position)
        first_len = 0 if self.keys is None else self.keys.shape[2]
        other_len = 0 if other.keys is None else other.keys.shape[2]
        ref_keys = self.keys if self.keys is not None else other.keys
        ref_values = self.values if self.values is not None else other.values
        assert ref_keys is not None and ref_values is not None
        heads, key_dim, value_dim = (
            ref_keys.shape[1],
            ref_keys.shape[3],
            ref_values.shape[3],
        )
        max_size = max(first_len, other_len)

        def pad(c: BatchKVCache) -> tuple[mx.array, mx.array, mx.array, mx.array]:
            k, v = c.keys, c.values
            if k is None:
                Bc = c.offset.shape[0]
                k = mx.reshape(mx.array([]), (Bc, heads, 0, key_dim))
                v = mx.reshape(mx.array([]), (Bc, heads, 0, value_dim))
            assert v is not None
            left = max_idx - c.position
            assert v is not None
            right = max_size - k.shape[2] - left
            if right < 0:
                k = k[..., :right, :]
                v = v[..., :right, :]
                right = 0
            if left != 0 or right != 0:
                pad = [(0, 0), (0, 0), (left, right), (0, 0)]
                k = mx.pad(k, pad)
                v = mx.pad(v, pad)
            left_padding = c.left_padding + left
            return k, v, c.offset, left_padding

        left = pad(self)
        right = pad(other)
        self.keys, self.values, self.offset, self.left_padding = (
            mx.concatenate([left[0], right[0]]),
            mx.concatenate([left[1], right[1]]),
            mx.concatenate([left[2], right[2]]),
            mx.concatenate([left[3], right[3]]),
        )
        self.position = max_idx

    @classmethod
    def merge(cls, caches: Sequence[KVCache]) -> BatchKVCache:
        """Merge single-row caches into one left-padded batch cache."""
        lengths = [c.size() for c in caches]
        max_length = max(lengths)

        # No cache has content so make an empty one
        if max_length == 0:
            return BatchKVCache([0] * len(caches))

        padding = [max_length - length for length in lengths]
        B = len(caches)
        H = max(c.keys.shape[1] for c in caches if c.keys is not None)
        Dk = max(c.keys.shape[3] for c in caches if c.keys is not None)
        Dv = max(c.values.shape[3] for c in caches if c.values is not None)
        dt = next(iter(c.keys.dtype for c in caches if c.keys is not None))

        keys = mx.zeros((B, H, max_length, Dk), dtype=dt)
        values = mx.zeros((B, H, max_length, Dv), dtype=dt)
        for i, (p, c) in enumerate(zip(padding, caches)):
            if c.keys is None:
                continue
            keys[i : i + 1, :, p : p + c.offset] = c.keys[..., : c.offset, :]
            assert c.values is not None
            values[i : i + 1, :, p : p + c.offset] = c.values[..., : c.offset, :]

        cache = cls(padding)
        cache.keys = keys
        cache.values = values
        cache.offset += keys.shape[2]
        cache.position = keys.shape[2]

        return cache

    def size(self) -> int:
        """Return the number of cached columns."""
        return self.position

    def empty(self) -> bool:
        """Report whether no keys have been stored."""
        return self.keys is None

    @property
    def batch_size(self) -> int:
        """Return the number of rows in the batch."""
        if self.keys is not None:
            assert self.values is not None
            return int(self.keys.shape[0])
        return int(self.left_padding.shape[0])

    def is_single_row(self) -> bool:
        """Report whether the batch has exactly one row."""
        return self.batch_size == 1

    @property
    def nbytes(self) -> int:
        """Return bytes held by the key and value buffers."""
        if self.keys is None:
            return 0
        assert self.values is not None
        return self.keys.nbytes + self.values.nbytes


class BatchRotatingKVCache(BaseCache):
    """Batched rotating K/V with per-row offsets and left-padding metadata.

    ``position`` and ``_offset`` track shared buffer columns; ``offset`` tracks
    per-row logical positions. ``prepare``/``finalize`` handle right-padded
    prefill, with ``lengths`` required for nonzero right padding. State includes
    K/V, row offsets and left padding; rotation and cursor metadata are separate.
    Quantization is not implemented for this cache type.
    """

    def memory_profile(self, token_count: int) -> CacheMemory:
        """Profile the windowed batch buffers for allocation accounting."""
        return _windowed_memory_profile(self, token_count)

    step = 256

    def __init__(self, max_size: int, left_padding: List[int]):
        """Create an empty batched rotating cache for left-padded rows."""
        self.keys: mx.array | None = None
        self.values: mx.array | None = None

        self.left_padding = mx.array(list(left_padding))
        self.offset: mx.array = mx.array([-pad for pad in left_padding])

        self.max_size: int = max_size
        self.position: int = 0
        self._offset = 0
        self.rotated = False

    def _trim(
        self, trim_size: int, v: mx.array, append: mx.array | None = None
    ) -> mx.array:
        if trim_size > 0:
            v = v[..., trim_size:, :]
        if append is not None:
            return mx.concatenate([v, append], axis=2)
        return v

    def _temporal_order(self) -> None:
        """
        Rearrange the cache into temporal order.
        """
        if self.rotated:
            assert self.keys is not None and self.values is not None
            self.keys = mx.roll(self.keys, -self.position, 2)
            self.values = mx.roll(self.values, -self.position, 2)
            self.position = self.keys.shape[2]
            self.rotated = False

    def _update_concat(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            # Put the keys/values in temporal order to
            # preserve context
            self._temporal_order()
            assert self.keys is not None and self.values is not None

            # Slice off the end if needed
            if self.keys.shape[2] > self.position:
                self.keys = self.keys[..., : self.position, :]
                self.values = self.values[..., : self.position, :]

            # The largest size is self.max_size + S - 1 to ensure
            # every token gets at least self.max_size context
            trim_size = self.position - self.max_size + 1
            if trim_size > 0:
                self.left_padding -= trim_size
            self.keys = self._trim(trim_size, self.keys, keys)
            self.values = self._trim(trim_size, self.values, values)
        assert self.keys is not None and self.values is not None
        self.offset += keys.shape[2]
        self._offset += keys.shape[2]
        self.position = self.keys.shape[2]

        # Make sure left_padding and offset are evaluated
        self.keys = array_depends(self.keys, (self.left_padding, self.offset))

        return self.keys, self.values

    def _update_in_place(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        # May not have hit the max size yet, so potentially
        # keep growing the cache
        B, n_kv_heads, S, k_head_dim = keys.shape
        prev = self._offset
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            v_head_dim = values.shape[3]
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                assert self.values is not None
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self.position = prev

        assert self.keys is not None and self.values is not None
        # Trim if needed
        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self.position = self.max_size
            self.left_padding -= trim_size

        # Rotate
        if self.position == self.max_size:
            self.rotated = True
            self.position = 0
        if self.rotated:
            self.left_padding -= S

        # Assign
        self.keys[..., self.position : self.position + S, :] = keys
        self.values[..., self.position : self.position + S, :] = values
        self._offset += S
        self.offset += S
        self.position += S

        # Make sure left_padding and offset are evaluated
        self.keys = array_depends(self.keys, (self.left_padding, self.offset))

        # If the buffer is not full, slice off the end
        if self._offset < self.max_size:
            return (
                self.keys[..., : self._offset, :],
                self.values[..., : self._offset, :],
            )
        return self.keys, self.values

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Append keys and values, rotating rows once the window fills."""
        if keys.shape[2] == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

    @property
    def state(self) -> tuple[mx.array | None, mx.array | None, mx.array, mx.array]:
        """Return keys, values, per-row offsets and left padding."""
        k, v = self.keys, self.values
        if k is not None and self._offset < k.shape[2]:
            assert v is not None
            k, v = k[..., : self._offset, :], v[..., : self._offset, :]
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, v: StateTree) -> None:
        keys, values, offset, padding = _state_items(v, 4)
        self.keys, self.values = _array_pair_state((keys, values))
        if not isinstance(offset, mx.array) or not isinstance(padding, mx.array):
            raise TypeError("Batch cache offset and left padding must be MLX arrays")
        self.offset, self.left_padding = offset, padding

    @property
    def meta_state(self) -> tuple[str, str, str, str]:
        """Return window size, buffer offset, cursor and rotation flag as text."""
        return (
            str(self.max_size),
            str(self._offset),
            str(self.position),
            str(self.rotated),
        )

    @meta_state.setter
    def meta_state(self, v: StateTree) -> None:
        maximum, offset, position, rotated = _state_items(v, 4)
        self.max_size, self._offset, self.position = map(
            parse_int, (maximum, offset, position)
        )
        if isinstance(rotated, bool):
            self.rotated = rotated
        elif rotated in ("True", "False"):
            self.rotated = rotated == "True"
        else:
            raise ValueError("Rotating cache metadata requires a boolean rotation flag")

    def is_trimmable(self) -> bool:
        """Report whether the cache has not rotated yet."""
        return self._offset < self.max_size

    def trim(self, n: int) -> int:
        """Drop up to ``n`` trailing columns from every row."""
        n = min(self._offset, n)
        self._offset -= n
        self.position -= n
        self.offset -= n
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4) -> QuantizedKVCache:
        """Reject quantization; batch rotating caches have no quantized layout."""
        raise NotImplementedError("BatchRotatingKVCache Quantization NYI")

    def make_mask(
        self, N: int, window_size: Optional[int] = None, return_array: bool = False
    ) -> mx.array:
        """Build the four-dimensional mask for ``N`` new tokens.

        Rows that already rotated have their window columns rolled back into
        temporal order so each query sees the correct keys.
        """
        left_padding = self.left_padding
        window_size = window_size or self.max_size
        offset = min(self.max_size - 1, self._offset)
        rinds = mx.arange(offset + N)
        linds = mx.arange(offset, offset + N) if offset else rinds
        linds = linds[:, None]
        rinds = rinds[None]
        mask = linds >= rinds
        mask &= linds < rinds + window_size
        in_place = N == 1
        if (trim_size := self.position - self.max_size + int(not in_place)) > 0:
            left_padding = left_padding - trim_size

        rotated = in_place and (self.rotated or self.position >= self.max_size)
        if rotated:
            left_padding = left_padding - 1

        mask = mask & (rinds >= mx.expand_dims(left_padding, (1, 2, 3)))

        if rotated:
            idx = self.position
            if idx >= self.max_size:
                idx = 0
            mask = mx.roll(mask, idx + 1, -1)

        return mask

    def filter(self, batch_indices: mx.array) -> None:
        """
        In-place filter to keep just the given indices in the cache.
        """
        if self.keys is not None:
            assert self.values is not None
            self.keys = self.keys[batch_indices]
            self.values = self.values[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

    def extend(self, other: BatchRotatingKVCache) -> None:
        """
        In-place extend this cache with the other cache.
        """
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        if (self.rotated != other.rotated) or self.position != other.position:
            self._temporal_order()
            other._temporal_order()

        max_idx = max(self.position, other.position)
        first_len = 0 if self.keys is None else self.keys.shape[2]
        other_len = 0 if other.keys is None else other.keys.shape[2]
        ref_keys = self.keys if self.keys is not None else other.keys
        ref_values = self.values if self.values is not None else other.values
        assert ref_keys is not None and ref_values is not None
        heads, key_dim, value_dim = (
            ref_keys.shape[1],
            ref_keys.shape[3],
            ref_values.shape[3],
        )
        max_size = max(first_len, other_len)

        def pad(
            c: BatchRotatingKVCache,
        ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
            left = max_idx - c.position
            k, v = c.keys, c.values
            if k is None:
                Bc = c.offset.shape[0]
                k = mx.reshape(mx.array([]), (Bc, heads, 0, key_dim))
                v = mx.reshape(mx.array([]), (Bc, heads, 0, value_dim))
            assert v is not None
            right = max_size - k.shape[2] - left
            if right < 0:
                k = k[..., :right, :]
                v = v[..., :right, :]
                right = 0
            if left != 0 or right != 0:
                pad = [(0, 0), (0, 0), (left, right), (0, 0)]
                k = mx.pad(k, pad)
                v = mx.pad(v, pad)
            left_padding = c.left_padding + left
            return k, v, c.offset, left_padding

        left = pad(self)
        right = pad(other)
        self.keys, self.values, self.offset, self.left_padding = (
            mx.concatenate([left[0], right[0]]),
            mx.concatenate([left[1], right[1]]),
            mx.concatenate([left[2], right[2]]),
            mx.concatenate([left[3], right[3]]),
        )
        self.position = max_idx
        self._offset = max(self._offset, other._offset)

    @classmethod
    def merge(cls, caches: Sequence[RotatingKVCache]) -> BatchRotatingKVCache:
        """Merge single-row rotating caches into one left-padded batch cache."""
        if not all(c.max_size == caches[0].max_size for c in caches):
            raise ValueError(
                "BatchRotatingKVCache can only merge caches with the same maximum size"
            )

        offsets = [c.offset for c in caches]
        lengths = [c.size() for c in caches]
        max_length = max(lengths)

        # No cache has content so make an empty one
        if max_length == 0:
            return cls(caches[0].max_size, [0] * len(caches))

        padding = [max_length - length for length in lengths]
        B = len(caches)
        H = max(c.keys.shape[1] for c in caches if c.keys is not None)
        Dk = max(c.keys.shape[3] for c in caches if c.keys is not None)
        Dv = max(c.values.shape[3] for c in caches if c.values is not None)
        dt = next(iter(c.keys.dtype for c in caches if c.keys is not None))

        keys = mx.zeros((B, H, max_length, Dk), dtype=dt)
        values = mx.zeros((B, H, max_length, Dv), dtype=dt)
        for i, (p, length, c) in enumerate(zip(padding, lengths, caches)):
            if c.keys is None or length == 0:
                continue
            keys[i : i + 1, :, p : p + length] = c.temporal_order(c.keys)[
                ..., -length:, :
            ]
            assert c.values is not None
            values[i : i + 1, :, p : p + length] = c.temporal_order(c.values)[
                ..., -length:, :
            ]

        cache = cls(caches[0].max_size, padding)
        cache.keys = keys
        cache.values = values
        cache.offset = mx.array(offsets)
        cache.position = keys.shape[2]
        cache._offset = keys.shape[2]

        return cache

    def size(self) -> int:
        """Return the number of cached tokens within the window."""
        return min(self._offset, self.max_size)

    def empty(self) -> bool:
        """Report whether no keys have been stored."""
        return self.keys is None

    @property
    def batch_size(self) -> int:
        """Return the number of rows in the batch."""
        if self.keys is not None:
            assert self.values is not None
            return int(self.keys.shape[0])
        return int(self.left_padding.shape[0])

    def is_single_row(self) -> bool:
        """Report whether the batch has exactly one row."""
        return self.batch_size == 1

    @property
    def nbytes(self) -> int:
        """Return bytes held by the key and value buffers."""
        if self.keys is None:
            return 0
        assert self.values is not None
        return self.keys.nbytes + self.values.nbytes


# MLX-VLM cache extensions.


class BatchQuantizedKVCache(BaseCache):
    """Batch-aware quantized KV cache for continuous batching.

    Mirrors ``BatchKVCache`` but stores keys/values in quantized form
    (packed uint32 + scales + biases), matching the format produced by
    ``mx.quantize``.  Supports ``extend()`` and ``filter()`` so that
    ``Batch.extend`` / ``Batch.filter`` work during continuous-batching.
    """

    step = 256

    def __init__(
        self,
        left_padding: List[int],
        group_size: int = 64,
        bits: int = 8,
    ):
        """Create an empty batched quantized cache for left-padded rows."""
        self.keys: QuantizedTensor | None = None
        self.values: QuantizedTensor | None = None
        self.left_padding = mx.array(list(left_padding))
        self.offset: mx.array = mx.array([-lp for lp in left_padding])
        self.position: int = 0
        self.group_size = group_size
        self.bits = bits

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[QuantizedTensor, QuantizedTensor]:
        """Quantize incoming keys/values and append to the cache.

        Args:
            keys:   float [B, H, S, D_k]
            values: float [B, H, S, D_v]

        Returns:
            Quantized (keys_tuple, values_tuple) sliced to ``position``.
        """
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self.position

        if self.keys is None or (prev + num_steps) > self.keys[0].shape[-2]:
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            shape = (B, n_kv_heads, new_steps)

            def _init(dim: int) -> QuantizedTensor:
                return (
                    mx.zeros((*shape, dim * self.bits // 32), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                )

            def _expand(x: mx.array) -> mx.array:
                new_x = mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)
                return mx.concatenate([x, new_x], axis=-2)

            if self.keys is not None:
                assert self.values is not None
                if prev % self.step != 0:
                    self.keys = _map_packed(lambda k: k[..., :prev, :], self.keys)
                    self.values = _map_packed(lambda v: v[..., :prev, :], self.values)
                self.keys = _map_packed(_expand, self.keys)
                self.values = _map_packed(_expand, self.values)
            else:
                self.keys = _init(k_head_dim)
                self.values = _init(v_head_dim)

        self.offset += num_steps
        self.position += num_steps

        assert self.keys is not None and self.values is not None
        q_keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        q_values = mx.quantize(values, group_size=self.group_size, bits=self.bits)
        for i in range(len(self.keys)):
            self.keys[i][..., prev : self.position, :] = q_keys[i]
            self.values[i][..., prev : self.position, :] = q_values[i]

        return (
            _map_packed(lambda k: k[..., : self.position, :], self.keys),
            _map_packed(lambda v: v[..., : self.position, :], self.values),
        )

    def filter(self, batch_indices: mx.array):
        """Keep only the sequences at *batch_indices*."""
        if self.keys is not None:
            assert self.values is not None
            self.keys = _map_packed(lambda k: k[batch_indices], self.keys)
            self.values = _map_packed(lambda v: v[batch_indices], self.values)
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

        min_lp = parse_int(self.left_padding.min().item())
        if min_lp > 0:
            if self.keys is not None:
                assert self.values is not None
                self.keys = _map_packed(lambda k: k[..., min_lp:, :], self.keys)
                self.values = _map_packed(lambda v: v[..., min_lp:, :], self.values)
            self.position -= min_lp
            self.left_padding -= min_lp

    def extend(self, other: "BatchQuantizedKVCache"):
        """Concatenate *other* batch into this cache along the batch dim."""
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        max_idx = max(self.position, other.position)
        self_len = 0 if self.keys is None else self.keys[0].shape[-2]
        other_len = 0 if other.keys is None else other.keys[0].shape[-2]
        max_size = max(max_idx, self_len, other_len)
        ref_keys = self.keys if self.keys is not None else other.keys
        ref_values = self.values if self.values is not None else other.values
        assert ref_keys is not None and ref_values is not None

        def _empty_like(parts: QuantizedTensor, batch_size: int) -> QuantizedTensor:
            def empty(part: mx.array) -> mx.array:
                shape = list(part.shape)
                shape[0] = batch_size
                shape[-2] = 0
                return mx.zeros(tuple(shape), dtype=part.dtype)

            return _map_packed(empty, parts)

        def _pad_quant(
            cache_obj: BatchQuantizedKVCache,
        ) -> tuple[QuantizedTensor, QuantizedTensor, mx.array, mx.array]:
            if cache_obj.keys is None:
                batch_size = int(cache_obj.offset.shape[0])
                keys = _empty_like(ref_keys, batch_size)
                values = _empty_like(ref_values, batch_size)
            else:
                keys = cache_obj.keys
                values = cache_obj.values
                assert values is not None
            left = max_idx - cache_obj.position
            right = max_size - keys[0].shape[-2] - left
            if right < 0:
                trimmed_keys = _map_packed(lambda k: k[..., : max_size - left, :], keys)
                trimmed_values = _map_packed(
                    lambda v: v[..., : max_size - left, :], values
                )
                right = 0
            else:
                trimmed_keys = keys
                trimmed_values = values

            if left != 0 or right != 0:
                pad_spec = [(0, 0)] * len(trimmed_keys[0].shape)
                pad_spec[-2] = (left, right)
                padded_keys = _map_packed(lambda k: mx.pad(k, pad_spec), trimmed_keys)
                padded_values = _map_packed(
                    lambda v: mx.pad(v, pad_spec), trimmed_values
                )
            else:
                padded_keys = trimmed_keys
                padded_values = trimmed_values

            lp = cache_obj.left_padding + left
            return padded_keys, padded_values, cache_obj.offset, lp

        r_self = _pad_quant(self)
        r_other = _pad_quant(other)

        sk, sv, so, slp = r_self
        ok, ov, oo, olp = r_other

        self.keys = (
            mx.concatenate([sk[0], ok[0]]),
            mx.concatenate([sk[1], ok[1]]),
            mx.concatenate([sk[2], ok[2]]),
        )
        self.values = (
            mx.concatenate([sv[0], ov[0]]),
            mx.concatenate([sv[1], ov[1]]),
            mx.concatenate([sv[2], ov[2]]),
        )
        self.offset = mx.concatenate([so, oo])
        self.left_padding = mx.concatenate([slp, olp])
        self.position = max_idx

    @property
    def state(
        self,
    ) -> tuple[QuantizedTensor | None, QuantizedTensor | None, mx.array, mx.array]:
        """Return packed keys, values, per-row offsets and left padding."""
        if self.keys is None:
            return None, None, self.offset, self.left_padding
        assert self.values is not None
        k = _map_packed(lambda x: x[..., : self.position, :], self.keys)
        v = _map_packed(lambda x: x[..., : self.position, :], self.values)
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, val: StateTree) -> None:
        keys, values, offset, padding = _state_items(val, 4)
        self.keys, self.values = _packed_pair_state((keys, values))
        if (self.keys is None) != (self.values is None):
            raise ValueError(
                "Batch packed keys and values must both be present or absent"
            )
        if not isinstance(offset, mx.array) or not isinstance(padding, mx.array):
            raise TypeError("Batch cache offset and left padding must be MLX arrays")
        self.offset, self.left_padding = offset, padding
        self.position = self.keys[0].shape[2] if self.keys is not None else 0

    @property
    def meta_state(self) -> tuple[str, str, str]:
        """Return the buffer cursor, group size and bit width as text."""
        return str(self.position), str(self.group_size), str(self.bits)

    @meta_state.setter
    def meta_state(self, v: StateTree) -> None:
        self.position, self.group_size, self.bits = _metadata_ints(v, 3)

    def is_trimmable(self) -> bool:
        """Report that batch rows can be trimmed."""
        return True

    def trim(self, n: int) -> int:
        """Drop up to ``n`` trailing columns from every row."""
        n = min(self.position, n)
        self.position -= n
        self.offset -= n
        return n

    def empty(self) -> bool:
        """Report whether no keys have been stored."""
        return self.keys is None

    @property
    def batch_size(self) -> int:
        """Return the number of rows in the batch."""
        if self.keys is not None:
            assert self.values is not None
            return int(self.keys[0].shape[0])
        return int(self.left_padding.shape[0])

    def is_single_row(self) -> bool:
        """Report whether the batch has exactly one row."""
        return self.batch_size == 1

    def make_mask(
        self,
        N: int,
        return_array: bool = False,
        *,
        window_size: int | None = None,
        right_padding: mx.array | None = None,
    ) -> mx.array:
        """Build the four-dimensional mask for ``N`` new tokens with row padding."""
        return create_causal_mask(
            N,
            offset=self.position,
            left_padding=self.left_padding,
            window_size=window_size,
            right_padding=right_padding,
        )
