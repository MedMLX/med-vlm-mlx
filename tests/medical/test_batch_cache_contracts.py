"""Tiny native batch-cache state and row-operation contracts, without checkpoints."""

import mlx.core as mx
import pytest

from mlx_vlm.contracts import array_depends, array_int_list, evaluate
from mlx_vlm.generate.common import maybe_quantize_kv_cache
from mlx_vlm.models.cache import (
    BatchKVCache,
    BatchQuantizedKVCache,
    BatchRotatingKVCache,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("missing_keys", [False, True])
def test_packed_state_requires_both_buffers(batched: bool, missing_keys: bool) -> None:
    packed = (mx.array([0], dtype=mx.uint32), mx.array([1.0]), mx.array([0.0]))
    keys, values = (None, packed) if missing_keys else (packed, None)
    with pytest.raises(ValueError, match="both"):
        if batched:
            cache = BatchQuantizedKVCache([0], group_size=64, bits=4)
            cache.state = (keys, values, mx.array([0]), mx.array([0]))
        else:
            QuantizedKVCache().state = (keys, values)


def test_array_dependency_preserves_value() -> None:
    source = mx.array([1, 2])
    result = array_depends(source, [mx.array([3])])
    evaluate(result)
    assert array_int_list(result) == [1, 2]


@pytest.mark.parametrize("kind", ["dense", "rotating", "packed"])
def test_batch_cache_state_roundtrip_and_row_selection(kind: str) -> None:
    cache = (
        BatchKVCache([0, 0])
        if kind == "dense"
        else (
            BatchRotatingKVCache(8, [0, 0])
            if kind == "rotating"
            else BatchQuantizedKVCache([0, 0], group_size=64, bits=4)
        )
    )
    keys = mx.ones((2, 1, 2, 64))
    cache.update_and_fetch(keys, keys * 2)
    if isinstance(cache, BatchKVCache):
        restored = BatchKVCache.from_state(cache.state, "")
    elif isinstance(cache, BatchRotatingKVCache):
        restored = BatchRotatingKVCache.from_state(cache.state, cache.meta_state)
        assert restored.rotated is False
    else:
        restored = BatchQuantizedKVCache.from_state(cache.state, cache.meta_state)
    assert restored.position == 2
    restored.filter(mx.array([1]))
    assert restored.batch_size == 1
    assert array_int_list(restored.offset) == [2]


def test_rotating_boolean_metadata_and_empty_roundtrip() -> None:
    cache = BatchRotatingKVCache(8, [0])
    restored = BatchRotatingKVCache.from_state(cache.state, cache.meta_state)
    assert restored.rotated is False
    restored.update_and_fetch(mx.ones((1, 1, 2, 64)), mx.ones((1, 1, 2, 64)))
    assert array_int_list(restored.offset) == [2]
    with pytest.raises(ValueError, match="boolean"):
        restored.meta_state = ("8", "2", "2", "invalid")


def test_dense_batch_extend_preserves_both_rows() -> None:
    left, right = BatchKVCache([0]), BatchKVCache([0])
    left.update_and_fetch(mx.ones((1, 1, 2, 64)), mx.ones((1, 1, 2, 64)))
    right.update_and_fetch(mx.ones((1, 1, 1, 64)) * 3, mx.ones((1, 1, 1, 64)) * 4)
    left.extend(right)
    assert left.batch_size == 2
    assert array_int_list(left.offset) == [2, 1]


def _filled_batch_cache(value: float) -> BatchKVCache:
    cache = BatchKVCache([0])
    x = mx.full((1, 1, 1, 8), value)
    cache.update_and_fetch(x, x)
    return cache


def test_generation_batch_rejects_missing_later_layer_before_mutation() -> None:
    from typing import cast

    from mlx_vlm.generate.ar import GenerationBatch
    from mlx_vlm.models.protocols import LanguageModelLike

    left_first, left_later = _filled_batch_cache(1), _filled_batch_cache(2)
    right_first = _filled_batch_cache(3)

    def sampler(logits: mx.array) -> mx.array:
        return mx.argmax(logits, axis=-1)

    left = GenerationBatch(
        cast(LanguageModelLike, object()),
        [1],
        mx.array([1]),
        [left_first, left_later],
        sampler,
        lambda token: False,
        [5],
    )
    right = GenerationBatch(
        cast(LanguageModelLike, object()),
        [2],
        mx.array([2]),
        [right_first],
        sampler,
        lambda token: False,
        [5],
    )

    with pytest.raises(ValueError, match="layer"):
        left.extend(right)

    assert left.uids == [1] and right.uids == [2]
    for entry in (left_first, left_later, right_first):
        assert entry.keys is not None and entry.keys.shape[0] == 1


def test_stream_kv_quantization_keeps_gemma3_sliding_caches() -> None:
    """Quantize full-attention caches while sliding-window rotating caches stay native."""
    states = mx.ones((1, 2, 4, 64))
    full, sliding = KVCache(), RotatingKVCache(max_size=8, keep=0)
    full.update_and_fetch(states, states)
    sliding.update_and_fetch(states, states)
    prompt_cache = [sliding, full]

    maybe_quantize_kv_cache(
        prompt_cache, quantized_kv_start=0, kv_group_size=64, kv_bits=4
    )

    assert prompt_cache[0] is sliding
    assert isinstance(prompt_cache[1], QuantizedKVCache)
