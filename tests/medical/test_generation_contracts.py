"""CLI contracts and tiny native feature checks never load a model."""

import mlx.core as mx
import pytest

from mlx_vlm.contracts import evaluate
from mlx_vlm.generate import dispatch


@pytest.mark.parametrize("value", [None, "features", [1], (1,), {"hidden": []}])
def test_vision_hook_rejects_non_array_outputs(value: object) -> None:
    with pytest.raises(TypeError, match="Vision encoder"):
        dispatch.require_vision_features(value)


def test_vision_hook_preserves_native_array_buffers() -> None:
    features = mx.zeros((1,), stream=mx.gpu)
    evaluate(features)
    assert dispatch.require_vision_features(features) is features
    grouped = [features]
    packed = (features,)
    assert dispatch.require_vision_features(grouped) is grouped
    assert dispatch.require_vision_features(packed) is packed


# Configuration and request ingress must preserve valid values and reject
# malformed containers before the native generation path consumes them.

from hypothesis import given, settings
from hypothesis import strategies as st


@given(st.lists(st.integers(), min_size=1, max_size=2))
def test_resize_shape_normalization_is_idempotent(sizes: list[int]) -> None:
    shape = dispatch.normalize_resize_shape(sizes)
    assert shape == (sizes[0], sizes[-1])
    assert dispatch.normalize_resize_shape(shape) == shape


json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(),
    lambda children: st.lists(children, max_size=4)
    | st.dictionaries(st.text(), children, max_size=4),
    max_leaves=12,
)


@given(json_values)
@settings(max_examples=40)
def test_resize_ingress_raises_only_validation_errors(value: object) -> None:
    try:
        shape = dispatch.normalize_resize_shape(value)
    except ValueError:
        return
    assert shape is None or (
        len(shape) == 2 and all(type(size) is int for size in shape)
    )
