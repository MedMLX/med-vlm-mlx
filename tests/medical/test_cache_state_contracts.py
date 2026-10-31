"""Native cache persistence keeps live windows and absolute token positions."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlx_vlm.contracts import StateTree
from mlx_vlm.models.cache import (
    KVCache,
)


@given(st.lists(st.none() | st.integers() | st.text(), max_size=4))
@settings(max_examples=30, deadline=None)
def test_dense_cache_rejects_malformed_scalar_state(
    items: list[int | str | None],
) -> None:
    state: list[StateTree] = [item for item in items]
    if len(items) == 2 and all(item is None for item in items):
        assert KVCache.from_state(state, "").empty()
    else:
        with pytest.raises((TypeError, ValueError)):
            KVCache.from_state(state, "")
