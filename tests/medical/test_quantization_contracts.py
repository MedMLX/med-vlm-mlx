"""Synthetic contracts for sampling penalties."""

import mlx.core as mx
import pytest

from mlx_vlm.sample_utils import make_frequency_penalty, make_logits_processors


def test_frequency_penalty_counts_repeated_indices() -> None:
    processor = make_frequency_penalty(0.5)
    result = processor(mx.array([1, 1, 2]), mx.zeros((1, 4)))
    assert result.tolist() == [[0.0, -1.0, -0.5, 0.0]]
    with pytest.raises(ValueError, match="context size"):
        make_logits_processors(repetition_penalty=1.1, repetition_context_size=None)
