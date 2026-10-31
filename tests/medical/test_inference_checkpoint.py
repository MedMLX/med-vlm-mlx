"""Strict medical checkpoint serialization and native Metal forward round trips."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest
from hypothesis import given
from hypothesis import strategies as st
from mlx.utils import tree_flatten

from mlx_vlm.contracts import evaluate, save_safetensors
from mlx_vlm.models.gemma3 import Model, ModelConfig, TextConfig, VisionConfig
from mlx_vlm.utils import load_model


@pytest.fixture(autouse=True)
def require_metal() -> None:
    if not mx.metal.is_available():
        raise RuntimeError("Inference checkpoint checks require Metal")
    mx.set_default_device(mx.gpu)


def tiny_model() -> Model:
    """Create a Gemma3 graph with the medical image path and tiny test widths."""
    return Model(
        ModelConfig(
            model_type="gemma3",
            image_token_index=63,
            text_config=TextConfig(
                model_type="gemma3",
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                vocab_size=64,
                sliding_window=32,
                query_pre_attn_scalar=8,
                mm_tokens_per_image=4,
            ),
            vision_config=VisionConfig(
                model_type="gemma3",
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=4,
                patch_size=2,
            ),
        )
    )


def save_checkpoint(path: Path, model: Model) -> None:
    """Serialize the exact native parameter graph and its architecture config."""
    from dataclasses import asdict

    (path / "config.json").write_text(json.dumps(asdict(model.config)))
    weights = dict(tree_flatten(model.parameters()))
    save_safetensors(
        str(path / "model.safetensors"), weights, metadata={"format": "mlx"}
    )


def test_strict_checkpoint_load_and_metal_forward(tmp_path: Path) -> None:
    model = tiny_model()
    model.eval()
    save_checkpoint(tmp_path, model)
    restored = load_model(tmp_path)
    assert isinstance(restored, Model)
    ids = mx.array([[1, 63, 63, 63, 63, 2]])
    pixels = mx.ones((1, 3, 4, 4), dtype=mx.float32)
    expected = model(ids, pixels).logits
    actual = restored(ids, pixels).logits
    evaluate(expected, actual)
    assert actual.shape == (1, 6, 64)
    assert bool(mx.all(mx.isfinite(actual)))
    assert bool(mx.allclose(actual, expected, atol=1e-6, rtol=1e-6))


def test_strict_load_rejects_missing_medical_weights(tmp_path: Path) -> None:
    model = tiny_model()
    save_checkpoint(tmp_path, model)
    weights = dict(tree_flatten(model.parameters()))
    weights.pop(next(iter(weights)))
    save_safetensors(str(tmp_path / "model.safetensors"), weights)
    with pytest.raises(ValueError, match="Missing|missing"):
        load_model(tmp_path)


@given(st.floats(min_value=-10, max_value=10, allow_nan=False, width=32))
def test_gemma_sanitize_preserves_native_head_and_is_idempotent(value: float) -> None:
    model = tiny_model()
    head = mx.full((64, 16), value, dtype=mx.float32)
    embedding = mx.full((64, 16), value + 1, dtype=mx.float32)
    weights = {
        "language_model.lm_head.weight": head,
        "language_model.model.embed_tokens.weight": embedding,
    }
    converted = model.language_model.sanitize(weights)
    repeated = model.language_model.sanitize(converted)
    assert converted.keys() == repeated.keys() == weights.keys()
    assert converted["language_model.lm_head.weight"] is head
    assert repeated["language_model.lm_head.weight"] is head


def test_gemma_sanitize_supplies_only_an_omitted_tied_head() -> None:
    model = tiny_model()
    embedding = mx.ones((64, 16))
    weights = {"language_model.model.embed_tokens.weight": embedding}
    converted = model.language_model.sanitize(weights)
    assert converted["language_model.lm_head.weight"] is embedding
    assert "language_model.lm_head.weight" not in weights
