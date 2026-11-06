"""Saved native checkpoints load exactly and strict loading rejects missing weights."""

import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.contracts import JSONObject, evaluate, save_safetensors
from mlx_vlm.models.gemma3 import Model as GemmaModel
from mlx_vlm.models.gemma3 import ModelConfig as GemmaConfig
from mlx_vlm.utils import load_model, save_weights


def _checkpoint_config() -> JSONObject:
    shared: JSONObject = {"model_type": "gemma3", "hidden_size": 16}
    shared |= {
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
    }
    text: JSONObject = {**shared, "vocab_size": 64, "num_key_value_heads": 1}
    text |= {"head_dim": 8, "rms_norm_eps": 1e-6, "query_pre_attn_scalar": 8}
    text["mm_tokens_per_image"] = 4
    vision: JSONObject = {**shared, "image_size": 4, "patch_size": 2}
    config: JSONObject = {"model_type": "gemma3", "text_config": text}
    config |= {"vision_config": vision, "image_token_index": 63}
    return config


def test_saved_native_checkpoint_preserves_parameters_and_logits(
    tmp_path: Path,
) -> None:
    """Exercise native dispatch, nested configs and sanitizer order on saved weights."""
    config = _checkpoint_config()
    model = GemmaModel(GemmaConfig.from_dict(config))
    model.eval()
    weights = dict(tree_flatten(model.parameters()))
    ids = mx.array([[2, 7, 9]])
    expected = model.language_model(ids).logits
    evaluate(expected)
    save_weights(tmp_path, model)
    (tmp_path / "config.json").write_text(json.dumps(config))
    restored = load_model(tmp_path)
    assert type(restored) is type(model)
    assert asdict(restored.config) == asdict(model.config)
    recovered = dict(tree_flatten(restored.parameters()))
    assert recovered.keys() == weights.keys()
    assert all(mx.array_equal(recovered[key], weights[key]).item() for key in weights)
    actual = restored.language_model(ids).logits
    evaluate(expected, actual)
    assert bool(mx.array_equal(expected, actual).item())


def test_strict_load_rejects_missing_medical_weights(tmp_path: Path) -> None:
    config = _checkpoint_config()
    model = GemmaModel(GemmaConfig.from_dict(config))
    save_weights(tmp_path, model)
    (tmp_path / "config.json").write_text(json.dumps(config))
    weights = dict(tree_flatten(model.parameters()))
    weights.pop(next(iter(weights)))
    save_safetensors(str(tmp_path / "model.safetensors"), weights)
    with pytest.raises(ValueError):
        load_model(tmp_path)
