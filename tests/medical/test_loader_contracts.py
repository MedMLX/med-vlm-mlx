"""Local checkpoint and streaming contracts without pretrained downloads."""

import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx
import pytest
from huggingface_hub import HfApi
from huggingface_hub.errors import LocalEntryNotFoundError
from hypothesis import given, settings
from hypothesis import strategies as st
from mlx.utils import tree_flatten

from mlx_vlm.contracts import JSONObject, JSONValue, evaluate
from mlx_vlm.models.gemma3 import Model as GemmaModel
from mlx_vlm.models.gemma3 import ModelConfig as GemmaConfig
from mlx_vlm.models.gemma3 import TextConfig as GemmaTextConfig
from mlx_vlm.tokenizer_utils import (
    ServerTokenStreamer,
    SPMStreamingDetokenizer,
    load_detokenizer_factory,
)
from mlx_vlm.utils import load, load_config, load_model, load_processor, save_weights


@pytest.mark.parametrize("loader", ["model", "config", "processor"])
def test_remote_checkpoint_respects_offline_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loader: str
) -> None:
    """A missing snapshot in an isolated cache must never contact the Hub."""

    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("offline loading attempted a Hub request")

    monkeypatch.setattr(HfApi, "repo_info", reject_network)
    with pytest.raises(LocalEntryNotFoundError):
        if loader == "model":
            load("test-offline/missing", cache_dir=tmp_path, local_files_only=True)
        elif loader == "config":
            load_config(
                "test-offline/missing", cache_dir=tmp_path, local_files_only=True
            )
        else:
            load_processor(
                "test-offline/missing", cache_dir=tmp_path, local_files_only=True
            )


def test_load_config_finds_snapshot_in_requested_cache(tmp_path: Path) -> None:
    """A cached revision can be used offline without the default Hub cache."""
    revision = "a" * 40
    snapshot = tmp_path / "models--test-offline--cached" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type": "gemma3"}')

    assert load_config(
        "test-offline/cached",
        revision=revision,
        cache_dir=tmp_path,
        local_files_only=True,
    ) == {"model_type": "gemma3"}


def test_remote_checkpoint_uses_supplied_hub_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authentication reaches the Hub client used for snapshot discovery."""

    class NetworkStopped(Exception):
        pass

    def capture_token(api: HfApi, *args: object, **kwargs: object) -> None:
        assert api.token == "test-token"
        raise NetworkStopped

    monkeypatch.setattr(HfApi, "repo_info", capture_token)
    with pytest.raises(NetworkStopped):
        load_config("test-offline/authenticated", token="test-token")


def _checkpoint_config(architecture: str) -> JSONObject:
    text: JSONObject = {
        "model_type": architecture,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6,
        "vocab_size": 64,
    }
    text.update(head_dim=8, mm_tokens_per_image=4, query_pre_attn_scalar=8)
    vision: JSONObject = {
        "model_type": architecture,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "image_size": 4,
        "patch_size": 2,
    }
    return {
        "model_type": architecture,
        "text_config": text,
        "vision_config": vision,
        "image_token_index": 63,
    }


@pytest.mark.parametrize("token_count", [0, -1, 2, 5])
def test_gemma_rejects_invalid_image_token_grid(token_count: int) -> None:
    with pytest.raises(ValueError, match="mm_tokens_per_image"):
        GemmaTextConfig(
            model_type="gemma3",
            hidden_size=16,
            num_hidden_layers=1,
            intermediate_size=32,
            mm_tokens_per_image=token_count,
        )

    config = _checkpoint_config("gemma3")
    text_config = config["text_config"]
    assert isinstance(text_config, dict)
    text_config["mm_tokens_per_image"] = token_count
    with pytest.raises(ValueError, match="mm_tokens_per_image"):
        GemmaConfig.from_dict(config)


@pytest.mark.parametrize("token_count", [1, 4, 256])
def test_gemma_accepts_square_image_token_grids(token_count: int) -> None:
    config = _checkpoint_config("gemma3")
    text_config = config["text_config"]
    assert isinstance(text_config, dict)
    text_config["mm_tokens_per_image"] = token_count
    assert GemmaConfig.from_dict(config).text_config.mm_tokens_per_image == token_count


def test_gemma_uses_top_level_image_token_count() -> None:
    config = _checkpoint_config("gemma3")
    text_config = config["text_config"]
    assert isinstance(text_config, dict)
    del text_config["mm_tokens_per_image"]
    assert GemmaConfig.from_dict(config).text_config.mm_tokens_per_image == 256
    config["mm_tokens_per_image"] = 4
    assert GemmaConfig.from_dict(config).text_config.mm_tokens_per_image == 4
    assert "mm_tokens_per_image" not in text_config
    config["mm_tokens_per_image"] = 5
    with pytest.raises(ValueError, match="mm_tokens_per_image"):
        GemmaConfig.from_dict(config)


def test_gemma_rejects_conflicting_image_token_counts() -> None:
    config = _checkpoint_config("gemma3")
    config["mm_tokens_per_image"] = 9
    with pytest.raises(ValueError, match="mm_tokens_per_image"):
        GemmaConfig.from_dict(config)
    config["mm_tokens_per_image"] = 4
    assert GemmaConfig.from_dict(config).text_config.mm_tokens_per_image == 4


@pytest.mark.parametrize("value", [True, 4.0, "4"])
@pytest.mark.parametrize("location", ["top_level", "nested"])
def test_gemma_rejects_coercible_noninteger_image_token_count(
    value: bool | float | str, location: str
) -> None:
    config = _checkpoint_config("gemma3")
    text_config = config["text_config"]
    assert isinstance(text_config, dict)
    if location == "top_level":
        del text_config["mm_tokens_per_image"]
        config["mm_tokens_per_image"] = value
    else:
        text_config["mm_tokens_per_image"] = value
    with pytest.raises(ValueError, match="mm_tokens_per_image"):
        GemmaConfig.from_dict(config)


@pytest.mark.parametrize("architecture", ["gemma3"])
@pytest.mark.parametrize("donate_weights", [False, True])
def test_saved_native_checkpoint_preserves_parameters_and_logits(
    tmp_path: Path, architecture: str, donate_weights: bool
) -> None:
    """Exercise native dispatch, nested configs and sanitizer order on saved weights."""
    config = _checkpoint_config(architecture)
    model = GemmaModel(GemmaConfig.from_dict(config))
    model.eval()
    weights = dict(tree_flatten(model.parameters()))
    ids = mx.array([[2, 7, 9]])
    expected = model.language_model(ids).logits
    evaluate(expected)
    save_weights(tmp_path, model, donate_weights=donate_weights)
    if donate_weights:
        assert all(value.size == 0 for _, value in tree_flatten(model.parameters()))
    (tmp_path / "config.json").write_text(json.dumps(config))
    restored = load_model(tmp_path)
    assert type(restored) is type(model)
    assert asdict(restored.config) == asdict(model.config)
    recovered = dict(tree_flatten(restored.parameters()))
    assert recovered.keys() == weights.keys()
    assert all(
        bool(mx.array_equal(recovered[key], value).item())
        for key, value in weights.items()
    )
    actual = restored.language_model(ids).logits
    evaluate(expected, actual)
    assert bool(mx.array_equal(expected, actual).item())


class _ByteTokenizer:
    vocab = {"▁": 0, "<0xE5>": 1, "<0xA4>": 2, "<0xA2>": 3, "▁ok": 4}

    def decode(self, token_ids: list[int]) -> str:
        raise ValueError(f"Cannot decode {token_ids}")


def test_streaming_spm_preserves_split_utf8_and_reports_decode_failure(
    tmp_path: Path,
) -> None:
    """Byte fallback groups survive token boundaries; genuine decode errors remain visible."""
    tokenizer = _ByteTokenizer()
    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "decoder": {
                    "type": "Sequence",
                    "decoders": [
                        {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                        {"type": "ByteFallback"},
                        {"type": "Fuse"},
                    ],
                }
            }
        )
    )
    codec = load_detokenizer_factory(tmp_path)(tokenizer)
    assert isinstance(codec, SPMStreamingDetokenizer)
    stream = ServerTokenStreamer(tokenizer, codec)
    output = "".join(stream.advance(token, None) for token in (1, 2, 3, 4))
    assert output + stream.finalize() == "夢 ok"
    with pytest.raises(ValueError, match="Cannot decode"):
        stream.advance(99, None)


class _MarkerTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return [] if text == "empty" else [len(text)]


def test_thinking_budget_rejects_missing_or_empty_encoded_markers() -> None:
    from mlx_vlm.utils import ThinkingBudgetCriteria

    tokenizer = _MarkerTokenizer()
    with pytest.raises(ValueError, match="explicit thinking_start_token"):
        ThinkingBudgetCriteria(tokenizer, 4)
    with pytest.raises(ValueError, match="at least one token"):
        ThinkingBudgetCriteria(tokenizer, 4, thinking_start_token="empty")
    with pytest.raises(ValueError, match="at least one token"):
        ThinkingBudgetCriteria(
            tokenizer, 4, thinking_start_token="<think>", thinking_end_token="empty"
        )
    criteria = ThinkingBudgetCriteria(tokenizer, 4, thinking_start_token="<think>")
    assert criteria.thinking_start_token_id == len("<think>")


class _MetadataTokenizer(_ByteTokenizer):
    def decode(self, token_ids: list[int]) -> str:
        tokens = list(self.vocab)
        return "".join(tokens[index] for index in token_ids)


@given(
    st.one_of(
        st.none(),
        st.integers(),
        st.text(),
        st.lists(st.integers()),
        st.dictionaries(st.text(), st.text()),
    )
)
@settings(max_examples=25)
def test_tokenizer_decoder_metadata_json_roundtrip(
    value: None | int | str | list[int] | dict[str, str],
) -> None:
    """Arbitrary JSON decoder descriptions select a codec without parser crashes."""
    decoder: JSONValue = (
        list[JSONValue](value)
        if isinstance(value, list)
        else dict[str, JSONValue](value) if isinstance(value, dict) else value
    )
    with TemporaryDirectory() as directory:
        path = Path(directory)
        metadata: JSONObject = {"decoder": decoder}
        (path / "tokenizer.json").write_text(json.dumps(metadata))
        first = load_detokenizer_factory(path)(_MetadataTokenizer())
        (path / "tokenizer.json").write_text(
            json.dumps(json.loads(json.dumps(metadata)))
        )
        second = load_detokenizer_factory(path)(_MetadataTokenizer())
        assert type(first) is type(second)
