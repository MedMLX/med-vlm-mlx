"""Medical selection and request boundaries without downloading checkpoints."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from mlx_vlm.contracts import JSONObject
from mlx_vlm.convert import configure_parser
from mlx_vlm.generate.cli import parse_arguments
from mlx_vlm.prompt_utils import apply_chat_template, get_message_json
from mlx_vlm.server.app import app
from mlx_vlm.server.model_discovery import is_model_directory
from mlx_vlm.server.runtime_config import RuntimeConfig
from mlx_vlm.utils import get_model_and_args, load, load_model, load_processor


@pytest.mark.parametrize("model_type", ["gemma3"])
def test_supported_medical_architecture_loads(model_type: str) -> None:
    module, resolved = get_model_and_args({"model_type": model_type})
    assert resolved == model_type
    assert module.__name__ == f"mlx_vlm.models.{model_type}"


@given(st.text().filter(lambda text: text != "gemma3"))
@settings(max_examples=30)
def test_unmaintained_architecture_is_rejected(model_type: str) -> None:
    with pytest.raises(ValueError, match="Unsupported model type"):
        get_model_and_args({"model_type": model_type})
    with pytest.raises(ValueError, match="Unsupported medical prompt model"):
        apply_chat_template(None, {"model_type": model_type}, "Describe this image")


@pytest.mark.parametrize("model_type", ["flux", "qwen2_5_vl"])
def test_unsupported_checkpoint_is_rejected_before_weight_decoding(
    tmp_path: Path,
    model_type: str,
) -> None:
    import json

    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    (tmp_path / "model.safetensors").write_bytes(b"invalid weights")
    with pytest.raises(ValueError, match="Unsupported model type"):
        load_model(tmp_path, lazy=True)


def test_checkpoint_code_is_rejected_before_resolution() -> None:
    with pytest.raises(ValueError, match="Checkpoint Python code"):
        load("unresolved-checkpoint", trust_remote_code=True)
    with pytest.raises(ValueError, match="Checkpoint Python code"):
        load_model(Path("unresolved-checkpoint"), trust_remote_code=True)
    with pytest.raises(ValueError, match="Checkpoint Python code"):
        load_processor(Path("unresolved-checkpoint"), trust_remote_code=True)


def test_converter_rejects_checkpoint_code_flag() -> None:
    with pytest.raises(SystemExit) as error:
        configure_parser().parse_args(
            ["--hf-path", "unresolved-checkpoint", "--trust-remote-code"]
        )
    assert error.value.code == 2


@pytest.mark.parametrize("model_type", ["gemma3"])
@given(st.lists(st.text(max_size=40), min_size=3, max_size=3))
def test_ordered_images_keep_their_original_positions(
    model_type: str, text: list[str]
) -> None:
    parts: list[JSONObject] = [
        {"type": "text", "text": text[0]},
        {"type": "image"},
        {"type": "text", "text": text[1]},
        {"type": "image"},
        {"type": "text", "text": text[2]},
    ]
    message = get_message_json(model_type, parts, num_images=2)
    assert isinstance(message, dict)
    assert message["content"] == "<start_of_image>".join(text)


def test_prompt_formatter_rejects_unsupported_inputs() -> None:
    with pytest.raises(ValueError, match="Audio inputs"):
        get_message_json("gemma3", [{"type": "input_audio", "input_audio": {}}])
    with pytest.raises(ValueError, match="Video inputs"):
        get_message_json("gemma3", [{"type": "video", "video": "scan.mp4"}])


def test_runtime_knobs_reload_the_medical_vision_cache() -> None:
    config = RuntimeConfig()
    before = config.fingerprint(["text_generation"])
    applied, rejected = config.apply_changes({"vision_cache_size": 8})
    assert not rejected
    assert config.reload_kinds(applied) == {"text_generation"}
    assert config.fingerprint(["text_generation"]) != before
    _, rejected = config.apply_changes({"spec_draft_model": "other-model"})
    assert rejected == [{"name": "spec_draft_model", "reason": "unknown knob"}]


def test_checkpoint_python_is_never_executed(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    (tmp_path / "custom.py").write_text(
        "from pathlib import Path\n" + f"Path({str(marker)!r}).touch()\n"
    )
    with pytest.raises(ValueError, match="model_file"):
        get_model_and_args(
            {"model_type": "gemma3", "model_file": "custom.py"}, tmp_path
        )
    assert not marker.exists()


@given(st.integers(min_value=1, max_value=8192), st.text(min_size=1))
@settings(max_examples=30)
def test_cli_preserves_prompt_and_token_count(tokens: int, prompt: str) -> None:
    args = parse_arguments(["--max-tokens", str(tokens), f"--prompt={prompt}"])
    assert args.max_tokens == tokens
    assert args.prompt == [prompt]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--output-modality", "image"],
        ["--audio", "input.wav"],
        ["--max-tokens", "0"],
        ["--temperature", "nan"],
        ["--video", "scan.mp4"],
    ],
)
def test_cli_rejects_invalid_or_unrelated_options(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        parse_arguments(["--prompt", "describe", *arguments])
    assert error.value.code == 2


@pytest.mark.parametrize("model_type", ["gemma3", "qwen2_5_vl", "sam3", "flux"])
def test_discovery_lists_only_medical_architectures(
    tmp_path: Path, model_type: str
) -> None:
    import json

    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    (tmp_path / "model.safetensors").write_bytes(b"metadata-only fixture")
    assert is_model_directory(tmp_path) == (model_type == "gemma3")


def test_http_health_and_removed_modalities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLX_VLM_PRELOAD_MODEL", raising=False)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        for route in (
            "/v1/images/generations",
            "/v1/audio/speech",
            "/v1/embeddings",
            "/v1/rerank",
        ):
            assert client.post(route, json={}).status_code == 404
