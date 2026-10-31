"""Local converter behavior with real native layers and processor sidecars."""

import importlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, ProcessorMixin
from transformers.models.gemma3.image_processing_pil_gemma3 import (
    Gemma3ImageProcessorPil,
)

from mlx_vlm.contracts import JSONObject
from mlx_vlm.convert import convert
from mlx_vlm.models.base import LanguageModelOutput
from mlx_vlm.models.gemma3.processing_gemma3 import Gemma3Processor


class TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 32)
        self.layers = [nn.Linear(32, 32)]

    def __call__(self, inputs: mx.array) -> LanguageModelOutput:
        return LanguageModelOutput(self.layers[0](self.embed(inputs)))


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = TinyDecoder()


class LocalTokenizer(PreTrainedTokenizerFast):
    image_token = "<image>"
    image_token_id = 2
    boi_token = "<begin_image>"
    eoi_token = "<end_image>"


def local_processor() -> Gemma3Processor:
    tokenizer = LocalTokenizer(
        tokenizer_object=Tokenizer.from_str(
            '{"version":"1.0","model":{"type":"WordLevel","vocab":{"[UNK]":0,"[PAD]":1},"unk_token":"[UNK]"},"pre_tokenizer":{"type":"Whitespace"}}'
        ),
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    return Gemma3Processor(Gemma3ImageProcessorPil(), tokenizer)


def test_convert_local_quantize_cast_and_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors.index.json").write_text('{"obsolete":true}')
    (source / "helper.py").write_text("# preserved processor helper\n")
    config: JSONObject = {"torch_dtype": "float16"}
    model = TinyModel()
    processor = local_processor()

    def fetch(
        path: Path, lazy: bool = False, **kwargs: object
    ) -> tuple[nn.Module, JSONObject, ProcessorMixin]:
        assert path == source and lazy and kwargs["trust_remote_code"] is False
        return model, config, processor

    monkeypatch.setattr(
        importlib.import_module("mlx_vlm.convert"), "fetch_from_hub", fetch
    )
    output = tmp_path / "output"
    convert(str(source), str(output), quantize=True, q_group_size=32)
    saved = json.loads((output / "config.json").read_text())
    assert saved["quantization"]["bits"] == 4
    assert (output / "helper.py").read_text() == "# preserved processor helper\n"
    assert (output / "processor_config.json").is_file()
    assert "obsolete" not in (output / "model.safetensors.index.json").read_text()
    weights = mx.load(output / "model.safetensors")
    assert isinstance(weights, dict)
    assert weights["language_model.embed.scales"].dtype == mx.float16
