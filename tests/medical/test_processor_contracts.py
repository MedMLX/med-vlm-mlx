"""Real local-tokenizer processor contracts; no checkpoint or network access."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image
from tokenizers import Tokenizer
from transformers import AutoImageProcessor, PreTrainedTokenizerFast
from transformers.models.gemma3.image_processing_pil_gemma3 import (
    Gemma3ImageProcessorPil,
)

from mlx_vlm.contracts import (
    parse_int,
    require_processed_inputs,
)
from mlx_vlm.models.gemma3.processing_gemma3 import Gemma3Processor


class MedicalTokenizer(PreTrainedTokenizerFast):
    image_token = "<image>"
    video_token = "<video>"
    boi_token = "<begin_image>"
    eoi_token = "<end_image>"
    image_token_id = 2
    video_token_id = 3


def local_tokenizer(with_bos: bool = False) -> MedicalTokenizer:
    # Serialized by the installed WordLevel tokenizer; no pretrained assets are needed.
    serialized = (
        '{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,'
        '"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,'
        '"model":{"type":"WordLevel","vocab":{"[UNK]":0,"[PAD]":1,"<image>":2,"<video>":3,'
        '"<begin_image>":4,"<end_image>":5,"Describe":6,"scan":7,"<|placeholder|>":8},"unk_token":"[UNK]"}}'
    )
    if with_bos:
        # Serialized by tokenizers.TemplateProcessing for one [PAD] BOS marker.
        postprocessor = '{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"[PAD]","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{"[PAD]":{"id":"[PAD]","ids":[1],"tokens":["[PAD]"]}}}'
        serialized = serialized.replace(
            '"post_processor":null', '"post_processor":' + postprocessor
        )
    backend = Tokenizer.from_str(serialized)
    return MedicalTokenizer(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=[
            "<image>",
            "<video>",
            "<begin_image>",
            "<end_image>",
            "<|placeholder|>",
        ],
    )


def test_gemma_real_image_processor_preserves_one_text_input() -> None:
    processor = Gemma3Processor(
        Gemma3ImageProcessorPil(
            size={"height": 32, "width": 32}, do_pan_and_scan=False
        ),
        local_tokenizer(),
        image_seq_length=2,
    )
    text_only = require_processed_inputs(
        getattr(processor(text="Describe scan"), "data")
    )
    input_ids = text_only["input_ids"]
    assert isinstance(input_ids, mx.array) and input_ids.shape[0] == 1
    assert input_ids.shape[1] == 2
    image = Image.new("RGB", (32, 32), "red")
    processed = require_processed_inputs(
        getattr(processor(images=image, text="Describe <begin_image>"), "data")
    )
    ids = processed["input_ids"]
    assert isinstance(ids, mx.array)
    assert parse_int(mx.sum(ids == 2).item()) == 2
    assert isinstance(processed["pixel_values"], mx.array)


def test_loaded_gemma_template_does_not_duplicate_bos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mlx_vlm.utils import prepare_inputs, should_add_special_tokens

    tokenizer = local_tokenizer(with_bos=True)
    tokenizer.chat_template = "[PAD] Describe"
    image_processor = Gemma3ImageProcessorPil(size={"height": 32, "width": 32})

    def load_local_tokenizer(
        path: str | Path, kwargs: dict[str, object]
    ) -> MedicalTokenizer:
        return tokenizer

    def load_local_images(*args: object, **kwargs: object) -> Gemma3ImageProcessorPil:
        return image_processor

    monkeypatch.setattr(
        "mlx_vlm.models.gemma3.processing_gemma3.load_tokenizer", load_local_tokenizer
    )
    monkeypatch.setattr(AutoImageProcessor, "from_pretrained", load_local_images)
    processor = Gemma3Processor.from_pretrained(tmp_path)
    add_special = should_add_special_tokens("gemma3", processor)
    prepared = prepare_inputs(
        processor, prompts="[PAD] Describe", add_special_tokens=add_special
    )
    ids = prepared["input_ids"]
    assert isinstance(ids, mx.array)
    assert ids.tolist() == [[1, 6]]


def test_processed_input_boundary_preserves_tensors_and_rejects_invalid_leaves() -> (
    None
):
    tensor = mx.ones((1, 2))
    numpy_tensor = np.ones((1, 2), dtype=np.float32)
    payload = {"mlx": tensor, "numpy": numpy_tensor}
    validated = require_processed_inputs(payload)
    assert validated["mlx"] is tensor and validated["numpy"] is numpy_tensor
    homogeneous = {
        "crops": [tensor, numpy_tensor],
        "batches": [[tensor], [numpy_tensor]],
    }
    validated_batches = require_processed_inputs(homogeneous)
    assert validated_batches["crops"] is homogeneous["crops"]
    assert validated_batches["batches"] is homogeneous["batches"]
    for mixed in ([tensor, [tensor]], [[tensor], tensor]):
        with pytest.raises(TypeError, match="Unsupported processor"):
            require_processed_inputs({"mixed": mixed})
    with pytest.raises(TypeError, match="Unsupported processor"):
        require_processed_inputs({"invalid": object()})
