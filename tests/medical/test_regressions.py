"""Medical path regressions and parity checks against official Transformers.

These are synthetic engineering tests, not medical task accuracy benchmarks.
Weights are seeded random tiny models and inputs are synthetic, so agreement
here does not show agreement for the full checkpoints. Run through
``make test``, which sets MLX_ENABLE_TF32=0 and uses the ``med-vlm-mlx`` environment.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, cast

from numpy.typing import NDArray
from transformers.models.gemma3 import modeling_gemma3 as gemma_reference
from typing_extensions import TypedDict

from mlx_vlm.contracts import (
    FloatArray,
    NumericArray,
    array_int_list,
)
from mlx_vlm.models.protocols import require_generation_model, require_language_model

os.environ["MLX_ENABLE_TF32"] = "0"

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.gemma3.configuration_gemma3 import (
    Gemma3Config,
    Gemma3TextConfig,
)
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
    get_block_sequence_ids_for_mask,
)
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig

from mlx_vlm.generate.ar import (
    GenerationBatch,
    PromptProcessingBatch,
    generate_step,
)
from mlx_vlm.models.gemma3 import Model, ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.gemma3.attention import image_attention_mask
from mlx_vlm.models.gemma3.language import Gemma3Model


class _VisionMasks(Protocol):
    def __call__(
        self,
        *,
        config: object,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: None,
        position_ids: torch.Tensor | None,
        block_sequence_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]: ...


# Installed Transformers returns the two mask kinds but annotates only bare dict.
create_masks_for_vision_model = cast(
    _VisionMasks, getattr(gemma_reference, "create_masks_for_vision_model")
)


CONTRACT = json.loads(
    (Path(__file__).parents[1] / "fixtures/medical-fp32.json").read_text()
)


@pytest.fixture(autouse=True)
def metal_backend() -> None:
    if not mx.metal.is_available():
        raise RuntimeError("This review requires the MLX Metal backend.")
    mx.set_default_device(mx.gpu)


@dataclass(frozen=True)
class GemmaPair:
    model: Model
    reference: Gemma3ForCausalLM


class TinyTextFields(TypedDict):
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    sliding_window: int
    query_pre_attn_scalar: int
    max_position_embeddings: int


def make_pair(sliding_window: int = 32) -> GemmaPair:
    mx.random.seed(17)
    cast(Callable[[int], torch.Generator], getattr(torch, "manual_seed"))(17)
    shared = TinyTextFields(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=6,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        sliding_window=sliding_window,
        query_pre_attn_scalar=8,
        max_position_embeddings=128,
    )
    reference = Gemma3ForCausalLM(Gemma3TextConfig(**shared)).eval()
    model = Model(
        ModelConfig(
            text_config=TextConfig(
                model_type="gemma3", mm_tokens_per_image=4, **shared
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
            model_type="gemma3",
            image_token_index=63,
        )
    )
    model.language_model.load_weights(
        [
            (name, mx.array(value.numpy()))
            for name, value in reference.state_dict().items()
        ],
        strict=True,
    )
    model.eval()
    return GemmaPair(model, reference)


@pytest.fixture
def pair() -> GemmaPair:
    return make_pair()


def host(array: object) -> NumericArray:
    return cast(NumericArray, np.array(array))


def torch_ids(array: mx.array) -> torch.Tensor:
    return torch.tensor(host(array), dtype=torch.int64)


def greedy_steps(
    pair: GemmaPair,
    ids: mx.array,
    pixels: mx.array | None,
    mask: mx.array | None,
    max_tokens: int = 4,
    prefill_step_size: int | None = None,
) -> list[tuple[int | mx.array | None, mx.array | None]]:
    return list(
        generate_step(
            ids,
            require_generation_model(pair.model),
            pixels,
            mask,
            max_tokens=max_tokens,
            temperature=0,
            prefill_step_size=prefill_step_size,
        )
    )


def image_inputs() -> tuple[mx.array, mx.array, mx.array]:
    ids = mx.array([[2, 63, 63, 63, 63, 7, 9]])
    pixels = mx.reshape(mx.arange(48, dtype=mx.float32), (1, 3, 4, 4)) / 48
    return ids, pixels, mx.ones(ids.shape, dtype=mx.bool_)


def reference_image_logits(pair: GemmaPair) -> tuple[FloatArray, mx.array]:
    ids, pixels, mask = image_inputs()
    features = pair.model.get_input_embeddings(ids, pixels, mask)
    return (
        reference_logits(pair, ids, features.inputs_embeds, mask),
        features.inputs_embeds,
    )


def reference_logits(
    pair: GemmaPair,
    ids: mx.array,
    unscaled_embeds: mx.array,
    mask: mx.array,
) -> FloatArray:
    scaler = mx.array(pair.model.config.text_config.hidden_size**0.5, mx.bfloat16)
    embeds = torch.tensor(host(unscaled_embeds) * scaler.item())
    token_types = torch_ids(ids == 63)
    masks = create_masks_for_vision_model(
        config=pair.reference.config,
        inputs_embeds=embeds,
        attention_mask=torch_ids(mask),
        past_key_values=None,
        position_ids=torch.arange(ids.shape[1]).reshape(1, -1),
        block_sequence_ids=get_block_sequence_ids_for_mask(token_types),
    )
    with torch.no_grad():
        expected = pair.reference(
            inputs_embeds=embeds,
            attention_mask=masks,
            use_cache=False,
        ).logits.numpy()
    return expected


def assert_close(
    actual: NumericArray, expected: NumericArray, kind: str = "logits"
) -> None:
    error = float(np.max(np.abs(actual - expected)))
    print(f"max_absolute_error={error:.9g}")
    budget = CONTRACT["arrays"][kind]
    assert np.all(np.isfinite(actual))
    assert np.all(np.isfinite(expected))
    np.testing.assert_allclose(
        actual, expected, atol=budget["atol"], rtol=budget["rtol"]
    )


def test_control_text_matches_transformers(pair: GemmaPair) -> None:
    ids = mx.array([[2, 4, 6, 8, 10, 12, 14]])
    actual = host(pair.model.language_model(ids).logits)
    with torch.no_grad():
        expected = pair.reference(
            input_ids=torch_ids(ids),
            use_cache=False,
        ).logits.numpy()
    assert_close(actual, expected)


def test_control_explicit_image_mask_matches_transformers(pair: GemmaPair) -> None:
    expected, embeds = reference_image_logits(pair)
    ids, _, _ = image_inputs()
    index = mx.arange(ids.shape[1])
    causal = index[:, None] >= index[None, :]
    image = ids[0] == 63
    correct_mask = (causal | (image[:, None] & image[None, :]))[None, None]
    actual = cast(
        NumericArray,
        np.array(
            pair.model.language_model(
                ids,
                inputs_embeds=embeds,
                mask=correct_mask,
            ).logits
        ),
    )
    assert_close(actual, expected)


def test_gemma_public_generation_matches_image_attention_reference(
    pair: GemmaPair,
) -> None:
    expected, _ = reference_image_logits(pair)
    reference_logprobs = torch.log_softmax(
        torch.tensor(expected[0, -1]), dim=-1
    ).numpy()
    ids, pixels, mask = image_inputs()
    steps = greedy_steps(pair, ids, pixels, mask, max_tokens=1)
    assert len(steps) == 1
    assert_close(host(steps[0][1]), reference_logprobs)


def test_medgemma_published_config_preserves_image_feature_scale() -> None:
    raw = json.loads(
        (Path(__file__).parents[1] / "fixtures/medgemma-1.5-4b.json").read_text()
    )
    config = ModelConfig.from_dict(raw)
    text = TextConfig.from_dict(
        raw["text_config"]
        | {
            "num_hidden_layers": 0,
            "vocab_size": 2,
        }
    )
    language = Gemma3Model(text)
    language["norm"] = (
        nn.Identity()
    )  # Observe the exact input scale, before any decoder.
    ids = mx.array([[config.image_token_index]])
    original = mx.ones((1, 1, text.hidden_size))
    embeddings, mask = Model.prepare_inputs_for_multimodal(
        config.image_token_index,
        original,
        mx.zeros_like(original),
        ids,
        mx.ones(ids.shape, mx.bool_),
    )
    actual = language(ids, inputs_embeds=embeddings, mask=mask)
    print(
        f"text_hidden_size={text.hidden_size}; "
        f"image_feature_multiplier={actual[0, 0, 0].item():.9g}"
    )
    # The Transformers multimodal model inserts projected image features directly.
    assert_close(host(actual), host(original))


@pytest.mark.parametrize("window", [4, 32])
def test_gemma_multiple_images_with_padding_matches_reference(window: int) -> None:
    pair = make_pair(window)
    ids = mx.array(
        [
            [0, 0, 2, 63, 63, 63, 63, 7, 63, 63, 63, 63, 9],
            [0, 2, 63, 63, 63, 63, 8, 63, 63, 63, 63, 9, 10],
        ]
    )
    pixels = mx.reshape(mx.arange(192, dtype=mx.float32), (4, 3, 4, 4)) / 192
    mask = ids != 0
    features = pair.model.get_input_embeddings(ids, pixels, mask)
    expected = reference_logits(pair, ids, features.inputs_embeds, mask)
    actual = host(pair.model(ids, pixels, mask=mask).logits)
    # Padding-query values are not consumed; all valid token logits are compared.
    assert_close(
        actual[cast(NDArray[np.bool_], np.array(mask))],
        expected[cast(NDArray[np.bool_], np.array(mask))],
    )


@pytest.mark.parametrize("left_padding", [0, 2])
def test_gemma_cached_decode_matches_reference_after_rotation(
    left_padding: int,
) -> None:
    pair = make_pair(sliding_window=4)
    ids, pixels, _ = image_inputs()
    ids = mx.pad(ids, [(0, 0), (left_padding, 0)])
    mask = ids != 0
    features = pair.model.get_input_embeddings(ids, pixels, mask)
    cache = pair.model.language_model.make_cache()
    actual = pair.model.language_model(
        ids,
        inputs_embeds=features.inputs_embeds,
        cache=cache,
        attention_mask_4d=features.attention_mask_4d,
    ).logits
    embeds = features.inputs_embeds
    assert_close(
        host(actual[:, -1:]),
        reference_logits(pair, ids, embeds, mask)[:, -1:],
    )
    for token in [11, 12, 13, 14, 15, 16, 17, 18, 19]:
        next_id = mx.array([[token]])
        actual = pair.model.language_model(
            next_id,
            cache=cache,
            attention_mask_4d=features.attention_mask_4d,
        ).logits
        ids = mx.concatenate([ids, next_id], axis=1)
        embeds = mx.concatenate(
            [embeds, pair.model.language_model.model.embed_tokens(next_id)], axis=1
        )
        mask = mx.concatenate([mask, mx.ones((1, 1), mx.bool_)], axis=1)
        assert_close(
            host(actual),
            reference_logits(pair, ids, embeds, mask)[:, -1:],
        )


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4])
@pytest.mark.parametrize("with_image", [False, True])
def test_gemma_public_chunk_policy_matches_whole_prefill(
    pair: GemmaPair,
    chunk_size: int,
    with_image: bool,
) -> None:
    ids, pixels, mask = image_inputs()
    if not with_image:
        ids = mx.array([[2, 3, 4, 5, 6, 7, 8]])
        pixels = None
    expected = greedy_steps(pair, ids, pixels, mask)
    actual = greedy_steps(pair, ids, pixels, mask, prefill_step_size=chunk_size)
    assert pair.model.chunked_prefill_policy(input_ids=ids) is not with_image
    for (token, logprobs), (ref_token, ref_logprobs) in zip(
        actual, expected, strict=True
    ):
        assert token == ref_token
        assert_close(
            host(logprobs),
            host(ref_logprobs),
        )


@pytest.mark.parametrize("chunk_size", [1, 2, 4])
@pytest.mark.parametrize("with_image", [False, True])
def test_gemma_decoder_batch_matches_public_generation(
    pair: GemmaPair, chunk_size: int, with_image: bool
) -> None:
    ids, pixels, mask = image_inputs()
    if not with_image:
        ids = mx.array([[2, 3, 4, 5, 6, 7, 8]])
        pixels = None
    expected = greedy_steps(pair, ids, pixels, mask)
    features = pair.model.get_input_embeddings(ids, pixels, mask)
    batch = PromptProcessingBatch(
        model=require_language_model(pair.model.language_model),
        uids=[0],
        input_ids=[array_int_list(ids[0])],
        max_tokens=[4],
        inputs_embeds=features.inputs_embeds,
        prompt_kwargs={"attention_mask_4d": features.attention_mask_4d},
        prefill_step_size=chunk_size,
    )
    assert (batch.prefill_step_size is None) is with_image
    while batch.needs_processing():
        batch.prompt_step()
    generated = batch.generate(
        sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
        stop_criteria=lambda _: False,
    )
    assert isinstance(generated, GenerationBatch)
    for ref_token, logprobs in expected:
        assert isinstance(ref_token, int)
        assert logprobs is not None
        expected_logprob = logprobs[ref_token].item()
        assert isinstance(expected_logprob, float)
        response = generated.next()[0]
        assert response.token == ref_token
        assert response.token_logprob == pytest.approx(expected_logprob, abs=2e-5)


@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize("window", [4, 32])
def test_gemma_padded_multirow_batch_matches_independent_generation(
    with_image: bool, window: int
) -> None:
    pair = make_pair(sliding_window=window)
    base_ids, pixels, _ = image_inputs()
    if not with_image:
        base_ids, pixels = mx.array([[2, 3, 4, 5, 6]]), None
    rows = [
        mx.concatenate([mx.array([[3] * extra], dtype=mx.int32), base_ids], axis=1)
        for extra in range(3)
    ]
    expected = [greedy_steps(pair, ids, pixels, mx.ones(ids.shape)) for ids in rows]
    width = max(ids.shape[1] for ids in rows)
    embeddings: list[mx.array] = []
    padded_ids: list[mx.array] = []
    padding_masks: list[mx.array] = []
    for ids in rows:
        padding = width - ids.shape[1]
        features = pair.model.get_input_embeddings(ids, pixels, mx.ones(ids.shape))
        embeddings.append(
            mx.pad(features.inputs_embeds, [(0, 0), (padding, 0), (0, 0)])
        )
        padded_ids.append(mx.pad(ids, [(0, 0), (padding, 0)]))
        padding_masks.append(mx.pad(mx.ones(ids.shape), [(0, 0), (padding, 0)]))
    batch = PromptProcessingBatch(
        model=require_language_model(pair.model.language_model),
        uids=list(range(3)),
        input_ids=[array_int_list(ids[0]) for ids in rows],
        max_tokens=[4] * 3,
        inputs_embeds=mx.concatenate(embeddings),
        prompt_kwargs={
            "attention_mask_4d": image_attention_mask(
                mx.concatenate(padded_ids), mx.concatenate(padding_masks), 63
            )
        },
        prefill_step_size=2,
    )
    while batch.needs_processing():
        batch.prompt_step()
    generated = batch.generate(
        sampler=lambda logits: mx.argmax(logits, axis=-1), stop_criteria=lambda _: False
    )
    for step in range(4):
        responses = generated.next()
        assert len(responses) == 3
        for response in responses:
            token, logprobs = expected[response.uid][step]
            assert isinstance(token, int) and logprobs is not None
            assert response.token == token
            assert response.token_logprob == pytest.approx(
                logprobs[token].item(), abs=2e-5
            )


def test_gemma_public_generation_preserves_padding_during_decode(
    pair: GemmaPair,
) -> None:
    ids, pixels, _ = image_inputs()
    ids = mx.pad(ids, [(0, 0), (2, 0)])
    mask = ids != 0
    features = pair.model.get_input_embeddings(ids, pixels, mask)
    embeds = features.inputs_embeds
    steps = greedy_steps(pair, ids, pixels, mask, prefill_step_size=2)
    for token, logprobs in steps:
        logits = reference_logits(pair, ids, embeds, mask)[0, -1]
        expected = torch.log_softmax(torch.tensor(logits), dim=-1).numpy()
        assert_close(host(logprobs), expected)
        next_id = mx.array([[token]])
        ids = mx.concatenate([ids, next_id], axis=1)
        embeds = mx.concatenate(
            [embeds, pair.model.language_model.model.embed_tokens(next_id)], axis=1
        )
        mask = mx.concatenate([mask, mx.ones((1, 1), mx.bool_)], axis=1)


def test_gemma_complete_vision_to_logits_matches_transformers(pair: GemmaPair) -> None:
    vision_config = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=4,
        patch_size=2,
    )
    # from_dict retained this legacy setting as an extra config attribute.
    setattr(vision_config, "vision_use_head", False)
    reference = Gemma3ForConditionalGeneration(
        Gemma3Config(
            text_config=pair.reference.config,
            vision_config=vision_config,
            mm_tokens_per_image=4,
            image_token_index=63,
            boi_token_index=60,
            eoi_token_index=61,
        )
    ).eval()
    # Transformers zeros this projector at initialization. Nonzero weights are
    # essential: otherwise a vision-to-logits comparison cannot detect bad pixels.
    torch.nn.init.normal_(
        reference.model.multi_modal_projector.mm_input_projection_weight,
        mean=0.0,
        std=0.02,
    )
    weights: list[tuple[str, mx.array]] = []
    for name, value in reference.state_dict().items():
        name = name.removeprefix("model.")
        if name.startswith("language_model."):
            name = name.replace("language_model.", "language_model.model.", 1)
        elif name.startswith("vision_tower."):
            name = name.replace("vision_tower.", "vision_tower.vision_model.", 1)
        elif name == "lm_head.weight":
            name = "language_model.lm_head.weight"
        array = mx.array(value.numpy())
        if name.endswith("patch_embedding.weight"):
            array = mx.transpose(array, (0, 2, 3, 1))
        weights.append((name, array))
    pair.model.load_weights(weights, strict=True)
    ids, pixels, mask = image_inputs()
    with torch.no_grad():
        expected = reference(
            input_ids=torch_ids(ids),
            pixel_values=torch.tensor(host(pixels)),
            attention_mask=torch_ids(mask),
            token_type_ids=torch_ids(ids == 63),
            use_cache=False,
        ).logits.numpy()
    actual = host(pair.model(ids, pixels, mask=mask).logits)
    assert_close(actual, expected)
    changed = host(pair.model(ids, -pixels, mask=mask).logits)
    assert np.max(np.abs(actual - changed)) > 1e-4, "Pixels must affect logits"
    vision, _, _ = pair.model.vision_tower(
        mx.transpose(pixels, (0, 2, 3, 1)), output_hidden_states=True
    )
    projected = host(pair.model.multi_modal_projector(vision))
    assert np.max(np.abs(projected)) > 1e-3, "Projector fixture must be nonzero"
    with torch.no_grad():
        # This FP32 fixture uses the reference's return_dict image-feature path.
        image_features = cast(
            BaseModelOutputWithPooling,
            reference.get_image_features(
                cast(
                    torch.FloatTensor,
                    torch.tensor(host(pixels)),
                )
            ),
        )
        assert image_features.pooler_output is not None
        reference_features = image_features.pooler_output.numpy()
    assert_close(projected, reference_features, "image_features")
    torch_masks = create_masks_for_vision_model(
        config=pair.reference.config,
        inputs_embeds=torch.zeros((1, ids.shape[1], 16)),
        attention_mask=torch_ids(mask),
        past_key_values=None,
        position_ids=torch.arange(ids.shape[1]).reshape(1, -1),
        block_sequence_ids=get_block_sequence_ids_for_mask(
            torch.tensor(host(ids == 63))
        ),
    )
    actual_mask = host(image_attention_mask(ids, mask, 63))
    full_mask = torch_masks["full_attention"]
    assert full_mask is not None
    reference_mask = cast(NumericArray, full_mask.numpy())
    np.testing.assert_array_equal(actual_mask, reference_mask)
