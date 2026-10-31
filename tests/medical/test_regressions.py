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
    evaluate,
)
from mlx_vlm.models.protocols import require_generation_model, require_language_model

os.environ["MLX_ENABLE_TF32"] = "0"

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
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
    BatchGenerator,
    GenerationBatch,
    PromptProcessingBatch,
    generate_step,
)
from mlx_vlm.generate.types import StopCriteria
from mlx_vlm.models.cache import KVCache
from mlx_vlm.models.gemma3 import Model, ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.gemma3.attention import image_attention_mask
from mlx_vlm.models.gemma3.language import Gemma3Model
from mlx_vlm.models.vision_features import merge_vision_features
from mlx_vlm.utils import ThinkingBudgetCriteria


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
    (Path(__file__).parents[2] / "contracts/medical-fp32.json").read_text()
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
    embeds = torch.tensor(cast(NumericArray, np.array(unscaled_embeds)) * scaler.item())
    token_types = torch.tensor(
        cast(NumericArray, np.array(ids == 63)), dtype=torch.int64
    )
    masks = create_masks_for_vision_model(
        config=pair.reference.config,
        inputs_embeds=embeds,
        attention_mask=torch.tensor(
            cast(NumericArray, np.array(mask)), dtype=torch.int64
        ),
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
    actual = cast(NumericArray, np.array(pair.model.language_model(ids).logits))
    with torch.no_grad():
        expected = pair.reference(
            input_ids=torch.tensor(
                cast(NumericArray, np.array(ids)), dtype=torch.int64
            ),
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
    steps = list(
        generate_step(
            ids,
            require_generation_model(pair.model),
            pixels,
            mask,
            max_tokens=1,
            temperature=0,
            prefill_step_size=None,
        )
    )
    assert len(steps) == 1
    assert_close(cast(NumericArray, np.array(steps[0][1])), reference_logprobs)


def test_gemma_direct_forward_does_not_see_future_text(pair: GemmaPair) -> None:
    ids, pixels, mask = image_inputs()
    changed = mx.array([[2, 63, 63, 63, 63, 7, 30]])
    first = cast(
        NumericArray, np.array(pair.model(ids, pixels, mask=mask).logits[:, 0])
    )
    second = cast(
        NumericArray, np.array(pair.model(changed, pixels, mask=mask).logits[:, 0])
    )
    assert_close(first, second)


def test_medgemma_published_config_preserves_image_feature_scale() -> None:
    raw = json.loads(
        (Path(__file__).parent / "fixtures/medgemma-1.5-4b.json").read_text()
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
    assert_close(
        cast(NumericArray, np.array(actual)), cast(NumericArray, np.array(original))
    )


@pytest.mark.parametrize("difference", [-1, 1])
def test_decoder_rejects_wrong_cache_count_before_mutation(
    pair: GemmaPair, difference: int
) -> None:
    language = pair.model.language_model
    for decoder in (language, language.model):
        caches = [KVCache() for _ in range(len(language.model.layers) + difference)]
        with pytest.raises(ValueError, match="cache"):
            decoder(mx.array([[1, 2]]), cache=caches)
        assert all(cache.offset == 0 for cache in caches)


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
    actual = cast(NumericArray, np.array(pair.model(ids, pixels, mask=mask).logits))
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
        cast(NumericArray, np.array(actual[:, -1:])),
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
            cast(NumericArray, np.array(actual)),
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
    expected = list(
        generate_step(
            ids,
            require_generation_model(pair.model),
            pixels,
            mask,
            max_tokens=4,
            temperature=0,
            prefill_step_size=None,
        )
    )
    actual = list(
        generate_step(
            ids,
            require_generation_model(pair.model),
            pixels,
            mask,
            max_tokens=4,
            temperature=0,
            prefill_step_size=chunk_size,
        )
    )
    assert pair.model.chunked_prefill_policy(input_ids=ids) is not with_image
    for (token, logprobs), (ref_token, ref_logprobs) in zip(
        actual, expected, strict=True
    ):
        assert token == ref_token
        assert_close(
            cast(NumericArray, np.array(logprobs)),
            cast(NumericArray, np.array(ref_logprobs)),
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
    expected = list(
        generate_step(
            ids,
            require_generation_model(pair.model),
            pixels,
            mask,
            max_tokens=4,
            temperature=0,
            prefill_step_size=None,
        )
    )
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
    expected = [
        list(
            generate_step(
                ids,
                require_generation_model(pair.model),
                pixels,
                mx.ones(ids.shape),
                max_tokens=4,
                temperature=0,
                prefill_step_size=None,
            )
        )
        for ids in rows
    ]
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


@dataclass
class BatchTokenHooks:
    stopping_criteria: StopCriteria
    thinking_budget_criteria: ThinkingBudgetCriteria | None = None


def collect_batch_tokens(
    generator: BatchGenerator, cache_counts: list[int] | None = None
) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for _ in range(80):
        if not generator.has_work:
            return result
        prompts, responses = generator.next()
        if cache_counts is not None:
            cache_counts.extend(prompt.cached_tokens for prompt in prompts)
        for response in responses:
            if response.token is not None:
                result.setdefault(response.uid, []).append(response.token)
    raise AssertionError("Tiny batch did not finish within 80 steps")


def test_gemma_partial_image_prefill_rejected_before_cache_mutation(
    pair: GemmaPair,
) -> None:
    ids, pixels, mask = image_inputs()
    features = pair.model.get_input_embeddings(ids, pixels, mask)
    cache = pair.model.language_model.make_cache()
    with pytest.raises(ValueError, match="splits a bidirectional image block"):
        pair.model.language_model(
            ids[:, :3],
            inputs_embeds=features.inputs_embeds[:, :3],
            cache=cache,
            attention_mask_4d=features.attention_mask_4d,
        )
    assert all(entry.offset == 0 for entry in cache)


def test_gemma_does_not_mutate_supplied_embeddings(pair: GemmaPair) -> None:
    ids, pixels, mask = image_inputs()
    features = pair.model.get_input_embeddings(ids, pixels, mask)
    before = cast(NumericArray, np.array(features.inputs_embeds))
    result = pair.model.language_model(
        ids,
        inputs_embeds=features.inputs_embeds,
        attention_mask_4d=features.attention_mask_4d,
    )
    evaluate(result.logits)
    np.testing.assert_array_equal(
        cast(NumericArray, np.array(features.inputs_embeds)), before
    )


@settings(max_examples=40, deadline=None, derandomize=True)
@given(st.lists(st.integers(0, 3), min_size=1, max_size=24))
def test_image_mask_obeys_causality_and_block_boundaries(tokens: list[int]) -> None:
    ids = mx.array([tokens])
    actual = cast(
        NumericArray, np.array(image_attention_mask(ids, ids != 0, image_token_id=3))
    )[0, 0]
    for query, row in enumerate(actual):
        for key, visible in enumerate(row):
            same_image = all(
                value == 3 for value in tokens[min(query, key) : max(query, key) + 1]
            )
            assert visible == (tokens[key] != 0 and (key <= query or same_image))


@settings(max_examples=40, deadline=None, derandomize=True)
@given(st.lists(st.integers(0, 3), min_size=1, max_size=24))
def test_feature_merge_round_trips_placeholder_order(tokens: list[int]) -> None:
    ids = mx.array([tokens])
    count = tokens.count(3)
    original = mx.reshape(mx.arange(len(tokens) * 2, dtype=mx.float32), (1, -1, 2))
    features = mx.reshape(mx.arange(count * 2, dtype=mx.float32), (count, 2)) + 100
    actual = cast(
        NumericArray, np.array(merge_vision_features(3, features, original, ids))
    )[0]
    image = cast(NumericArray, np.array(tokens)) == 3
    np.testing.assert_array_equal(actual[image], cast(NumericArray, np.array(features)))
    np.testing.assert_array_equal(
        actual[~image], cast(NumericArray, np.array(original))[0, ~image]
    )
    with pytest.raises(ValueError, match="placeholder count"):
        merge_vision_features(3, mx.ones((count + 1, 2)), original, ids)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_vision_features_fail_fast(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        merge_vision_features(
            3, mx.array([[invalid, 1.0]]), mx.zeros((1, 1, 2)), mx.array([[3]])
        )


def test_gemma_public_generation_preserves_padding_during_decode(
    pair: GemmaPair,
) -> None:
    ids, pixels, _ = image_inputs()
    ids = mx.pad(ids, [(0, 0), (2, 0)])
    mask = ids != 0
    features = pair.model.get_input_embeddings(ids, pixels, mask)
    embeds = features.inputs_embeds
    steps = list(
        generate_step(
            ids,
            require_generation_model(pair.model),
            pixels,
            mask,
            max_tokens=4,
            prefill_step_size=2,
        )
    )
    for token, logprobs in steps:
        logits = reference_logits(pair, ids, embeds, mask)[0, -1]
        expected = torch.log_softmax(torch.tensor(logits), dim=-1).numpy()
        assert_close(cast(NumericArray, np.array(logprobs)), expected)
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
            input_ids=torch.tensor(
                cast(NumericArray, np.array(ids)), dtype=torch.int64
            ),
            pixel_values=torch.tensor(cast(NumericArray, np.array(pixels))),
            attention_mask=torch.tensor(
                cast(NumericArray, np.array(mask)), dtype=torch.int64
            ),
            token_type_ids=torch.tensor(
                cast(NumericArray, np.array(ids == 63)), dtype=torch.int64
            ),
            use_cache=False,
        ).logits.numpy()
    actual = cast(NumericArray, np.array(pair.model(ids, pixels, mask=mask).logits))
    assert_close(actual, expected)
    changed = cast(NumericArray, np.array(pair.model(ids, -pixels, mask=mask).logits))
    assert np.max(np.abs(actual - changed)) > 1e-4, "Pixels must affect logits"
    vision, _, _ = pair.model.vision_tower(
        mx.transpose(pixels, (0, 2, 3, 1)), output_hidden_states=True
    )
    projected = cast(NumericArray, np.array(pair.model.multi_modal_projector(vision)))
    assert np.max(np.abs(projected)) > 1e-3, "Projector fixture must be nonzero"
    with torch.no_grad():
        # This FP32 fixture uses the reference's return_dict image-feature path.
        image_features = cast(
            BaseModelOutputWithPooling,
            reference.get_image_features(
                cast(
                    torch.FloatTensor,
                    torch.tensor(cast(NumericArray, np.array(pixels))),
                )
            ),
        )
        assert image_features.pooler_output is not None
        reference_features = image_features.pooler_output.numpy()
    assert_close(projected, reference_features, "image_features")
    torch_masks = create_masks_for_vision_model(
        config=pair.reference.config,
        inputs_embeds=torch.zeros((1, ids.shape[1], 16)),
        attention_mask=torch.tensor(
            cast(NumericArray, np.array(mask)), dtype=torch.int64
        ),
        past_key_values=None,
        position_ids=torch.arange(ids.shape[1]).reshape(1, -1),
        block_sequence_ids=get_block_sequence_ids_for_mask(
            torch.tensor(cast(NumericArray, np.array(ids == 63)))
        ),
    )
    actual_mask = cast(NumericArray, np.array(image_attention_mask(ids, mask, 63)))
    full_mask = torch_masks["full_attention"]
    assert full_mask is not None
    reference_mask = cast(NumericArray, full_mask.numpy())
    np.testing.assert_array_equal(actual_mask, reference_mask)
