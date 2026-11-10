"""Medical path regressions and parity checks against official Transformers.

These are synthetic engineering tests, not medical task accuracy benchmarks.
Weights are seeded random tiny models and inputs are synthetic, so agreement
here does not show agreement for the full checkpoints. Run through
``make test``, which sets MLX_ENABLE_TF32=0 and uses the ``med-vlm-mlx`` environment.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, cast

from numpy.typing import NDArray
from transformers.models.gemma3 import modeling_gemma3 as gemma_reference
from typing_extensions import TypedDict

from mlx_vlm.contracts import (
    CacheComponent,
    FloatArray,
    ImageSource,
    ModelArgument,
    NumericArray,
    ProcessedInputs,
    PromptInputs,
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
from mlx_vlm.generate.dispatch import stream_generate
from mlx_vlm.generate.types import StopCriteria
from mlx_vlm.models.base import LanguageModelOutput
from mlx_vlm.models.gemma3 import Model, ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.gemma3.attention import (
    cached_attention_mask,
    image_attention_mask,
    image_attention_metadata,
)
from mlx_vlm.models.gemma3.language import Gemma3Model
from mlx_vlm.models.gemma3.vision import VisionModel
from mlx_vlm.server.generation import ResponseGenerator
from mlx_vlm.tokenizer_utils import NaiveStreamingDetokenizer
from mlx_vlm.utils import StoppingCriteria, ThinkingBudgetCriteria
from mlx_vlm.vision_cache import VisionFeatureCache


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


def test_gemma_vision_cache_reuses_projected_features_and_invalidates(
    pair: GemmaPair,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep cache hits identical and reject changed pixels, weights or models."""
    ids, pixels, mask = image_inputs()
    cache = VisionFeatureCache()
    expected = pair.model.get_input_embeddings(ids, pixels, mask).inputs_embeds
    first = pair.model.get_input_embeddings(
        ids,
        pixels,
        mask,
        vision_cache=cache,
    ).inputs_embeds
    np.testing.assert_array_equal(host(first), host(expected))
    assert len(cache) == 1

    original = type(pair.model.vision_tower).__call__
    calls = 0

    def encode(
        tower: VisionModel,
        x: mx.array,
        output_hidden_states: bool | None = None,
    ) -> tuple[mx.array, mx.array, tuple[mx.array, ...] | None]:
        nonlocal calls
        calls += 1
        return original(tower, x, output_hidden_states=output_hidden_states)

    monkeypatch.setattr(type(pair.model.vision_tower), "__call__", encode)
    hit = pair.model.get_input_embeddings(
        ids,
        pixels,
        mask,
        vision_cache=cache,
    ).inputs_embeds
    np.testing.assert_array_equal(host(hit), host(expected))
    assert calls == 0
    pair.model.get_input_embeddings(ids, -pixels, mask, vision_cache=cache)
    assert calls == 1
    previous = host(expected)
    for mutation in ("replace", "update", "load_weights", "array", "numpy"):
        weight = pair.model.multi_modal_projector.mm_input_projection_weight
        if mutation == "replace":
            pair.model.multi_modal_projector.mm_input_projection_weight = weight * 2
        elif mutation == "numpy":
            evaluate(weight)
            borrowed = cast(FloatArray, np.asarray(weight))
            borrowed[:] *= 2
            pair.model.invalidate_vision_cache()
        else:
            weight[:] = weight * 2
            if mutation == "update":
                pair.model.update(
                    {"multi_modal_projector": {"mm_input_projection_weight": weight}}
                )
            elif mutation == "load_weights":
                pair.model.load_weights(
                    [("multi_modal_projector.mm_input_projection_weight", weight)],
                    strict=False,
                )
            else:
                pair.model.invalidate_vision_cache()
        expected_after_edit = host(
            pair.model.get_input_embeddings(ids, pixels, mask).inputs_embeds
        )
        assert not np.array_equal(expected_after_edit, previous), mutation
        before = calls
        actual = pair.model.get_input_embeddings(
            ids, pixels, mask, vision_cache=cache
        ).inputs_embeds
        np.testing.assert_array_equal(host(actual), expected_after_edit)
        assert calls == before + 1, mutation
        hit_after_edit = pair.model.get_input_embeddings(
            ids, pixels, mask, vision_cache=cache
        ).inputs_embeds
        np.testing.assert_array_equal(host(hit_after_edit), expected_after_edit)
        assert calls == before + 1, mutation
        previous = expected_after_edit
    before = calls
    make_pair().model.get_input_embeddings(ids, pixels, mask, vision_cache=cache)
    assert calls == before + 1


@pytest.mark.parametrize("entry", ["public", "http"])
def test_generation_entry_points_populate_and_reuse_vision_cache(
    pair: GemmaPair,
    entry: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids, pixels, mask = image_inputs()
    cache = VisionFeatureCache()

    class Tokenizer:
        vocab = {str(index): index for index in range(64)}
        stopping_criteria: StopCriteria = StoppingCriteria([])
        thinking_budget_criteria: ThinkingBudgetCriteria | None = None

        def decode(self, token_ids: list[int]) -> str:
            return " ".join(str(token) for token in token_ids)

        @property
        def detokenizer(self) -> NaiveStreamingDetokenizer:
            return NaiveStreamingDetokenizer(self)

    tokenizer = Tokenizer()
    model = require_generation_model(pair.model)
    worker = ResponseGenerator.__new__(ResponseGenerator)
    worker.model = model
    worker.vision_cache = cache

    def request() -> NumericArray | list[int] | None:
        if entry == "http":
            gpu_embed = cast(
                Callable[
                    [ProcessedInputs, list[ImageSource]], tuple[mx.array, PromptInputs]
                ],
                getattr(worker, "_gpu_embed"),
            )
            _, outputs = gpu_embed(
                {
                    "input_ids": ids,
                    "pixel_values": pixels,
                    "attention_mask": mask,
                },
                [pixels],
            )
            return host(outputs["inputs_embeds"])
        results = list(
            stream_generate(
                model,
                tokenizer,
                "",
                image=pixels,
                input_ids=ids,
                pixel_values=pixels,
                mask=mask,
                vision_cache=cache,
                max_tokens=2,
                temperature=0,
            )
        )
        return results[-1].token_ids

    first = request()
    assert len(cache) == 1

    def forbid_encoder(
        tower: VisionModel,
        x: mx.array,
        output_hidden_states: bool | None = None,
    ) -> tuple[mx.array, mx.array, tuple[mx.array, ...] | None]:
        raise AssertionError("A repeated request must reuse its image features")

    monkeypatch.setattr(VisionModel, "__call__", forbid_encoder)
    np.testing.assert_array_equal(request(), first)


def test_vision_cache_bounds_bytes_and_revalidates_file_versions(
    tmp_path: Path,
) -> None:
    cache = VisionFeatureCache(max_bytes=24)
    source = tmp_path / "image.png"
    source.write_bytes(b"first")
    first = mx.ones((4,), dtype=mx.float32)
    cache.put(source, first)
    assert cache.get(source) is first
    source.write_bytes(b"second")
    assert cache.get(source) is None
    cache.put("other", first)
    assert cache.current_bytes == 16 and len(cache) == 1
    cache.put("oversized", mx.ones((7,), dtype=mx.float32))
    assert cache.get("other") is first and cache.get("oversized") is None
    cache.clear()
    assert cache.current_bytes == 0 and len(cache) == 0

    features = [first]
    cache.put("sequence", features)
    features.append(mx.ones((10,), dtype=mx.float32))
    stored = cache.get("sequence")
    assert stored is not None and not isinstance(stored, mx.array) and len(stored) == 1
    cache.put("replacement", first)
    assert cache.current_bytes == 16 and len(cache) == 1


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


@pytest.mark.parametrize("max_tokens", [-1, 1.5])
def test_generation_rejects_unreachable_token_limits(
    pair: GemmaPair, max_tokens: int | float
) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        next(
            generate_step(
                mx.array([[2, 4, 6]]),
                require_generation_model(pair.model),
                None,
                None,
                max_tokens=cast(int, max_tokens),
            )
        )


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
@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_gemma_multiple_images_with_padding_matches_reference(
    window: int, padding_side: str
) -> None:
    pair = make_pair(window)
    ids = mx.array(
        [
            [0, 0, 2, 63, 63, 63, 63, 7, 63, 63, 63, 63, 9],
            [0, 2, 63, 63, 63, 63, 8, 63, 63, 63, 63, 9, 10],
        ]
    )
    if padding_side == "right":
        ids = mx.concatenate([mx.roll(ids[:1], -2, 1), mx.roll(ids[1:], -1, 1)])
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
        prompt_key_validity=features.prompt_key_validity,
        prompt_image_blocks=features.prompt_image_blocks,
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
            prompt_key_validity=features.prompt_key_validity,
            prompt_image_blocks=features.prompt_image_blocks,
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
    assert pair.model.chunked_prefill_policy(input_ids=ids)
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
        prompt_kwargs={
            "prompt_key_validity": features.prompt_key_validity,
            "prompt_image_blocks": features.prompt_image_blocks,
        },
        prefill_step_size=chunk_size,
    )
    assert batch.prefill_step_size == chunk_size
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
    validity, blocks = image_attention_metadata(
        mx.concatenate(padded_ids), mx.concatenate(padding_masks), 63
    )
    batch = PromptProcessingBatch(
        model=require_language_model(pair.model.language_model),
        uids=list(range(3)),
        input_ids=[array_int_list(ids[0]) for ids in rows],
        max_tokens=[4] * 3,
        inputs_embeds=mx.concatenate(embeddings),
        prompt_kwargs={"prompt_key_validity": validity, "prompt_image_blocks": blocks},
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


@pytest.mark.parametrize("terminal_image", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 3, 5])
def test_multiple_image_blocks_keep_all_keys_in_chunked_prefill(
    terminal_image: bool,
    chunk_size: int,
) -> None:
    pair = make_pair(sliding_window=4)
    ids = mx.array([[0, 2, 63, 63, 63, 63, 7, 63, 63, 63, 63]])
    if not terminal_image:
        ids = mx.concatenate([ids, mx.array([[9]])], axis=1)
    mask = ids != 0
    _, pixels, _ = image_inputs()
    pixels = mx.concatenate([pixels, -pixels])
    features = pair.model.get_input_embeddings(ids, pixels, mask)
    embeds = features.inputs_embeds
    expected = greedy_steps(pair, ids, pixels, mask)
    actual = greedy_steps(pair, ids, pixels, mask, prefill_step_size=chunk_size)
    for (token, probabilities), (ref_token, ref_probabilities) in zip(
        actual, expected, strict=True
    ):
        assert token == ref_token
        assert_close(host(probabilities), host(ref_probabilities))
        reference = reference_logits(pair, ids, embeds, mask)[0, -1]
        assert token == int(np.argmax(reference))
        next_id = mx.array([[token]])
        ids = mx.concatenate([ids, next_id], axis=1)
        embeds = mx.concatenate(
            [embeds, pair.model.language_model.model.embed_tokens(next_id)], axis=1
        )
        mask = mx.concatenate([mask, mx.ones((1, 1), mx.bool_)], axis=1)


def test_compact_prompt_visibility_matches_dense_windows_and_rotation() -> None:
    ids = mx.array([[0, 2, 63, 63, 7, 63, 63, 8, 0], [2, 63, 63, 3, 63, 63, 4, 0, 0]])
    validity, blocks = image_attention_metadata(ids, ids != 0, 63)
    dense = image_attention_mask(ids, ids != 0, 63)
    for offset, queries, keys, window, rotation in [
        (0, 4, 4, None, 0),
        (4, 3, 7, 4, 0),
        (9, 1, 4, 4, 2),
        (12, 1, 13, None, 0),
    ]:
        expected = cached_attention_mask(dense, offset, queries, keys, window, rotation)
        actual = cached_attention_mask(
            validity, offset, queries, keys, window, rotation, image_blocks=blocks
        )
        np.testing.assert_array_equal(host(actual), host(expected))


def test_embedding_visibility_storage_grows_linearly(pair: GemmaPair) -> None:
    ids = mx.ones((1, 32768), dtype=mx.int32)
    features = pair.model.get_input_embeddings(ids)
    assert features.attention_mask_4d is None
    assert (
        features.prompt_key_validity is not None
        and features.prompt_image_blocks is not None
    )
    assert features.prompt_key_validity.shape == ids.shape
    assert features.prompt_image_blocks.shape == ids.shape
    assert (
        features.prompt_key_validity.nbytes + features.prompt_image_blocks.nbytes
        == 5 * ids.size
    )


def test_compact_prompt_metadata_preserves_reused_image_prefix() -> None:
    pair = make_pair(sliding_window=4)
    ids, pixels, mask = image_inputs()
    prefix = pair.model.get_input_embeddings(ids, pixels, mask)
    cache = pair.model.language_model.make_cache()
    output = pair.model.language_model(
        ids,
        inputs_embeds=prefix.inputs_embeds,
        cache=cache,
        prompt_key_validity=prefix.prompt_key_validity,
        prompt_image_blocks=prefix.prompt_image_blocks,
    )
    evaluate(output.logits)
    suffix = mx.array([[11, 12, 13]])
    complete = mx.concatenate([ids, suffix], axis=1)
    validity, blocks = image_attention_metadata(complete, mx.ones(complete.shape), 63)
    expected = greedy_steps(pair, complete, pixels, mx.ones(complete.shape))
    actual = list(
        generate_step(
            suffix,
            require_generation_model(pair.model),
            None,
            mx.ones(suffix.shape),
            prompt_cache=[entry for entry in cache],
            prompt_key_validity=validity,
            prompt_image_blocks=blocks,
            temperature=0,
            prefill_step_size=1,
            max_tokens=4,
        )
    )
    for (token, probabilities), (ref_token, ref_probabilities) in zip(
        actual, expected, strict=True
    ):
        assert token == ref_token
        assert_close(host(probabilities), host(ref_probabilities))


def test_explicit_dense_prompt_attention_still_overrides_embedding_metadata(
    pair: GemmaPair,
) -> None:
    ids, pixels, mask = image_inputs()
    expected = greedy_steps(pair, ids, pixels, mask)
    actual = list(
        generate_step(
            ids,
            require_generation_model(pair.model),
            pixels,
            mask,
            attention_mask_4d=image_attention_mask(ids, mask, 63),
            temperature=0,
            prefill_step_size=1,
            max_tokens=4,
        )
    )
    for (token, probabilities), (ref_token, ref_probabilities) in zip(
        actual, expected, strict=True
    ):
        assert token == ref_token
        assert_close(host(probabilities), host(ref_probabilities))


@dataclass
class FixedLogitDecoder:
    """Supply controlled native logits at the generation/model boundary."""

    logits: mx.array

    @property
    def layers(self) -> tuple[nn.Module, ...]:
        return ()

    def __call__(
        self,
        inputs: mx.array | None = None,
        /,
        *,
        cache: Sequence[CacheComponent] | None = None,
        **kwargs: ModelArgument,
    ) -> LanguageModelOutput:
        return LanguageModelOutput(self.logits[:, None, :])


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_http_greedy_preserves_normalization_created_ties(dtype: mx.Dtype) -> None:
    # All logits are finite. Subtracting logsumexp rounds the slightly higher
    # second logit onto the first; the established normalized path chooses token0.
    logits = mx.zeros((1, 8192), dtype=dtype)
    logits[:, 1] = 1e-8 if dtype == mx.float32 else 1e-4
    probabilities = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    assert mx.argmax(logits, axis=-1)[0].item() == 1
    assert mx.argmax(probabilities, axis=-1)[0].item() == 0
    batch = GenerationBatch(
        model=FixedLogitDecoder(logits),
        uids=[0],
        inputs=mx.array([2]),
        prompt_cache=[],
        sampler=lambda scores: mx.argmax(scores, axis=-1),
        stop_criteria=lambda _: False,
        max_tokens=[2],
        greedy_sampling=True,
    )
    batch.compute_logprobs = False
    batch.next()
    result = batch.next()[0]
    assert result.token == 0
    assert result.token_logprob == 0 and result.top_logprobs is None


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
