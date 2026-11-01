"""Autoregressive token generation, batching and continuous prefill scheduling."""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import time
import warnings
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    cast,
    runtime_checkable,
)

import mlx.core as mx
from tqdm import tqdm
from typing_extensions import TypedDict

from mlx_vlm.contracts import (
    CacheComponent,
    CacheEntry,
    CacheFactory,
    ImageSource,
    InputTensor,
    ModelArgument,
    PromptInputs,
    StateView,
    array_int_list,
    array_int_rows,
    array_number_rows,
    async_evaluate,
    evaluate,
    is_cache_entry_list,
    is_pixel_values,
    is_str_list,
)
from mlx_vlm.generate.common import (
    DEFAULT_COMPLETION_BATCH_SIZE,
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_P,
    DEFAULT_PREFILL_BATCH_SIZE,
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_QUANTIZED_KV_START,
    DEFAULT_REPETITION_CONTEXT_SIZE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    chunked_prefill_enabled,
    default_prefill_step_size_for_offload,
    generation_stream,
    maybe_quantize_kv_cache,
    wired_limit,
)
from mlx_vlm.generate.types import (
    BatchGenerationOptions,
    ProcessorLike,
    TokenizerLike,
    Unpack,
    is_logits_processor_list,
    is_per_row_logits_processor_list,
    resolve_tokenizer,
)
from mlx_vlm.kv_quant import kv_quant_config
from mlx_vlm.models import cache
from mlx_vlm.models.protocols import (
    GenerationModelLike,
    LanguageModelLike,
    generation_cache_layers,
    generation_cache_model,
)
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.sample_utils import (
    apply_top_k,
    clamp_temperature,
    make_logits_processors,
    make_sampler,
    top_p_sampling,
)
from mlx_vlm.tokenizer_utils import make_streaming_detokenizer
from mlx_vlm.utils import (
    ThinkingBudgetCriteria,
    group_images_by_shape,
    prepare_inputs,
    should_add_special_tokens,
)

logger = logging.getLogger("mlx_vlm.generate")

DEFAULT_TOP_N_SIGMA = 0.0
DEFAULT_CACHE_EVAL_INTERVAL = 50

type PendingSequence = tuple[
    int,
    list[int],
    int,
    PromptInputs,
    list[Callable[[mx.array, mx.array], mx.array]] | None,
    ThinkingBudgetCriteria | None,
]


def _get_batch_cache_eval_interval() -> int:
    raw = os.environ.get("MLX_VLM_BATCH_CACHE_EVAL_INTERVAL")
    if raw is None:
        return DEFAULT_CACHE_EVAL_INTERVAL
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("Ignoring invalid MLX_VLM_BATCH_CACHE_EVAL_INTERVAL=%r", raw)
        return DEFAULT_CACHE_EVAL_INTERVAL


def _position_seed(seed: int, row_id: int, position: int) -> int:
    x = (int(seed) ^ 0x9E3779B9) & 0xFFFFFFFF
    x = (x + (int(row_id) + 1) * 0x85EBCA6B) & 0xFFFFFFFF
    x = (x ^ ((int(position) + 1) * 0xC2B2AE35)) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    return int(x & 0xFFFFFFFF)


def _position_keys(seed: int, row_ids: List[int], positions: List[int]) -> mx.array:
    return mx.stack(
        [
            mx.random.key(_position_seed(seed, row, pos))
            for row, pos in zip(row_ids, positions)
        ]
    )


def _array_float_list(value: mx.array) -> list[float]:
    """Read a vector of real log probabilities at the native array boundary."""
    values = array_number_rows(mx.reshape(value, (1, -1)))[0]
    if any(isinstance(item, complex) for item in values):
        raise TypeError("Log probabilities must be real values")
    return [float(item) for item in values if isinstance(item, (int, float))]


def _array_float_rows(value: mx.array) -> list[list[float]]:
    """Preserve batch rows when exposing real top-token log probabilities."""
    rows = array_number_rows(value)
    if any(isinstance(item, complex) for row in rows for item in row):
        raise TypeError("Log probabilities must be real values")
    return [
        [float(item) for item in row if isinstance(item, (int, float))] for row in rows
    ]


class _PositionedTargetSampler:
    """Sampler with stateless target draws keyed by generated-token position."""

    def __init__(
        self,
        *,
        temperature: float,
        top_p: float,
        top_k: int = 0,
        seed: int,
    ):
        self.temperature = clamp_temperature(float(temperature))
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.seed = int(seed)

    def _apply_top_k(self, logprobs: mx.array) -> mx.array:
        return apply_top_k(logprobs, self.top_k) if self.top_k > 0 else logprobs

    def __call__(self, logprobs: mx.array) -> mx.array:
        logprobs = self._apply_top_k(logprobs)
        if self.top_p > 0 and self.top_p < 1.0:
            return top_p_sampling(logprobs, self.top_p, self.temperature)
        return mx.random.categorical(logprobs * (1 / self.temperature))

    def sample_target(
        self,
        logprobs: mx.array,
        *,
        row_ids: List[int],
        positions: List[int],
    ) -> mx.array:
        if logprobs.shape[0] != len(row_ids) or len(row_ids) != len(positions):
            raise ValueError("row_ids and positions must match logprobs batch size.")
        logprobs = self._apply_top_k(logprobs)
        keys = _position_keys(self.seed, row_ids, positions)
        if self.top_p > 0 and self.top_p < 1.0:
            return mx.vmap(self._sample_top_p_one, in_axes=(0, 0))(logprobs, keys)
        return mx.vmap(self._sample_one, in_axes=(0, 0))(logprobs, keys)

    def _sample_one(self, logprobs: mx.array, key: mx.array) -> mx.array:
        return mx.random.categorical(logprobs * (1 / self.temperature), key=key)

    def _sample_top_p_one(self, logprobs: mx.array, key: mx.array) -> mx.array:
        return top_p_sampling(logprobs, self.top_p, self.temperature, key=key)


def normalize_resize_shape(values: object) -> tuple[int, int] | None:
    """Validate a one- or two-element size, rejecting strings and booleans."""
    if values is None:
        return None
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("resize_shape must contain 1 or 2 integers")
    items = cast(Sequence[object], values)
    if not (len(items) in (1, 2) and all(type(value) is int for value in items)):
        raise ValueError("resize_shape must contain 1 or 2 integers")
    sizes = cast(Sequence[int], items)  # Every element passed the exact-int check.
    return (sizes[0], sizes[0]) if len(sizes) == 1 else (sizes[0], sizes[1])


def generate_step(
    input_ids: mx.array,
    model: GenerationModelLike,
    pixel_values: InputTensor | list[InputTensor] | list[list[InputTensor]] | None,
    mask: mx.array | None,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = DEFAULT_REPETITION_CONTEXT_SIZE,
    presence_penalty: Optional[float] = None,
    presence_context_size: Optional[int] = DEFAULT_REPETITION_CONTEXT_SIZE,
    frequency_penalty: Optional[float] = None,
    frequency_context_size: Optional[int] = DEFAULT_REPETITION_CONTEXT_SIZE,
    top_p: float = DEFAULT_TOP_P,
    min_p: float = DEFAULT_MIN_P,
    top_k: int = DEFAULT_TOP_K,
    top_n_sigma: float = DEFAULT_TOP_N_SIGMA,
    p_less: bool = False,
    typical_p: float = 1.0,
    logit_bias: Optional[Dict[int, float]] = None,
    prompt_cache: Optional[List[CacheEntry]] = None,
    max_kv_size: Optional[int] = None,
    kv_bits: Optional[float] = None,
    kv_group_size: int = DEFAULT_KV_GROUP_SIZE,
    quantized_kv_start: int = DEFAULT_QUANTIZED_KV_START,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    prefill_step_size: Optional[int] = DEFAULT_PREFILL_STEP_SIZE,
    seed: Optional[int] = None,
    verbose: bool = False,
    thinking_budget_criteria: ThinkingBudgetCriteria | None = None,
    **kwargs: ModelArgument,
) -> Generator[Tuple[int | mx.array | None, mx.array | None]]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        input_ids (mx.array): The input prompt token ids.
        model (nn.Module): The model to use for generation.
        pixel_values: The pixel values for vision models (optional).
        mask: The attention mask (optional).
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): The temperature for sampling, if 0 the argmax is used.
        repetition_penalty (float, optional): The penalty factor for repeating
          tokens.
        repetition_context_size (int, optional): The number of tokens to
          consider for repetition penalty.
        presence_penalty (float, optional): Additive penalty for tokens that
          already appeared in recent generated context.
        presence_context_size (int, optional): The number of tokens to
          consider for presence penalty.
        frequency_penalty (float, optional): Additive penalty scaled by token
          frequency in recent generated context.
        frequency_context_size (int, optional): The number of tokens to
          consider for frequency penalty.
        top_p (float, optional): Nucleus sampling, higher means model considers
          more less likely words.
        min_p (float, optional): Minimum probability threshold relative to the
          highest-probability token.
        top_k (int, optional): Restrict sampling to the top-k tokens.
        logit_bias (dictionary, optional): Additive logit bias.
        prompt_cache (list, optional): Pre-existing KV cache for the prompt.
        max_kv_size (int, optional): Maximum KV cache size.
        kv_bits (float, optional): Number of bits for KV cache quantization.
        kv_group_size (int): Group size for uniform KV cache quantization.
        quantized_kv_start (int): Start index for quantized KV cache.
        sampler (Callable[mx.array, mx.array], optional): A sampler for sampling a
          token from a vector of log probabilities.
        logits_processors (List[Callable[[mx.array, mx.array], mx.array]], optional):
          A list of functions that take tokens and logits and return the processed
          logits.
        prefill_step_size (int): Number of tokens to process per prefill step.
          Chunking is subject to the model's prefill policy; for example, Gemma
          image blocks must remain whole.

    Yields:
        Generator[Tuple[mx.array, mx.array], None, None]: A generator producing
          one token and a vector of log probabilities.
    """

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    language_model = model.language_model

    temperature = clamp_temperature(temperature)
    if sampler is None:
        if (
            seed is not None
            and temperature > 0
            and min_p == DEFAULT_MIN_P
            and top_n_sigma == DEFAULT_TOP_N_SIGMA
            and not p_less
            and typical_p == 1.0
        ):
            sampler = _PositionedTargetSampler(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
            )
        else:
            sampler = make_sampler(
                temp=temperature,
                top_p=top_p,
                min_p=min_p,
                top_k=top_k,
                top_n_sigma=top_n_sigma,
                p_less=p_less,
                typical_p=typical_p,
            )

    processors = make_logits_processors(
        logit_bias,
        repetition_penalty,
        repetition_context_size,
        presence_penalty,
        presence_context_size,
        frequency_penalty,
        frequency_context_size,
    )
    if logits_processors is not None:
        processors.extend(logits_processors)

    y = input_ids
    tokens = mx.array([], dtype=input_ids.dtype)
    target_sample_position = 0

    # Create the KV cache for generation
    if prompt_cache is None:
        created_cache = cache.make_prompt_cache(
            language_model,
            max_kv_size=max_kv_size,
        )
        if not is_cache_entry_list(created_cache):
            raise TypeError(
                "Generation requires cache entries exposing state and metadata"
            )
        prompt_cache = created_cache
    assert prompt_cache is not None

    def _step(y: mx.array, inputs_embeds: mx.array | None = None):
        nonlocal tokens, kwargs, target_sample_position

        step_kwargs = {
            **kwargs,
            "logits_to_keep": 1,
        }

        with mx.stream(generation_stream):
            if "decoder_input_ids" in step_kwargs:
                outputs = language_model(
                    cache=prompt_cache,
                    **step_kwargs,
                )
            else:
                outputs = language_model(
                    y,
                    inputs_embeds=inputs_embeds,
                    cache=prompt_cache,
                    **step_kwargs,
                )

            logits = outputs.logits[:, -1, :]

            if len(processors) > 0 and len(y) > 0:
                tokens = mx.concat([tokens, y.flatten()])

                for processor in processors:
                    logits = processor(tokens, logits)

            quantize_cache_fn(prompt_cache)

            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            y = _sample_with_positions(
                sampler,
                logprobs,
                row_ids=[0] * logprobs.shape[0],
                positions=list(
                    range(
                        target_sample_position,
                        target_sample_position + logprobs.shape[0],
                    )
                ),
            )
            target_sample_position += logprobs.shape[0]

            kwargs = {
                key: kwargs[key]
                for key in getattr(model, "generation_persistent_kwargs", ())
                if key in kwargs
            }

            return y, logprobs.squeeze(0) if logprobs.shape[0] == 1 else logprobs

    with mx.stream(generation_stream):
        # A reused prompt cache leaves only the uncached suffix here. Embedding
        # helpers see only that suffix and may return local positions, so
        # explicit caller metadata takes precedence over their derived values.
        explicit_prompt_metadata = {
            key: kwargs[key]
            for key in ("attention_mask_4d",)
            if kwargs.get(key) is not None
        }
        # Get input embeddings (handles both multimodal and text-only)
        get_input_embeddings = model.get_input_embeddings
        if not callable(get_input_embeddings):
            raise TypeError("Generation model must implement get_input_embeddings")
        embedding_output = get_input_embeddings(
            input_ids, pixel_values, mask=mask, **kwargs
        )

        inputs_embeds = embedding_output.inputs_embeds

        kwargs.update(
            {
                k: v
                for k, v in embedding_output.to_dict().items()
                if k != "inputs_embeds" and v is not None
            }
        )
        kwargs.update(explicit_prompt_metadata)
        policy_kwargs = kwargs
        prefill_step_size = default_prefill_step_size_for_offload(
            model, prefill_step_size, DEFAULT_PREFILL_STEP_SIZE
        )
        if prefill_step_size is not None and not chunked_prefill_enabled(
            model,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            prompt_cache=prompt_cache,
            prefill_kwargs=policy_kwargs,
        ):
            prefill_step_size = None
        # Chunk whenever there is more than one prompt token left to process.
        # The chunk loop discards its output, so the [B, N, vocab] logits are
        # never evaluated; the unchunked path feeds the whole prompt to _step,
        # which reads logits[:, -1, :] and so materializes every row. Gating on
        # prefill_step_size made short prompts peak higher than long ones.
        if prefill_step_size is not None and inputs_embeds.shape[1] > 1:
            # Chunked prefill with embeddings
            total_tokens = inputs_embeds.shape[1]
            with tqdm(
                total=total_tokens, desc="Prefill", unit="tok", disable=not verbose
            ) as pbar:
                while inputs_embeds.shape[1] > 1:
                    n_to_process = min(prefill_step_size, inputs_embeds.shape[1] - 1)
                    chunk_kwargs = {
                        **kwargs,
                        "logits_to_keep": 1,
                    }
                    chunk_output = language_model(
                        inputs=input_ids[:, :n_to_process],
                        inputs_embeds=inputs_embeds[:, :n_to_process],
                        cache=prompt_cache,
                        n_to_process=n_to_process,
                        **chunk_kwargs,
                    )
                    del chunk_output
                    quantize_cache_fn(prompt_cache)
                    evaluate([c.state for c in prompt_cache])
                    inputs_embeds = inputs_embeds[:, n_to_process:]
                    input_ids = input_ids[:, n_to_process:]
                    mx.clear_cache()
                    pbar.update(n_to_process)

            input_ids = input_ids[:, -1:]

        y, logprobs = _step(input_ids, inputs_embeds=inputs_embeds)

    async_evaluate(y, logprobs)

    next_y, next_logprobs = y, logprobs
    n = 0
    while True:
        if n != max_tokens:
            next_y, next_logprobs = _step(y[None])
            async_evaluate(next_y, next_logprobs)
        if n == 0:
            evaluate(y)
        if n == max_tokens:
            break

        if (n + 1) % DEFAULT_CACHE_EVAL_INTERVAL == 0:
            evaluate([c.state for c in prompt_cache])

        token: object = y.item()
        assert isinstance(token, int)
        yield token, logprobs
        if n % 256 == 0:
            mx.clear_cache()

        if thinking_budget_criteria is not None:
            forced_token_id = thinking_budget_criteria.pop_forced_token_id()
            if forced_token_id is not None:
                next_y = mx.array([forced_token_id], dtype=next_y.dtype)
        y, logprobs = next_y, next_logprobs
        n += 1


def _left_pad_prompts(prompts: list[list[int]], max_length: int | None = None):
    if max_length is None:
        max_length = max(len(p) for p in prompts)

    return mx.array([[0] * (max_length - len(p)) + p for p in prompts])


_SEQUENCE_ALIGNED_PROMPT_KWARGS = {"attention_mask", "token_type_ids"}


def _prompt_kwarg_batch_size(key: str, v: mx.array) -> int:
    return v.shape[0] if v.ndim > 0 else 0


def _prompt_kwarg_row(key: str, v: mx.array, row_idx: int, batch_size: int) -> mx.array:
    if v.shape[0] == batch_size:
        return v[row_idx : row_idx + 1]
    return v[:1]


def _split_prompt_kwargs_per_row(
    prompt_kwargs: PromptInputs, batch_size: int
) -> List[PromptInputs]:
    """Normalize batched prompt kwargs into one dict per batch row.

    ``model.get_input_embeddings()`` commonly returns batch-sized tensors
    (notably ``inputs_embeds``). ``BatchGenerator.insert()`` stores prompt
    kwargs per sequence, so passing the same batched dict for every row causes
    the prompt builder to concatenate those batched tensors ``batch_size``
    times, effectively squaring the batch dimension.
    """
    if batch_size <= 1:
        return [prompt_kwargs or {}]

    rows: list[PromptInputs] = [{} for _ in range(batch_size)]
    for k, v in (prompt_kwargs or {}).items():
        if isinstance(v, mx.array) and _prompt_kwarg_batch_size(k, v) >= 1:
            for i in range(batch_size):
                rows[i][k] = _prompt_kwarg_row(k, v, i, batch_size)
        else:
            for row in rows:
                row[k] = v
    return rows


def _unpad_batch_prompts(
    prompts: Union[mx.array, List[List[int]]],
    attention_mask: Optional[mx.array],
    prompt_kwargs: List[PromptInputs],
) -> Tuple[List[List[int]], List[PromptInputs]]:
    """Convert a masked dense batch to the generator's ragged row format.

    ``BatchGenerator`` owns prompt padding and derives cache offsets from the
    row lengths it receives. Remove processor padding, together with the
    corresponding columns of every sequence-aligned prompt tensor, before the
    rows are added to its queue.
    """
    if attention_mask is None:
        if isinstance(prompts, mx.array):
            if prompts.ndim != 2:
                raise ValueError("Batched prompts must be a 2D array")
            return array_int_rows(prompts), prompt_kwargs
        return prompts, prompt_kwargs

    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be a 2D MLX array")

    if isinstance(prompts, mx.array):
        if prompts.ndim != 2 or attention_mask.shape != prompts.shape:
            raise ValueError("attention_mask must have the same 2D shape as prompts")
        batch_size, sequence_length = prompts.shape
        # Materialize the tokenizer outputs together. Both are converted to
        # Python metadata below, so one explicit boundary avoids row-by-row
        # GPU syncs.
        evaluate(prompts, attention_mask)
        input_rows = array_int_rows(prompts)
    else:
        batch_size, sequence_length = attention_mask.shape
        if len(prompts) != batch_size or any(
            len(row) != sequence_length for row in prompts
        ):
            raise ValueError("attention_mask must have the same 2D shape as prompts")
        evaluate(attention_mask)
        input_rows = [list(row) for row in prompts]

    if len(prompt_kwargs) != batch_size:
        raise ValueError("prompt_kwargs must contain one dictionary per prompt")

    mask_rows = array_number_rows(attention_mask)

    unpadded_prompts: List[List[int]] = []
    for row_idx, (ids, row_mask, kwargs) in enumerate(
        zip(input_rows, mask_rows, prompt_kwargs)
    ):
        start = next((i for i, value in enumerate(row_mask) if bool(value)), None)
        if start is None:
            raise ValueError(f"attention_mask row {row_idx} contains no prompt tokens")
        stop = sequence_length - next(
            (i for i, value in enumerate(reversed(row_mask)) if bool(value)), 0
        )
        if any(not bool(value) for value in row_mask[start:stop]):
            raise ValueError("attention_mask must contain one contiguous prompt span")

        unpadded_prompts.append(ids[start:stop])
        if start == 0 and stop == sequence_length:
            continue

        for key, value in list(kwargs.items()):
            if not isinstance(value, mx.array):
                continue
            if key == "attention_mask_4d":
                kwargs[key] = value[:, :, start:stop, start:stop]
                continue
            if key == "inputs_embeds":
                sequence_aligned = value.ndim >= 2 and value.shape[1] == sequence_length
            else:
                sequence_aligned = _is_sequence_aligned_prompt_kwarg(
                    key, value, sequence_length
                )
            if sequence_aligned:
                kwargs[key] = _slice_sequence_aligned_prompt_kwarg(
                    key, value, start=start, stop=stop
                )

    return unpadded_prompts, prompt_kwargs


def _is_sequence_aligned_prompt_kwarg(
    key: str, v: mx.array, sequence_length: int
) -> bool:
    if key not in _SEQUENCE_ALIGNED_PROMPT_KWARGS:
        return False
    return v.ndim >= 2 and v.shape[1] == sequence_length


def _pad_sequence_aligned_prompt_kwarg(
    key: str, v: mx.array, target_length: int, *, left: bool
) -> mx.array:
    sequence_axis = 1
    pad = target_length - v.shape[sequence_axis]
    if pad <= 0:
        return v
    pad_shape = tuple(
        pad if axis == sequence_axis else size for axis, size in enumerate(v.shape)
    )
    pad_v = mx.zeros(pad_shape, dtype=v.dtype)
    parts = [pad_v, v] if left else [v, pad_v]
    return mx.concatenate(parts, axis=sequence_axis)


def _slice_sequence_aligned_prompt_kwarg(
    key: str, v: mx.array, start: Optional[int] = None, stop: Optional[int] = None
) -> mx.array:
    sequence_axis = 1
    slices = [slice(None)] * v.ndim
    slices[sequence_axis] = slice(start, stop)
    return v[tuple(slices)]


def _concat_prompt_kwarg_rows(key: str, rows: List[mx.array]) -> mx.array:
    return mx.concatenate(rows, axis=0)


def _pad_prompt_attention_mask(value: mx.array, left: int, right: int) -> mx.array:
    """Align both prompt-mask axes with padded cached-prefix and suffix columns."""
    if value.ndim != 4 or value.shape[1] != 1 or value.shape[2] != value.shape[3]:
        raise ValueError("Prompt attention must have shape [batch, 1, length, length]")
    if left < 0 or right < 0:
        raise ValueError("Prompt attention padding cannot be negative")
    return mx.pad(value, [(0, 0), (0, 0), (left, right), (left, right)])


def _merge_prefill_prompt_kwargs(
    prompt_kwargs_list: Sequence[PromptInputs | None],
    input_ids: List[List[int]],
) -> Tuple[mx.array, PromptInputs]:
    """Batch per-row prompt kwargs for a left-padded prefill forward."""
    lengths = [len(ids) for ids in input_ids]
    max_length = max(lengths)

    row_embeds: List[mx.array] = []
    embed_dtype = None
    embed_dim = None
    for kw, length in zip(prompt_kwargs_list, lengths):
        if not kw or kw.get("inputs_embeds") is None:
            raise ValueError("inputs_embeds is required")
        embeds = kw["inputs_embeds"]  # [1, length, D]
        assert isinstance(embeds, mx.array)
        embed_dtype = embeds.dtype
        embed_dim = embeds.shape[-1]
        if length < max_length:
            pad = mx.zeros(
                (embeds.shape[0], max_length - length, embed_dim),
                dtype=embed_dtype,
            )
            embeds = mx.concatenate([pad, embeds], axis=1)
        row_embeds.append(embeds)
    inputs_embeds = mx.concatenate(row_embeds, axis=0)

    merged_kwargs: PromptInputs = {}
    per_row_keys: dict[str, list[mx.array]] = {}
    batch_size = len(prompt_kwargs_list)
    for i, (kw, length) in enumerate(zip(prompt_kwargs_list, lengths)):
        if not kw:
            continue
        for k, v in kw.items():
            if k == "inputs_embeds":
                continue
            if isinstance(v, mx.array) and _prompt_kwarg_batch_size(k, v) >= 1:
                row_v = _prompt_kwarg_row(k, v, i, batch_size)
                if k == "attention_mask_4d":
                    row_v = _pad_prompt_attention_mask(row_v, max_length - length, 0)
                elif _is_sequence_aligned_prompt_kwarg(k, row_v, length):
                    row_v = _pad_sequence_aligned_prompt_kwarg(
                        k, row_v, max_length, left=True
                    )
                per_row_keys.setdefault(k, []).append(row_v)
            elif k not in merged_kwargs:
                merged_kwargs[k] = v
    for k, vs in per_row_keys.items():
        merged_kwargs[k] = _concat_prompt_kwarg_rows(k, vs)

    return inputs_embeds, merged_kwargs


def _is_batch_cache_entry(entry: CacheComponent) -> bool:
    """Return whether a cache entry already owns a batch dimension."""
    return callable(getattr(entry, "filter", None)) and callable(
        getattr(entry, "extend", None)
    )


class _CacheMerger(Protocol):
    def __call__(self, caches: list[CacheEntry]) -> object: ...


@runtime_checkable
class _ExtendableCache(CacheEntry, Protocol):
    def extend(self, other: CacheEntry) -> None: ...


def _merge_cache(entry: CacheEntry) -> CacheEntry:
    merge = getattr(type(entry), "merge", None)
    if _is_batch_cache_entry(entry) or not callable(merge):
        return entry
    # Native cache class methods accept singleton lists for batch conversion.
    merged = cast(_CacheMerger, merge)([entry])
    if not isinstance(merged, CacheEntry):
        raise TypeError("Merged cache must expose state and meta_state")
    return merged


def _extend_cache(
    cache_a: list[CacheEntry], cache_b: list[CacheEntry]
) -> list[CacheEntry]:
    """Extend cache_a with cache_b along the batch dimension.

    Every layer is checked for batch support before any is mutated.
    """
    if not cache_a:
        return cache_b
    if not cache_b:
        return cache_a
    if len(cache_a) != len(cache_b):
        raise ValueError("batch cache extension requires matching layer counts")
    pairs = [(_merge_cache(ca), _merge_cache(cb)) for ca, cb in zip(cache_a, cache_b)]
    for ca, cb in pairs:
        if not isinstance(ca, _ExtendableCache) or not isinstance(cb, _ExtendableCache):
            raise ValueError("Cache does not yet support batching")
    extended: list[CacheEntry] = []
    for ca, cb in pairs:
        if not isinstance(ca, _ExtendableCache):
            raise ValueError("Cache does not yet support batching")
        ca.extend(cb)
        extended.append(ca)
    return extended


def _make_cache(
    model: LanguageModelLike | GenerationModelLike,
    left_padding: list[int] | None,
    kv_bits: float | None = None,
    kv_group_size: int = 64,
) -> list[CacheEntry]:
    """
    Convert a list of regular caches into their corresponding
    batch-aware caches.

    When *kv_bits* is set, a quantized batch cache is used instead of
    ``BatchKVCache`` so that KV states are quantized on-the-fly during
    generation, reducing memory usage for long sequences.
    """
    left_padding = [0] if left_padding is None else left_padding
    quant_config = kv_quant_config(kv_bits, kv_group_size)

    def _make_quant_cache(lp: list[int]) -> CacheEntry:
        assert quant_config is not None
        return cache.BatchQuantizedKVCache(
            lp, group_size=quant_config["group_size"], bits=quant_config["bits"]
        )

    def to_batch_cache(c: CacheComponent, quantize: bool = True) -> CacheEntry:
        if isinstance(c, cache.KVCache):
            if kv_bits is not None and quantize:
                return _make_quant_cache(left_padding)
            return cache.BatchKVCache(left_padding)
        elif isinstance(c, cache.RotatingKVCache):
            if c.keep > 0:
                raise ValueError("RotatingKVCache with keep tokens is not supported.")
            return cache.BatchRotatingKVCache(c.max_size, left_padding)
        else:
            raise ValueError(f"{type(c)} does not yet support batching")

    if isinstance(model, CacheFactory):
        model_cache = model.make_cache()
        n = len(model_cache)
        return [
            to_batch_cache(c, quantize=cache.should_quantize_kv_layer(i, n))
            for i, c in enumerate(model_cache)
        ]
    else:
        if kv_bits is not None:
            n = len(generation_cache_layers(model))
            return [
                (
                    _make_quant_cache(left_padding)
                    if cache.should_quantize_kv_layer(i, n)
                    else cache.BatchKVCache(left_padding)
                )
                for i in range(n)
            ]
        return [
            cache.BatchKVCache(left_padding) for _ in generation_cache_layers(model)
        ]


@dataclass
class BatchStats:
    """
    An data object to hold generation stats.

    Args:
        prompt_tokens (int): The number of prompt tokens processed.
        prompt_tps (float): The prompt processing tokens-per-second.
        prompt_time (float): The time in seconds spent in prompt processing.
        generation_tokens (int): The number of generated tokens.
        generation_tps (float): The tokens-per-second for generation.
        generation_time (float): The time in seconds spent in generation .
        peak_memory (float): The peak memory used so far in GB.
    """

    prompt_tokens: int = 0
    prompt_tps: float = 0
    prompt_time: float = 0
    generation_tokens: int = 0
    generation_tps: float = 0
    generation_time: float = 0
    peak_memory: float = 0


@dataclass
class BatchResponse:
    """
    An data object to hold a batch generation response.

    Args:
        texts: (List[str]): The generated text for each prompt.
        stats (BatchStats): Statistics about the generation.
        image_sizes: (Optional[List[Tuple[int, int]]]): Original (height, width)
            for each image. Useful for tracking which images produced which responses
            and for debugging padding/batching behavior.
    """

    texts: List[str]
    stats: BatchStats
    image_sizes: Optional[List[Optional[Tuple[int, int]]]] = None


@dataclass
class PromptProgress:
    """Per-request prompt processing metrics for continuous batching."""

    uid: int
    prompt_tokens: int
    prompt_tps: float = 0.0
    prompt_time: float = 0.0
    cached_tokens: int = 0


def _sample_with_positions(
    sampler: Callable[[mx.array], mx.array],
    logprobs: mx.array,
    *,
    row_ids: Optional[List[int]] = None,
    positions: Optional[List[int]] = None,
) -> mx.array:
    sample_target = getattr(sampler, "sample_target", None)
    if callable(sample_target) and row_ids is not None and positions is not None:
        sampled = sample_target(logprobs, row_ids=row_ids, positions=positions)
        if not isinstance(sampled, mx.array):
            raise TypeError("Position-aware samplers must return an MLX array")
        return sampled
    return sampler(logprobs)


class GenerationBatch:
    """
    Batched token generator with double-buffered pipelining.

    Manages the generation phase after prompt processing, with KV caches,
    sampling, and stop detection for multiple sequences. Uses async_eval
    to overlap GPU computation with CPU processing (decode-ahead pattern).
    """

    uids: list[int]
    prompt_cache: list[CacheEntry]
    sampler: Callable[[mx.array], mx.array]
    max_tokens: list[int]
    _num_tokens: list[int]
    token_context: list[list[int]]
    logits_processors: list[list[Callable[[mx.array, mx.array], mx.array]] | None]
    thinking_budget_criteria: list[ThinkingBudgetCriteria | None]
    next_lps: mx.array | None
    next_top_idx: mx.array | None
    next_top_lp: mx.array | None

    @dataclass
    class Response:
        uid: int
        token: int | None
        token_logprob: float
        finish_reason: Optional[str]
        top_logprobs: Optional[List[Tuple[int, float]]] = None

    def __init__(
        self,
        model: LanguageModelLike | GenerationModelLike,
        uids: List[int],
        inputs: mx.array,
        prompt_cache: List[CacheEntry],
        sampler: Callable[[mx.array], mx.array],
        stop_criteria: Callable[[int], bool],
        max_tokens: List[int],
        top_logprobs_k: int = 0,
        greedy_sampling: bool = False,
        token_context: Optional[List[List[int]]] = None,
        logits_processors: Optional[
            List[Optional[List[Callable[[mx.array, mx.array], mx.array]]]]
        ] = None,
        thinking_budget_criteria: Optional[
            List[Optional[ThinkingBudgetCriteria]]
        ] = None,
    ):
        """Initialize decoding for prefilled ``uids`` from their first token.

        ``inputs`` holds one prompt-final token per row and ``prompt_cache``
        already covers each prompt. Per-row processors and thinking criteria are
        optional lists that are padded or truncated to the batch size.
        """
        self.model = model
        # MLX's dynamic child lookup is the native module boundary.
        language_model = getattr(model, "language_model", model)
        self._language_model = cast(LanguageModelLike, language_model)
        self.uids = uids
        self.prompt_cache = prompt_cache
        self.sampler = sampler
        self.stop_criteria = stop_criteria
        self.max_tokens = max_tokens
        self._num_tokens = [0] * len(uids)
        self.compute_logprobs = True
        self.top_logprobs_k = top_logprobs_k
        self.greedy_sampling = greedy_sampling
        self.logits_processors = logits_processors or []
        self.thinking_budget_criteria = thinking_budget_criteria or []
        self.token_context = [list(ctx) for ctx in (token_context or [])]
        self._ensure_token_context()

        self._current_tokens = None
        self._current_lps = None
        self._next_tokens = inputs
        self.next_lps = None
        self.next_top_idx = None
        self.next_top_lp = None

    def __len__(self):
        """Return the number of sequences currently decoding."""
        return len(self.uids)

    def cache_states(self):
        """Return the per-layer KV cache state arrays for evaluation."""
        return [c.state for c in self.prompt_cache if hasattr(c, "state")]

    def _ensure_logits_processor_slots(self, *, force: bool = False):
        if not (force or (self.logits_processors and any(self.logits_processors))):
            return
        if len(self.logits_processors) < len(self.uids):
            missing = len(self.uids) - len(self.logits_processors)
            self.logits_processors.extend([None] * missing)
        elif len(self.logits_processors) > len(self.uids):
            self.logits_processors = self.logits_processors[: len(self.uids)]

    def _ensure_token_context(self, *, force: bool = False):
        if not (force or (self.logits_processors and any(self.logits_processors))):
            if not self.logits_processors:
                self.token_context = []
            return
        if len(self.token_context) < len(self.uids):
            missing = len(self.uids) - len(self.token_context)
            self.token_context.extend([[] for _ in range(missing)])
        elif len(self.token_context) > len(self.uids):
            self.token_context = self.token_context[: len(self.uids)]

    def _fused_greedy_step(
        self, inputs: mx.array, fwd_kwargs: PromptInputs
    ) -> mx.array | None:
        if not self.greedy_sampling or self.compute_logprobs or self.top_logprobs_k > 0:
            return None

        fused_greedy_decode = getattr(self._language_model, "fused_greedy_decode", None)
        if not callable(fused_greedy_decode):
            return None

        decode_kwargs = dict(fwd_kwargs)
        if self.logits_processors and any(self.logits_processors):
            supports_processors = getattr(
                self._language_model, "supports_fused_greedy_logits_processors", None
            )
            if not callable(supports_processors) or not supports_processors(
                self.logits_processors
            ):
                return None
            decode_kwargs["logits_processors"] = self.logits_processors
        sampled = fused_greedy_decode(
            inputs[:, None],
            cache=self.prompt_cache,
            **decode_kwargs,
        )
        if sampled is None:
            return None
        assert isinstance(sampled, mx.array)
        if sampled.ndim == 2 and sampled.shape[1] == 1:
            sampled = sampled[:, 0]
        return sampled

    def _step(self):
        """Perform one generation step with double buffering."""
        self._current_tokens = self._next_tokens
        self._current_lps = self.next_lps
        inputs = self._current_tokens
        assert inputs is not None

        fwd_kwargs: PromptInputs = {}

        sampled = self._fused_greedy_step(inputs, fwd_kwargs)
        if sampled is not None:
            self._next_tokens = sampled
            self.next_lps = None
            self.next_top_idx = None
            self.next_top_lp = None
            async_evaluate(self._next_tokens)
            evaluate(inputs)
            return array_int_list(inputs), None, None, None

        output = self._language_model(
            inputs[:, None], cache=self.prompt_cache, **fwd_kwargs
        )
        logits = output if isinstance(output, mx.array) else output.logits
        logits = logits[:, -1, :]

        if self.logits_processors and any(self.logits_processors):
            last_tokens = array_int_list(inputs)
            self._ensure_token_context()
            for i, token in enumerate(last_tokens):
                self.token_context[i].append(token)

            processed_logits: list[mx.array] = []
            for i in range(logits.shape[0]):
                sample_logits = logits[i : i + 1]
                processors = self.logits_processors[i] or []
                for processor in processors:
                    if hasattr(processor, "process_last_token"):
                        hook: object = getattr(processor, "process_last_token")
                        if not callable(hook):
                            raise TypeError(
                                f"'{type(hook).__name__}' object is not callable"
                            )
                        processed: object = hook(last_tokens[i], sample_logits)
                        assert isinstance(processed, mx.array)
                        sample_logits = processed
                    else:
                        sample_logits = processor(
                            mx.array(self.token_context[i]), sample_logits
                        )
                processed_logits.append(sample_logits)
            logits = mx.concatenate(processed_logits, axis=0)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        sampled = _sample_with_positions(
            self.sampler,
            logprobs,
            row_ids=[0] * len(self.uids),
            positions=[n + 1 for n in self._num_tokens],
        )

        self._next_tokens = sampled
        prev_top_idx = self.next_top_idx
        prev_top_lp = self.next_top_lp

        eval_targets = [self._next_tokens]
        if self.compute_logprobs:
            self.next_lps = logprobs[mx.arange(sampled.shape[0]), sampled]
            eval_targets.append(self.next_lps)
        else:
            self.next_lps = None

        k = self.top_logprobs_k
        if k > 0:
            # argsort ascending; take last K columns and reverse for descending.
            sort_idx = mx.argsort(logprobs, axis=-1)
            top_idx = sort_idx[..., -k:][..., ::-1].astype(mx.int32)
            top_lp = mx.take_along_axis(logprobs, top_idx, axis=-1)
            self.next_top_idx = top_idx
            self.next_top_lp = top_lp
            eval_targets.extend([top_idx, top_lp])
        else:
            self.next_top_idx = None
            self.next_top_lp = None

        async_evaluate(*eval_targets)

        if self._current_lps is not None:
            to_eval = [inputs, self._current_lps]
            if prev_top_idx is not None and prev_top_lp is not None:
                to_eval.extend([prev_top_idx, prev_top_lp])
            evaluate(*to_eval)
            top_idx_list = (
                array_int_rows(prev_top_idx) if prev_top_idx is not None else None
            )
            top_lp_list = (
                _array_float_rows(prev_top_lp) if prev_top_lp is not None else None
            )
            return (
                array_int_list(inputs),
                _array_float_list(self._current_lps),
                top_idx_list,
                top_lp_list,
            )
        else:
            evaluate(inputs)
            return array_int_list(inputs), None, None, None

    def _eval_pending_state(self):
        """Materialize lazy decode outputs before mutating batch-owned state."""
        targets: list[StateView] = []
        stack: list[StateView] = [
            self._current_tokens,
            self._current_lps,
            self._next_tokens,
            self.next_lps,
            self.next_top_idx,
            self.next_top_lp,
        ]
        for c in self.prompt_cache:
            try:
                stack.append(c.state)
            except (AttributeError, TypeError):
                pass

        # Recursing through a local closure would retain targets in a cycle.
        while stack:
            value = stack.pop()
            if isinstance(value, mx.array):
                targets.append(value)
            elif isinstance(value, (list, tuple)):
                stack.extend(value)

        if targets:
            evaluate(*targets)

    def extend(self, other: "GenerationBatch"):
        """Extend this batch with another generation batch.

        Cache compatibility is checked before cache or batch metadata changes.
        Other cache extension failures are not guaranteed to roll back prior layers.
        """
        self_was_empty = len(self.uids) == 0
        if not self_was_empty and len(other.uids) > 0:
            if len(self.prompt_cache) != len(other.prompt_cache):
                raise ValueError("batch cache extension requires matching layer counts")
            self._eval_pending_state()
            other._eval_pending_state()

        # Check all packed layers before changing batch metadata or any cache row.
        extended_cache = _extend_cache(self.prompt_cache, other.prompt_cache)

        self_has_processors = self.logits_processors and any(self.logits_processors)
        other_has_processors = other.logits_processors and any(other.logits_processors)
        if self_has_processors or other_has_processors:
            self._ensure_logits_processor_slots(force=bool(other_has_processors))
            other._ensure_logits_processor_slots(force=bool(self_has_processors))
            self._ensure_token_context(force=bool(other_has_processors))
            other._ensure_token_context(force=bool(self_has_processors))
        else:
            self.token_context = []
            other.token_context = []
            self.logits_processors = []
            other.logits_processors = []

        self.uids.extend(other.uids)
        self.prompt_cache = extended_cache
        self.max_tokens.extend(other.max_tokens)
        self._num_tokens.extend(other._num_tokens)
        self.token_context.extend(other.token_context)
        self.logits_processors.extend(other.logits_processors)
        self.thinking_budget_criteria.extend(other.thinking_budget_criteria)
        self._ensure_logits_processor_slots()
        self._ensure_token_context()

        if self._current_tokens is None:
            self._current_tokens = other._current_tokens
            self._current_lps = other._current_lps
        elif other._current_tokens is not None:
            self._current_tokens = mx.concatenate(
                [self._current_tokens, other._current_tokens]
            )
            if self._current_lps is not None and other._current_lps is not None:
                self._current_lps = mx.concatenate(
                    [self._current_lps, other._current_lps]
                )

        if self._next_tokens is None:
            self._next_tokens = other._next_tokens
            self.next_lps = other.next_lps
            self.next_top_idx = other.next_top_idx
            self.next_top_lp = other.next_top_lp
        elif other._next_tokens is not None:
            self._next_tokens = mx.concatenate([self._next_tokens, other._next_tokens])
            if self.next_lps is not None and other.next_lps is not None:
                self.next_lps = mx.concatenate([self.next_lps, other.next_lps])

            if (
                self.next_top_idx is not None
                and other.next_top_idx is not None
                and self.next_top_idx.shape[-1] == other.next_top_idx.shape[-1]
            ):
                self.next_top_idx = mx.concatenate(
                    [self.next_top_idx, other.next_top_idx]
                )
                assert self.next_top_lp is not None and other.next_top_lp is not None
                self.next_top_lp = mx.concatenate([self.next_top_lp, other.next_top_lp])
            else:
                self.next_top_idx = None
                self.next_top_lp = None

    def filter(self, keep: List[int]):
        """Filter the batch to keep only the specified indices."""
        if len(keep) < len(self.uids):
            self._eval_pending_state()

        self.uids = [self.uids[idx] for idx in keep]
        self.max_tokens = [self.max_tokens[idx] for idx in keep]
        self._num_tokens = [self._num_tokens[idx] for idx in keep]
        if self.token_context:
            self.token_context = [self.token_context[idx] for idx in keep]
        if self.logits_processors:
            self.logits_processors = [self.logits_processors[idx] for idx in keep]
        if self.thinking_budget_criteria:
            self.thinking_budget_criteria = [
                self.thinking_budget_criteria[idx] for idx in keep
            ]

        if not keep:
            self.prompt_cache.clear()
            self._current_tokens = None
            self._current_lps = None
            self._next_tokens = None
            self.next_lps = None
            self.next_top_idx = None
            self.next_top_lp = None
            self.token_context = []
            self.logits_processors = []
            self.thinking_budget_criteria = []
        else:
            keep_arr = mx.array(keep, mx.int32)
            for c in self.prompt_cache:
                filter_rows = getattr(c, "filter")
                assert callable(filter_rows)
                filter_rows(keep_arr)
            if self._next_tokens is not None:
                self._next_tokens = self._next_tokens[keep_arr]
            if self.next_lps is not None:
                self.next_lps = self.next_lps[keep_arr]
            if self.next_top_idx is not None:
                self.next_top_idx = self.next_top_idx[keep_arr]
                assert self.next_top_lp is not None
                self.next_top_lp = self.next_top_lp[keep_arr]

    def next(self) -> List[Response]:
        """Generate the next batch of tokens."""
        if not self.uids:
            return []

        tokens, lp_list, top_idx_list, top_lp_list = self._step()

        keep: list[int] = []
        responses: list[GenerationBatch.Response] = []
        forced_next_tokens: list[int | None] = [None] * len(self.uids)
        for i in range(len(self.uids)):
            finish_reason = None
            self._num_tokens[i] += 1
            tok = tokens[i]
            if (
                i < len(self.thinking_budget_criteria)
                and self.thinking_budget_criteria[i] is not None
            ):
                criteria = self.thinking_budget_criteria[i]
                assert criteria is not None
                criteria(tok)
                forced_next_tokens[i] = criteria.pop_forced_token_id()

            if self.stop_criteria(tok):
                finish_reason = "stop"
            elif self._num_tokens[i] >= self.max_tokens[i]:
                finish_reason = "length"

            if finish_reason is None:
                keep.append(i)

            top_lp = None
            if top_idx_list is not None and top_lp_list is not None:
                top_lp = list(zip(top_idx_list[i], top_lp_list[i]))

            responses.append(
                self.Response(
                    uid=self.uids[i],
                    token=tok,
                    token_logprob=lp_list[i] if lp_list is not None else 0.0,
                    finish_reason=finish_reason,
                    top_logprobs=top_lp,
                )
            )

        has_forced_next_tokens = any(token is not None for token in forced_next_tokens)
        if has_forced_next_tokens:
            assert self._next_tokens is not None
            force_mask = mx.array(
                [token is not None for token in forced_next_tokens], dtype=mx.bool_
            )
            replacements = mx.array(
                [token if token is not None else 0 for token in forced_next_tokens],
                dtype=self._next_tokens.dtype,
            )
            self._next_tokens = mx.where(force_mask, replacements, self._next_tokens)

        if has_forced_next_tokens:
            async_evaluate(self._next_tokens)

        if len(keep) < len(self.uids):
            self.filter(keep)

        return responses

    @classmethod
    def empty(
        cls,
        model: LanguageModelLike | GenerationModelLike,
        sampler: Callable[[mx.array], mx.array],
        stop_criteria: Callable[[int], bool],
        compute_logprobs: bool = True,
        top_logprobs_k: int = 0,
        greedy_sampling: bool = False,
    ):
        """Create an empty generation batch."""
        batch = cls.__new__(cls)
        batch.model = model
        batch._language_model = getattr(model, "language_model", model)
        batch.uids = []
        batch.prompt_cache = []
        batch.sampler = sampler
        batch.stop_criteria = stop_criteria
        batch.max_tokens = []
        batch._num_tokens = []
        batch.compute_logprobs = compute_logprobs
        batch.top_logprobs_k = top_logprobs_k
        batch.greedy_sampling = greedy_sampling
        batch.token_context = []
        batch.logits_processors = []
        batch.thinking_budget_criteria = []
        batch._current_tokens = None
        batch._current_lps = None
        batch._next_tokens = None
        batch.next_lps = None
        batch.next_top_idx = None
        batch.next_top_lp = None
        return batch


class PromptProcessingBatch:
    """
    Handles VLM prompt processing with inputs_embeds and chunked prefill.

    Processes prompt tokens incrementally (one chunk per step) to allow
    interleaving with generation for continuous batching. Transitions to
    a GenerationBatch when prompt processing is complete.
    """

    prompt_cache: list[CacheEntry]

    def __init__(
        self,
        model: LanguageModelLike | GenerationModelLike,
        uids: List[int],
        input_ids: List[List[int]],
        max_tokens: List[int],
        inputs_embeds: mx.array,
        prompt_kwargs: PromptInputs,
        logits_processors: Optional[
            List[Optional[List[Callable[[mx.array, mx.array], mx.array]]]]
        ] = None,
        thinking_budget_criteria: Optional[
            List[Optional[ThinkingBudgetCriteria]]
        ] = None,
        prefill_step_size: Optional[int] = DEFAULT_PREFILL_STEP_SIZE,
        kv_bits: float | None = None,
        kv_group_size: int = DEFAULT_KV_GROUP_SIZE,
        greedy_sampling: bool = False,
    ):
        """Initialize left-padded prompt processing for one prefill group.

        ``input_ids`` and ``inputs_embeds`` describe the same prompts before
        padding; shorter rows are left-padded to the longest prompt so their
        cache offsets match the generated suffix.
        """
        self.model = model
        self.uids = uids
        self._prompt_uids = list(uids)
        self.max_tokens = max_tokens
        self.prefill_step_size = prefill_step_size
        self.greedy_sampling = greedy_sampling

        lengths = [len(ids) for ids in input_ids]
        max_length = max(lengths)
        left_padding = [max_length - l for l in lengths]
        self._input_ids = _left_pad_prompts(input_ids, max_length=max_length)
        self._total_prompt_tokens = sum(lengths)
        self._processed_prompt_columns: int = 0

        self.logits_processors = logits_processors or []
        self.thinking_budget_criteria = thinking_budget_criteria or []
        self._token_context = (
            [list(ids) for ids in input_ids]
            if self.logits_processors and any(self.logits_processors)
            else []
        )
        self._inputs_embeds = inputs_embeds
        self._prompt_kwargs = prompt_kwargs or {}
        self._prompt_length_aware_keys: List[str] = []
        if self._prompt_kwargs:
            prompt_batch = self._inputs_embeds.shape[0]
            prompt_len = self._inputs_embeds.shape[1]
            for k, v in self._prompt_kwargs.items():
                if (
                    isinstance(v, mx.array)
                    and _prompt_kwarg_batch_size(k, v) == prompt_batch
                    and _is_sequence_aligned_prompt_kwarg(k, v, prompt_len)
                ):
                    self._prompt_length_aware_keys.append(k)

        self._prompt_time_s = 0.0
        self._prompt_tokens_per_row = list(lengths)

        if len(input_ids) == 1 and kv_bits is None and hasattr(model, "make_cache"):
            created_cache = cache.make_prompt_cache(generation_cache_model(model))
            if not is_cache_entry_list(created_cache):
                raise TypeError(
                    "Prompt processing requires cache entries exposing state and metadata"
                )
            self.prompt_cache = created_cache
        else:
            self.prompt_cache = _make_cache(
                model,
                left_padding,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
            )

        self.prefill_step_size = default_prefill_step_size_for_offload(
            self.model, self.prefill_step_size, DEFAULT_PREFILL_STEP_SIZE
        )
        if self.prefill_step_size is not None:
            policy_kwargs = dict(self._prompt_kwargs)
            if not chunked_prefill_enabled(
                self.model,
                input_ids=self._input_ids,
                inputs_embeds=self._inputs_embeds,
                prompt_cache=self.prompt_cache,
                prefill_kwargs=policy_kwargs,
            ):
                self.prefill_step_size = None

    def __len__(self):
        """Return the number of prompts in this prefill group."""
        return len(self.uids)

    def needs_processing(self):
        """True if prompt needs chunked processing before generate()."""
        if self.prefill_step_size is None:
            return False
        # See the note in stream_generate: prompts at or below prefill_step_size
        # would otherwise skip chunking and materialize the full logits tensor.
        return self._inputs_embeds.shape[1] > 1

    def _prompt_kwargs_for_step(self, n: Optional[int] = None) -> PromptInputs:
        if n is None or not self._prompt_length_aware_keys:
            return self._prompt_kwargs
        out = dict(self._prompt_kwargs)
        for k in self._prompt_length_aware_keys:
            value = out[k]
            assert isinstance(value, mx.array)
            out[k] = _slice_sequence_aligned_prompt_kwarg(k, value, stop=n)
        return out

    def prompt_step(self) -> int:
        """Process one chunk of the prompt. Returns tokens processed."""
        if not self.needs_processing():
            return 0

        step = self.prefill_step_size or self._inputs_embeds.shape[1]
        n = min(step, self._inputs_embeds.shape[1] - 1)
        if n <= 0:
            return 0
        prompt_kwargs = {
            **self._prompt_kwargs_for_step(n),
        }
        self.model(
            self._input_ids[:, :n],
            cache=self.prompt_cache,
            inputs_embeds=self._inputs_embeds[:, :n],
            n_to_process=n,
            **prompt_kwargs,
        )
        async_evaluate([c.state for c in self.prompt_cache])
        self._inputs_embeds = self._inputs_embeds[:, n:]
        self._input_ids = self._input_ids[:, n:]
        for k in self._prompt_length_aware_keys:
            value = self._prompt_kwargs[k]
            assert isinstance(value, mx.array)
            self._prompt_kwargs[k] = _slice_sequence_aligned_prompt_kwarg(
                k, value, start=n
            )
        mx.clear_cache()
        return n

    def record_prompt_time(self, elapsed_s: float) -> None:
        """Accumulate elapsed prompt-processing time in seconds."""
        self._prompt_time_s += max(0.0, float(elapsed_s))

    def prompt_progress(self) -> List[PromptProgress]:
        """Return per-request prompt token counts and throughput.

        Rows report their own unpadded token count against the accumulated
        batch processing time, so an empty list means no time has been recorded.
        """
        if self._prompt_time_s <= 0:
            return []
        return [
            PromptProgress(
                uid=uid,
                prompt_tokens=prompt_tokens,
                prompt_tps=prompt_tokens / self._prompt_time_s,
                prompt_time=self._prompt_time_s,
            )
            for uid, prompt_tokens in zip(
                self._prompt_uids, self._prompt_tokens_per_row
            )
        ]

    def generate(
        self,
        sampler: Callable[[mx.array], mx.array],
        stop_criteria: Callable[[int], bool],
        compute_logprobs: bool = True,
        top_logprobs_k: int = 0,
    ) -> GenerationBatch:
        """Process final tokens and transition to GenerationBatch."""
        call_kwargs = dict(self._prompt_kwargs)

        call_kwargs["logits_to_keep"] = 1

        output = self.model(
            self._input_ids,
            cache=self.prompt_cache,
            inputs_embeds=self._inputs_embeds,
            **call_kwargs,
        )
        logits = output if isinstance(output, mx.array) else output.logits
        logits = logits[:, -1, :]
        if self.logits_processors and any(self.logits_processors):
            processed_logits: list[mx.array] = []
            for i in range(logits.shape[0]):
                sample_logits = logits[i : i + 1]
                processors = self.logits_processors[i] or []
                for processor in processors:
                    sample_logits = processor(
                        mx.array(self._token_context[i]), sample_logits
                    )
                processed_logits.append(sample_logits)
            logits = mx.concatenate(processed_logits, axis=0)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        first_tokens = _sample_with_positions(
            sampler,
            logprobs,
            row_ids=[0] * len(self.uids),
            positions=[0] * len(self.uids),
        )

        async_evaluate(first_tokens)

        gen_batch = GenerationBatch(
            model=self.model,
            uids=list(self.uids),
            inputs=first_tokens,
            prompt_cache=self.prompt_cache,
            sampler=sampler,
            stop_criteria=stop_criteria,
            max_tokens=list(self.max_tokens),
            top_logprobs_k=top_logprobs_k,
            greedy_sampling=self.greedy_sampling,
            token_context=[list(ctx) for ctx in self._token_context],
            logits_processors=list(self.logits_processors),
            thinking_budget_criteria=list(self.thinking_budget_criteria),
        )
        gen_batch.compute_logprobs = compute_logprobs

        if compute_logprobs:
            gen_batch.next_lps = logprobs[
                mx.arange(first_tokens.shape[0]), first_tokens
            ]

        # Prime top-K buffers so the first token can emit top_logprobs too.
        if top_logprobs_k > 0:
            k = top_logprobs_k
            sort_idx = mx.argsort(logprobs, axis=-1)
            top_idx = sort_idx[..., -k:][..., ::-1].astype(mx.int32)
            top_lp = mx.take_along_axis(logprobs, top_idx, axis=-1)
            gen_batch.next_top_idx = top_idx
            gen_batch.next_top_lp = top_lp

        # Final prefill produces the first generated token and mutates the
        # prompt cache. Materialize that boundary before the decode loop so
        # the first decode step does not inherit the full lazy prefill graph.
        cache_states: list[StateView] = []
        for c in self.prompt_cache:
            try:
                cache_states.append(c.state)
            except (AttributeError, TypeError):
                pass
        eval_targets: list[StateView] = [first_tokens]
        if cache_states:
            eval_targets.append(cache_states)
        if compute_logprobs:
            eval_targets.append(gen_batch.next_lps)
        if top_logprobs_k > 0:
            eval_targets.extend([gen_batch.next_top_idx, gen_batch.next_top_lp])
        evaluate(*eval_targets)

        self.uids = []
        self.prompt_cache = []
        self._token_context = []
        self.logits_processors = []
        return gen_batch

    @property
    def total_prompt_tokens(self):
        """Return the number of unpadded prompt tokens in this group."""
        return self._total_prompt_tokens


class BatchGenerator:
    """
    Continuous batching with separate prompt processing and generation phases.

    next() returns (prompt_responses, generation_responses) where:
    - prompt_responses contains completed prompt-batch timing stats
    - generation_responses is a list of GenerationBatch.Response objects
    """

    def __init__(
        self,
        model: LanguageModelLike | GenerationModelLike,
        processor: ProcessorLike | TokenizerLike,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stop_tokens: Optional[set[int]] = None,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        completion_batch_size: int = DEFAULT_COMPLETION_BATCH_SIZE,
        prefill_batch_size: int = DEFAULT_PREFILL_BATCH_SIZE,
        prefill_step_size: Optional[int] = DEFAULT_PREFILL_STEP_SIZE,
        prompt_cache: list[CacheEntry] | None = None,
        kv_bits: float | None = None,
        kv_group_size: int = DEFAULT_KV_GROUP_SIZE,
        quantized_kv_start: int = DEFAULT_QUANTIZED_KV_START,
        compute_logprobs: bool = True,
        top_logprobs_k: int = 0,
        logits_processors: Optional[
            List[Callable[[mx.array, mx.array], mx.array]]
        ] = None,
        stream: mx.Stream | mx.ThreadLocalStream | None = None,
        greedy_sampling: bool = False,
    ):
        """Configure continuous batching over a generation model and processor.

        ``stop_tokens`` are added to the tokenizer's stopping criteria. Without a
        ``sampler`` the generator uses greedy argmax decoding. ``prefill_step_size``
        may be disabled by the model's chunked-prefill policy, and prompt and
        generation batches are capped by ``prefill_batch_size`` and
        ``completion_batch_size``.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.processor = processor
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.quantized_kv_start = quantized_kv_start
        self.compute_logprobs = compute_logprobs
        self.top_logprobs_k = top_logprobs_k
        self.logits_processors = logits_processors or []
        self.greedy_sampling = greedy_sampling or sampler is None
        self.tokenizer = resolve_tokenizer(processor)
        self.sampler: Callable[[mx.array], mx.array] = sampler or (
            lambda x: mx.argmax(x, axis=-1)
        )
        self.uid_count = 0
        self.prefill_step_size = prefill_step_size
        self.prefill_batch_size = prefill_batch_size
        self.completion_batch_size = completion_batch_size

        self._stream = stream or generation_stream

        self.tokenizer.stopping_criteria.add_eos_token_ids(stop_tokens)

        self._generation_batch: GenerationBatch = GenerationBatch.empty(
            self.model,
            self.sampler,
            self.tokenizer.stopping_criteria,
            compute_logprobs=self.compute_logprobs,
            top_logprobs_k=self.top_logprobs_k,
            greedy_sampling=self.greedy_sampling,
        )
        self._prompt_batch: Optional[PromptProcessingBatch] = None
        self._unprocessed_sequences: list[PendingSequence] = []

        self._prompt_tokens_counter = 0
        self._prompt_time_counter: float = 0
        self._gen_tokens_counter = 0
        self._steps_counter = 0
        self._cache_eval_interval = _get_batch_cache_eval_interval()

        self._wire_stack = contextlib.ExitStack()
        self._wire_stack.enter_context(wired_limit(model, [self._stream]))

    @property
    def stream(self):
        """Return the MLX stream that owns this generator's work."""
        return self._stream

    def close(self):
        """Release the wired-memory limit held for this generator."""
        if self._wire_stack is not None:
            self._wire_stack.close()
            self._wire_stack = None

    def __del__(self):
        """Close the generator when it is collected."""
        self.close()

    def insert(
        self,
        prompts: mx.array | list[list[int]],
        max_tokens: Union[List[int], int, None] = None,
        prompt_kwargs: Optional[List[PromptInputs]] = None,
        logits_processors: Optional[
            List[Optional[List[Callable[[mx.array, mx.array], mx.array]]]]
        ] = None,
        thinking_budget_criteria: Optional[
            List[Optional[ThinkingBudgetCriteria]]
        ] = None,
        attention_mask: Optional[mx.array] = None,
    ):
        """Queue prompts, consuming tokenizer padding metadata when provided.

        A dense ``prompts`` array may include tokenizer padding when accompanied
        by a same-shaped ``attention_mask``. The padded columns are removed from
        tokens and sequence-aligned prompt tensors before anything is queued.
        """
        uids: list[int] = []

        if max_tokens is None or isinstance(max_tokens, int):
            max_tokens = [max_tokens or self.max_tokens] * len(prompts)

        if prompt_kwargs is None:
            prompt_kwargs = [{} for _ in range(len(prompts))]
        prompts, prompt_kwargs = _unpad_batch_prompts(
            prompts, attention_mask, prompt_kwargs
        )
        if logits_processors is None:
            logits_processors = [self.logits_processors for _ in prompts]
        elif len(logits_processors) != len(prompts):
            raise ValueError("Insufficient number of logits_processors provided")
        if thinking_budget_criteria is None:
            thinking_budget_criteria = [None for _ in prompts]
        elif len(thinking_budget_criteria) != len(prompts):
            raise ValueError("Insufficient number of thinking_budget_criteria provided")

        assert logits_processors is not None
        assert thinking_budget_criteria is not None
        for p, m, kw, lp, tc in zip(
            prompts,
            max_tokens,
            prompt_kwargs,
            logits_processors,
            thinking_budget_criteria,
        ):
            self._unprocessed_sequences.append((self.uid_count, p, m, kw, lp, tc))
            uids.append(self.uid_count)
            self.uid_count += 1
        # Sort in ascending order of length
        self._unprocessed_sequences = sorted(
            self._unprocessed_sequences, key=lambda x: len(x[1])
        )
        return uids

    def remove(self, uid: int) -> bool:
        """Remove a sequence from the batch by uid."""
        with mx.stream(self._stream):
            # Waiting in the queue.
            for i, (seq_uid, _, _, _, _, _) in enumerate(self._unprocessed_sequences):
                if seq_uid == uid:
                    self._unprocessed_sequences.pop(i)
                    return True

            # Being prefilled
            if self._prompt_batch is not None and uid in self._prompt_batch.uids:
                if len(self._prompt_batch.uids) == 1:
                    self._prompt_batch.uids = []
                    self._prompt_batch.prompt_cache = []
                    self._prompt_batch = None
                    mx.clear_cache()
                    return True

            # Already decoding.
            if uid in self._generation_batch.uids:
                idx = self._generation_batch.uids.index(uid)
                keep = [i for i in range(len(self._generation_batch.uids)) if i != idx]
                self._generation_batch.filter(keep)
                return True

            return False

    @property
    def unprocessed_prompts(self):
        """Backward-compatible alias for server flush logic."""
        return self._unprocessed_sequences

    @property
    def has_pending_prompts(self):
        """True if there are prompts waiting or being processed."""
        return len(self._unprocessed_sequences) > 0 or self._prompt_batch is not None

    @property
    def has_work(self):
        """True if there is any remaining work."""
        return (
            len(self._generation_batch) > 0
            or self._prompt_batch is not None
            or len(self._unprocessed_sequences) > 0
        )

    def stats(self):
        """Return accumulated batch statistics."""
        stats = BatchStats()
        stats.prompt_tokens = self._prompt_tokens_counter
        stats.prompt_time = self._prompt_time_counter
        stats.prompt_tps = (
            self._prompt_tokens_counter / self._prompt_time_counter
            if self._prompt_time_counter > 0
            else 0
        )
        stats.generation_tokens = self._gen_tokens_counter
        stats.peak_memory = mx.get_peak_memory() / 1e9
        return stats

    @staticmethod
    def _record_prompt_batch_time(
        prompt_batch: PromptProcessingBatch, elapsed_s: float
    ) -> None:
        recorder = getattr(prompt_batch, "record_prompt_time", None)
        if callable(recorder):
            recorder(elapsed_s)

    @staticmethod
    def _prompt_batch_progress(
        prompt_batch: PromptProcessingBatch,
    ) -> List[PromptProgress]:
        progress = getattr(prompt_batch, "prompt_progress", None)
        if callable(progress):
            result = progress()
            if not isinstance(result, list) or not all(
                isinstance(item, PromptProgress) for item in cast(list[object], result)
            ):
                raise TypeError(
                    "Prompt progress must be a list of PromptProgress records"
                )
            return cast(list[PromptProgress], result)
        return []

    def _extend_generation_batch(self, gen_batch: GenerationBatch) -> None:
        if len(self._generation_batch) == 0:
            self._generation_batch = gen_batch
        else:
            self._generation_batch.extend(gen_batch)

    def _next(self, **kwargs: ModelArgument):
        generation_responses: list[GenerationBatch.Response] = []
        prompt_responses: list[PromptProgress] = []

        # Decode-first: always emit a generation step before touching prefill.
        if len(self._generation_batch) > 0:
            generation_responses = self._generation_batch.next()
            self._gen_tokens_counter += len(generation_responses)
            self._steps_counter += 1
            if (
                self._cache_eval_interval > 0
                and self._steps_counter % self._cache_eval_interval == 0
            ):
                cache_states = getattr(self._generation_batch, "cache_states", None)
                if callable(cache_states):
                    evaluate(cache_states())
                else:
                    evaluate([c.state for c in self._generation_batch.prompt_cache])
                mx.clear_cache()

        if len(self._generation_batch) >= self.completion_batch_size:
            return prompt_responses, generation_responses

        if self._prompt_batch is not None:
            if self._prompt_batch.needs_processing():
                tic = time.perf_counter()
                self._prompt_batch.prompt_step()
                elapsed = time.perf_counter() - tic
                self._prompt_time_counter += elapsed
                self._record_prompt_batch_time(self._prompt_batch, elapsed)
                return prompt_responses, generation_responses

            tic = time.perf_counter()
            gen_batch = self._prompt_batch.generate(
                self.sampler,
                self.tokenizer.stopping_criteria,
                compute_logprobs=self.compute_logprobs,
                top_logprobs_k=self.top_logprobs_k,
            )
            elapsed = time.perf_counter() - tic
            self._prompt_time_counter += elapsed
            self._record_prompt_batch_time(self._prompt_batch, elapsed)
            prompt_responses = self._prompt_batch_progress(self._prompt_batch)
            self._extend_generation_batch(gen_batch)
            self._prompt_batch = None
            mx.clear_cache()
            return prompt_responses, generation_responses

        num_active = len(self._generation_batch)
        num_to_add = self.completion_batch_size - num_active
        if self._unprocessed_sequences and num_to_add >= self.prefill_batch_size:
            n = min(self.prefill_batch_size, len(self._unprocessed_sequences))
            sequences = self._unprocessed_sequences[:n]
            self._unprocessed_sequences = self._unprocessed_sequences[n:]

            uids = [s[0] for s in sequences]
            input_ids = [s[1] for s in sequences]
            max_tokens_list = [s[2] for s in sequences]
            prompt_kwargs_list = [s[3] for s in sequences]
            logits_processors = [s[4] for s in sequences]
            thinking_budget_criteria = [s[5] for s in sequences]

            inputs_embeds, merged_kwargs = _merge_prefill_prompt_kwargs(
                prompt_kwargs_list, input_ids
            )

            prompt_batch_cls = PromptProcessingBatch
            self._prompt_batch = prompt_batch_cls(
                model=self.model,
                uids=uids,
                input_ids=input_ids,
                max_tokens=max_tokens_list,
                inputs_embeds=inputs_embeds,
                prompt_kwargs=merged_kwargs,
                logits_processors=logits_processors,
                thinking_budget_criteria=thinking_budget_criteria,
                prefill_step_size=self.prefill_step_size,
                kv_bits=self.kv_bits,
                kv_group_size=self.kv_group_size,
                greedy_sampling=getattr(self, "greedy_sampling", False),
            )
            self._prompt_tokens_counter += self._prompt_batch.total_prompt_tokens

            if self._prompt_batch.needs_processing():
                tic = time.perf_counter()
                n = self._prompt_batch.prompt_step()
                elapsed = time.perf_counter() - tic
                self._prompt_time_counter += elapsed
                self._record_prompt_batch_time(self._prompt_batch, elapsed)
            else:
                tic = time.perf_counter()
                gen_batch = self._prompt_batch.generate(
                    self.sampler,
                    self.tokenizer.stopping_criteria,
                    compute_logprobs=self.compute_logprobs,
                    top_logprobs_k=self.top_logprobs_k,
                )
                elapsed = time.perf_counter() - tic
                self._prompt_time_counter += elapsed
                self._record_prompt_batch_time(self._prompt_batch, elapsed)
                prompt_responses = self._prompt_batch_progress(self._prompt_batch)
                self._extend_generation_batch(gen_batch)
                self._prompt_batch = None
                mx.clear_cache()

            return prompt_responses, generation_responses

        return prompt_responses, generation_responses

    def next(self, **kwargs: ModelArgument):
        """Run one scheduler step on this generator's stream.

        Returns prompt-progress records for a finished prompt batch together
        with one response per decoding sequence.
        """
        with mx.stream(self._stream):
            return self._next(**kwargs)


class BatchGeneratorOptions(TypedDict, total=False, closed=True):
    """Options accepted by :func:`batch_generate` and ``BatchGenerator``."""

    max_tokens: int
    stop_tokens: Optional[set[int]]
    sampler: Optional[Callable[[mx.array], mx.array]]
    completion_batch_size: int
    prefill_batch_size: int
    prefill_step_size: Optional[int]
    prompt_cache: list[CacheEntry] | None
    kv_bits: float | None
    kv_group_size: int
    quantized_kv_start: int
    compute_logprobs: bool
    top_logprobs_k: int
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]]
    stream: mx.Stream | mx.ThreadLocalStream | None
    greedy_sampling: bool


def batch_generate(
    model: GenerationModelLike,
    processor: ProcessorLike,
    images: Union[str, List[str], None] = None,
    prompts: Optional[List[str]] = None,
    max_tokens: Union[int, List[int]] = 128,
    verbose: bool = False,
    group_by_shape: bool = True,
    track_image_sizes: bool = True,
    **kwargs: Unpack[BatchGenerationOptions],
) -> BatchResponse:
    """
    Generate responses for the given batch of prompts with variable-sized images.

    This function implements the transformers-style approach to batching:
    1. Group images with the same shape for efficient batch processing
    2. Process each group as a batch (no padding waste within groups)
    3. Track original image sizes for proper attention masking
    4. Restore results to original batch order

    Key insight: Instead of padding all images to the same spatial dimensions
    (which wastes computation and may hurt accuracy), we group same-sized
    images together so there's zero padding within each group.

    Args:
       model (nn.Module): The language model.
       processor (PreTrainedTokenizer): The tokenizer/processor.
       images (Union[str, List[str]]): Images (paths, URLs, or PIL images).
       prompts (List[str]): The input prompts.
       max_tokens (Union[int, List[int]]): Maximum number of output tokens. This
          can be per prompt if a list is provided.
       verbose (bool): If ``True``, print tokens and timing information.
       group_by_shape (bool): If ``True``, group same-shaped images for efficient
          batch processing.
       track_image_sizes (bool): If ``True``, track and return original image sizes.
       kwargs: The remaining options get passed to :obj:`BatchGenerator`.
          See :obj:`BatchGenerator` for more details.

    Returns:
        BatchResponse with generated texts, statistics, and optionally image_sizes.
    """
    from PIL import Image

    from mlx_vlm.utils import process_image

    processor.detokenizer.reset()
    if prompts is None:
        raise TypeError("object of type 'NoneType' has no len()")

    # Handle single image case
    if isinstance(images, str):
        images = [images]

    # Handle no images case
    if images is None:
        texts, stats = _generate_batch(
            model, processor, prompts, None, max_tokens, verbose, **kwargs
        )
        return BatchResponse(texts, stats)

    # Load and preprocess images
    image_processor = getattr(processor, "image_processor", None)

    processed_images: list[Image.Image] = []
    image_sizes_original: list[tuple[int, int]] = []
    for img in images:
        pil_img = process_image(img, None, image_processor)
        processed_images.append(pil_img)
        # Track original size
        if hasattr(pil_img, "height"):
            image_sizes_original.append((pil_img.height, pil_img.width))
        else:
            image_sizes_original.append((0, 0))

    # Group images by shape for efficient processing (no padding within groups)
    if group_by_shape and len(processed_images) > 1:
        grouped_images, grouped_indices = group_images_by_shape(processed_images)

        if verbose:
            print(f"[batch_generate] Found {len(grouped_images)} unique image shapes")
    else:
        # Single image or grouping disabled - treat as one group
        shape = (
            (processed_images[0].height, processed_images[0].width)
            if processed_images
            else (0, 0)
        )
        grouped_images = {shape: processed_images}
        grouped_indices = {shape: list(range(len(processed_images)))}

    # Process each shape group
    all_texts: list[str | None] = [None] * len(prompts)
    all_image_sizes: list[tuple[int, int] | None] = [None] * len(prompts)
    total_stats = BatchStats()

    for shape, indices in grouped_indices.items():
        # Get images and prompts for this shape group
        group_images = [processed_images[i] for i in indices]
        group_prompts = [prompts[i] for i in indices]
        group_sizes = [image_sizes_original[i] for i in indices]

        # Handle per-sample max_tokens
        if isinstance(max_tokens, list):
            group_max_tokens = [max_tokens[i] for i in indices]
        else:
            group_max_tokens = max_tokens

        group_kwargs: BatchGenerationOptions = {**kwargs}
        logits_processors = group_kwargs.get("logits_processors")
        if logits_processors is not None:
            if is_logits_processor_list(logits_processors):
                group_kwargs["logits_processors"] = logits_processors
            else:
                group_kwargs["logits_processors"] = [
                    logits_processors[i] for i in indices
                ]

        # Process the entire group at once (same shape = no padding needed)
        chunk_texts, chunk_stats = _generate_batch(
            model,
            processor,
            group_prompts,
            group_images,
            group_max_tokens,
            **group_kwargs,
        )

        # Store results in original order
        for j, orig_idx in enumerate(indices):
            all_texts[orig_idx] = chunk_texts[j]
            all_image_sizes[orig_idx] = group_sizes[j]

        # Accumulate stats
        total_stats.prompt_tokens += chunk_stats.prompt_tokens
        total_stats.prompt_time += chunk_stats.prompt_time
        total_stats.generation_tokens += chunk_stats.generation_tokens
        total_stats.generation_time += chunk_stats.generation_time

    text_only_indices = list(range(len(processed_images), len(prompts)))
    if text_only_indices:
        group_prompts = [prompts[i] for i in text_only_indices]
        if isinstance(max_tokens, list):
            group_max_tokens = [max_tokens[i] for i in text_only_indices]
        else:
            group_max_tokens = max_tokens

        group_kwargs: BatchGenerationOptions = {**kwargs}
        logits_processors = group_kwargs.get("logits_processors")
        if logits_processors is not None:
            if is_logits_processor_list(logits_processors):
                group_kwargs["logits_processors"] = logits_processors
            else:
                group_kwargs["logits_processors"] = [
                    logits_processors[i] for i in text_only_indices
                ]

        chunk_texts, chunk_stats = _generate_batch(
            model,
            processor,
            group_prompts,
            None,
            group_max_tokens,
            **group_kwargs,
        )

        for j, orig_idx in enumerate(text_only_indices):
            all_texts[orig_idx] = chunk_texts[j]

        total_stats.prompt_tokens += chunk_stats.prompt_tokens
        total_stats.prompt_time += chunk_stats.prompt_time
        total_stats.generation_tokens += chunk_stats.generation_tokens
        total_stats.generation_time += chunk_stats.generation_time

    mx.clear_cache()

    # Compute final stats
    if total_stats.prompt_time > 0:
        total_stats.prompt_tps = total_stats.prompt_tokens / total_stats.prompt_time
    if total_stats.generation_time > 0:
        total_stats.generation_tps = (
            total_stats.generation_tokens / total_stats.generation_time
        )
    total_stats.peak_memory = mx.get_peak_memory() / 1e9

    if verbose:
        print(f"[batch_generate] Finished processing {len(prompts)} samples")
        print(
            f"[batch_generate] Prompt: {total_stats.prompt_tokens} tokens, {total_stats.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"[batch_generate] Generation: {total_stats.generation_tokens} tokens, "
            f"{total_stats.generation_tps:.3f} tokens-per-sec"
        )
        print(f"[batch_generate] Peak memory: {total_stats.peak_memory:.3f} GB")

    assert is_str_list(all_texts)
    response = BatchResponse(all_texts, total_stats)
    if track_image_sizes:
        response.image_sizes = all_image_sizes
    return response


def _clone_or_share_logits_processor(
    processor: Callable[[mx.array, mx.array], mx.array],
) -> Callable[[mx.array, mx.array], mx.array]:
    if hasattr(processor, "clone"):
        clone = getattr(processor, "clone")
        assert callable(clone)
        # User callback factories preserve the callback's two-array contract.
        cloned = clone()
        if not callable(cloned):
            raise TypeError("Logits processor clone() must return a callable")
        return cast(Callable[[mx.array, mx.array], mx.array], cloned)
    warnings.warn(
        "Sharing logits processor across batch entries because it does not "
        "implement clone(). Stateful logits processors should implement clone() "
        "to avoid shared state across sequences.",
        RuntimeWarning,
        stacklevel=2,
    )
    return processor


def _generate_batch[ImageT: ImageSource](
    model: GenerationModelLike,
    processor: ProcessorLike | TokenizerLike,
    prompts: List[str],
    images: list[ImageT] | None = None,
    max_tokens: Union[int, List[int]] = 100,
    verbose: bool = False,
    **kwargs: Unpack[BatchGenerationOptions],
) -> Tuple[List[str], BatchStats]:

    batch_size = len(prompts)
    logits_processors: (
        list[Callable[[mx.array, mx.array], mx.array]]
        | list[list[Callable[[mx.array, mx.array], mx.array]] | None]
        | None
    ) = kwargs.pop("logits_processors", None)

    num_images_list = [
        1 if i < (len(images) if images is not None else 0) else 0
        for i in range(len(prompts))
    ]
    formatted_prompts = [
        apply_chat_template(
            processor,
            model.config,
            prompt,
            num_images=num_images_list[i],
        )
        for i, prompt in enumerate(prompts)
    ]

    if not all(isinstance(value, str) for value in formatted_prompts):
        raise TypeError("Batch chat templates must return text prompts")
    formatted_prompts = cast(list[str], formatted_prompts)
    add_special_tokens = should_add_special_tokens(model.config.model_type, processor)

    resize_shape = normalize_resize_shape(kwargs.pop("resize_shape", None))
    image_token_index = getattr(model.config, "image_token_index", None)

    inputs = prepare_inputs(
        processor,
        images=images,
        prompts=formatted_prompts,
        image_token_index=image_token_index,
        resize_shape=resize_shape,
        add_special_tokens=add_special_tokens,
        pad_to_uniform_size=False,  # Since images are pre-grouped by shape, they're already uniform size
    )
    input_ids = inputs.get("input_ids", None)
    pixel_values = inputs.get("pixel_values", None)
    mask = inputs.get("attention_mask", None)

    data_kwargs: PromptInputs = {
        k: v
        for k, v in inputs.items()
        if k not in ["input_ids", "pixel_values", "attention_mask"]
    }

    if not isinstance(input_ids, mx.array):
        raise TypeError("Batch input_ids must be an MLX array")
    if not is_pixel_values(pixel_values):
        raise TypeError("Batch pixel_values must contain supported tensors")
    if mask is not None and not isinstance(mask, mx.array):
        raise TypeError("Batch attention_mask must be an MLX array")
    embedding_output = model.get_input_embeddings(
        input_ids, pixel_values, mask=mask, **data_kwargs
    )

    gen_kwargs: PromptInputs = {
        **data_kwargs,
        **{k: v for k, v in embedding_output.to_dict().items() if v is not None},
    }

    kwargs["prefill_step_size"] = default_prefill_step_size_for_offload(
        model,
        kwargs.get("prefill_step_size", DEFAULT_PREFILL_STEP_SIZE),
        DEFAULT_PREFILL_STEP_SIZE,
    )
    if kwargs.get("prefill_step_size", DEFAULT_PREFILL_STEP_SIZE) is not None:
        policy_kwargs = dict(gen_kwargs)
        if not chunked_prefill_enabled(
            model,
            input_ids=input_ids,
            inputs_embeds=embedding_output.inputs_embeds,
            prefill_kwargs=policy_kwargs,
        ):
            kwargs.pop("prefill_step_size", None)
            kwargs["prefill_step_size"] = None

    # Use batch_size for prefill and completion to ensure consistent processing
    forwarded_keys = set(kwargs) - {
        "prefill_batch_size",
        "completion_batch_size",
        "compute_logprobs",
    }
    if invalid := forwarded_keys - BatchGeneratorOptions.__annotations__.keys():
        raise TypeError(f"Unsupported batch generation options: {sorted(invalid)}")
    generator_options = cast(BatchGeneratorOptions, kwargs)
    generator_options["prefill_batch_size"] = batch_size
    generator_options["completion_batch_size"] = batch_size
    generator_options["compute_logprobs"] = False
    gen = BatchGenerator(
        model.language_model,
        processor,
        **generator_options,
    )

    if logits_processors and is_logits_processor_list(logits_processors):
        logits_processors = [
            [_clone_or_share_logits_processor(p) for p in logits_processors]
            for _ in range(batch_size)
        ]

    # The nonempty shared form was expanded above; an empty list is also a
    # valid empty per-row collection and is retained unchanged.
    assert logits_processors is None or is_per_row_logits_processor_list(
        logits_processors
    )
    prompt_kwargs = _split_prompt_kwargs_per_row(gen_kwargs, batch_size)
    uids = gen.insert(
        input_ids,
        max_tokens,
        prompt_kwargs=prompt_kwargs,
        logits_processors=logits_processors,
        attention_mask=mask,
    )
    results: dict[int, list[int]] = {uid: [] for uid in uids}

    tic = time.perf_counter()
    while gen.has_work:
        _, generation_responses = gen.next()
        for r in generation_responses:
            if r.finish_reason != "stop":
                assert r.token is not None
                results[r.uid].append(r.token)
    total_time = time.perf_counter() - tic

    gen.close()

    detokenizer = make_streaming_detokenizer(processor)
    texts: list[str] = []
    for uid in uids:
        detokenizer.reset()
        for t in results[uid]:
            detokenizer.add_token(t)
        detokenizer.finalize()
        texts.append(detokenizer.text)

    stats = gen.stats()
    stats.generation_time = total_time - stats.prompt_time
    if stats.generation_time > 0:
        stats.generation_tps = stats.generation_tokens / stats.generation_time
    return texts, stats
