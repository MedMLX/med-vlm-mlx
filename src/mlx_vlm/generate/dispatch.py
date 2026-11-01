"""Prepare multimodal inputs and run streaming or complete text generation."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence
from typing import Generator, List, Optional, cast

import mlx.core as mx
from transformers import PreTrainedTokenizer

from mlx_vlm.contracts import (
    CacheEntry,
    ImageSource,
    InputTensor,
    PromptInputs,
    array_int_list,
    evaluate,
    is_cache_entry_list,
    is_int_list,
    is_pixel_values,
)
from mlx_vlm.generate.common import (
    GenerationResult,
    generation_stream,
    wired_limit,
)
from mlx_vlm.models import cache
from mlx_vlm.models.gemma3.attention import image_attention_mask
from mlx_vlm.models.protocols import GenerationModelLike
from mlx_vlm.tokenizer_utils import make_streaming_detokenizer
from mlx_vlm.utils import (
    StoppingCriteria,
    ThinkingBudgetCriteria,
    prepare_inputs,
    should_add_special_tokens,
)
from mlx_vlm.vision_cache import VisionFeatures

logger = logging.getLogger("mlx_vlm.generate")


DEFAULT_THINKING_START_TOKEN = "<think>"
DEFAULT_THINKING_END_TOKEN = "</think>"


from mlx_vlm.generate.ar import normalize_resize_shape as normalize_resize_shape
from mlx_vlm.generate.types import (
    GenerateStepKwargs,
    GenerateTextKwargs,
    GenerationOptionsBase,
    ProcessorForwardOptions,
    ProcessorLike,
    StreamGenerateKwargs,
    TokenEncoder,
    TokenizerLike,
    Unpack,
    resolve_tokenizer,
)


def prepare_generation_inputs(
    model: GenerationModelLike,
    processor: ProcessorLike | TokenizerLike | PreTrainedTokenizer,
    prompt: str,
    image: ImageSource | list[ImageSource] | list[str] | None,
    kwargs: StreamGenerateKwargs,
) -> tuple[
    mx.array,
    InputTensor | list[InputTensor] | list[list[InputTensor]] | None,
    mx.array | None,
    PromptInputs,
]:
    """Prepare multimodal inputs once, or consume caller-prepared tensors."""
    resize_shape = normalize_resize_shape(kwargs.pop("resize_shape", None))
    if kwargs.get("input_ids") is not None:
        input_ids = kwargs.pop("input_ids")
        pixel_values = kwargs.pop("pixel_values", None)
        mask = kwargs.pop("mask", None)
        return input_ids, pixel_values, mask, cast(PromptInputs, dict(kwargs))

    forwarded = dict(kwargs)
    # These are the explicit processor controls; other values remain model inputs.
    for name in ("padding", "pad_to_uniform_size"):
        if name in forwarded and not isinstance(forwarded[name], bool):
            raise TypeError(f"{name} must be a boolean")
    for name in ("padding_side", "return_tensors"):
        if name in forwarded and not isinstance(forwarded[name], str):
            raise TypeError(f"{name} must be text")
    processor_options = cast(ProcessorForwardOptions, forwarded)
    inputs = prepare_inputs(
        processor,
        images=image,
        prompts=prompt,
        image_token_index=getattr(model.config, "image_token_index", None),
        resize_shape=resize_shape,
        add_special_tokens=should_add_special_tokens(
            model.config.model_type, processor
        ),
        **processor_options,
    )
    data_kwargs: PromptInputs = {
        key: value
        for key, value in inputs.items()
        if key not in ("input_ids", "pixel_values", "attention_mask")
    }
    # Processor output consists of the tensor/JSON values allowed by model forwards.
    cast(PromptInputs, kwargs).update(data_kwargs)
    input_ids = inputs.get("input_ids")
    pixel_values = inputs.get("pixel_values")
    mask = inputs.get("attention_mask")
    if not isinstance(input_ids, mx.array):
        raise TypeError("Prepared input_ids must be an MLX array")
    if not is_pixel_values(pixel_values):
        raise TypeError("Prepared pixel_values must be tensors or lists of tensors")
    if mask is not None and not isinstance(mask, mx.array):
        raise TypeError("Prepared attention_mask must be an MLX array")
    return input_ids, pixel_values, mask, data_kwargs


def require_vision_features(value: object) -> VisionFeatures:
    """Validate native encoder output without copying its lazy array buffers."""
    if isinstance(value, mx.array):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, mx.array)
        for item in cast(list[object] | tuple[object, ...], value)
    ):
        return cast(Sequence[mx.array], value)
    raise TypeError("Vision encoder must return an MLX array or a list/tuple of arrays")


def _drop_cached_media_inputs(kwargs: GenerationOptionsBase) -> None:
    """After full-prompt position priming, embed the cached suffix as text only."""
    kwargs.pop("cached_image_features", None)


def _cache_fully_retained(c: CacheEntry) -> bool:
    """Whether ``c`` still holds its whole sequence from position 0.

    Only such a cache can be rolled back to an earlier prefix.
    """
    max_size = getattr(c, "max_size", None)
    if max_size is not None:  # RotatingKVCache: reusable until the window wraps
        return int(getattr(c, "offset", 0) or 0) <= int(max_size)
    return True  # KVCache / QuantizedKVCache retain the whole sequence


def _prefix_cache_trim_amount(
    kv_cache: List[CacheEntry], prefix_len: int
) -> Optional[int]:
    """Trailing tokens to drop so ``kv_cache`` keeps only its first ``prefix_len``.

    The old reuse path sliced key/value arrays directly --
    ``keys[..., :prefix_len, :]`` -- which is only valid for a flat cache. On a
    rotating (sliding-window) cache it slices a ring buffer by a logical length,
    taking an arbitrary rotation of the window and leaving the ring index stale:
    silent output corruption. Returns the number of tokens to drop
    (``0`` when the whole cache is reusable), or ``None`` when an entry has already
    evicted part of the prefix or cannot be trimmed, and the caller must
    cold-prefill instead.
    """
    cached_len = max((int(getattr(c, "offset", 0) or 0) for c in kv_cache), default=0)
    n_drop = max(0, cached_len - prefix_len)
    if n_drop:
        for c in kv_cache:
            is_trimmable = getattr(c, "is_trimmable")
            assert callable(is_trimmable)
            if not (is_trimmable() and _cache_fully_retained(c)):
                return None
    return n_drop


from mlx_vlm.generate.ar import generate_step


def stream_generate(
    model: GenerationModelLike,
    processor: ProcessorLike | TokenizerLike | PreTrainedTokenizer,
    prompt: str,
    image: ImageSource | list[ImageSource] | list[str] | None = None,
    **kwargs: Unpack[StreamGenerateKwargs],
) -> Generator[GenerationResult]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        model (nn.Module): The model to use for generation.
        processor (PreTrainedTokenizer): The tokenizer/processor.
        prompt (str): The input prompt text.
        image: Optional image input or inputs accepted by the processor.
        prefill_step_size (int, optional): Number of tokens to process per prefill
          step when the model permits chunked prefill. The model may disable
          chunking for inputs such as Gemma image blocks.
        kwargs: Generation, media, and cache options declared by
          ``StreamGenerateKwargs``; remaining step options go to
          :func:`generate_step`.

    Yields:
        GenerationResult: Incremental text segments and generation statistics.
          Concatenate ``text`` from all yielded results for the complete output.
          The final result carries ``finish_reason`` and the generated ``token_ids``;
          the latter includes the stop token when generation stops on one.

    Iterate to exhaustion, including after the final result, so optional prompt
    cache updates and generation cleanup following the final yield can run.
    """
    tokenizer = resolve_tokenizer(processor)
    verbose = kwargs.pop("verbose", False)
    # Set up thinking budget criteria if requested
    thinking_budget = kwargs.pop("thinking_budget", None)
    thinking_end_token = kwargs.pop("thinking_end_token", DEFAULT_THINKING_END_TOKEN)
    thinking_start_token = kwargs.pop(
        "thinking_start_token", DEFAULT_THINKING_START_TOKEN
    )
    enable_thinking = kwargs.pop("enable_thinking", False)

    # Skip special tokens
    skip_special_tokens = kwargs.pop("skip_special_tokens", False)
    skip_special_token_ids: set[int] = set()
    if skip_special_tokens:
        special_ids: object = getattr(tokenizer, "all_special_ids", [])
        if not is_int_list(special_ids):
            raise TypeError("Tokenizer all_special_ids must be an integer list")
        skip_special_token_ids = set(special_ids)

    vision_cache = kwargs.pop("vision_cache", None)
    prompt_cache_state = kwargs.pop("prompt_cache_state", None)
    # Arrays are valid media inputs and cannot be tested for scalar truth.
    if isinstance(image, list) and not image:
        image = None

    input_ids, pixel_values, mask, _ = prepare_generation_inputs(
        model, processor, prompt, image, kwargs
    )

    # Vision feature caching: reuse cached image features across turns
    if vision_cache is not None and image is not None and pixel_values is not None:
        cached = vision_cache.get(image)
        if cached is not None:
            kwargs["cached_image_features"] = cached
        elif hasattr(model, "encode_images"):
            encode_images = getattr(model, "encode_images")
            if not callable(encode_images):
                raise TypeError("model.encode_images must be callable")
            features = require_vision_features(encode_images(pixel_values, **kwargs))
            evaluate(*features)
            vision_cache.put(image, features)
            kwargs["cached_image_features"] = features
        elif hasattr(model, "encode_image"):
            encode_image = getattr(model, "encode_image")
            if not callable(encode_image):
                raise TypeError("model.encode_image must be callable")
            features = require_vision_features(encode_image(pixel_values))
            evaluate(features)
            vision_cache.put(image, features)
            kwargs["cached_image_features"] = features

    # Prompt cache reuse: skip common prefix from previous turn
    reused_prefix_len = 0
    full_input_ids = input_ids
    full_input_ids_list = array_int_list(input_ids.flatten())
    image_token_id = getattr(model.config, "image_token_index", None)

    def _suffix_is_text_only(prefix_len: int) -> bool:
        # Cached image features cover the whole prompt, so a reused prefix must
        # contain every image placeholder and leave a text-only suffix.
        return image_token_id not in full_input_ids_list[prefix_len:]

    if prompt_cache_state is not None and prompt_cache_state.cache is not None:
        prefix_len = prompt_cache_state.find_prefix_length(full_input_ids_list)
        kv_cache = prompt_cache_state.cache
        # None => a cache can't be trimmed back to the shared prefix (wrapped
        # rotating window); reusing it would corrupt state, so cold-prefill instead.
        n_drop = _prefix_cache_trim_amount(kv_cache, prefix_len)
        if (
            0 < prefix_len < input_ids.shape[1]
            and n_drop is not None
            and _suffix_is_text_only(prefix_len)
        ):
            # Drop cached tokens past the shared prefix via each cache's own trim().
            for c in kv_cache:
                if n_drop:
                    trim = getattr(c, "trim")
                    assert callable(trim)
                    trim(n_drop)
            reused_prefix_len = prefix_len
            # Trim to only new tokens
            input_ids = input_ids[:, prefix_len:]
            pixel_values = None
            _drop_cached_media_inputs(kwargs)
            kwargs["prompt_cache"] = kv_cache

    if reused_prefix_len:
        if not isinstance(image_token_id, int):
            raise ValueError("MedGemma configuration requires image_token_index")
        # The decoder indexes the complete prompt mask by its cache offset.
        kwargs["attention_mask_4d"] = image_attention_mask(
            full_input_ids, mask, image_token_id
        )
        if mask is not None:
            mask = mask[:, reused_prefix_len:]

    if thinking_budget is not None:
        if thinking_start_token is None:
            raise ValueError("thinking_start_token is required with a thinking budget")
        if not isinstance(tokenizer, TokenEncoder):
            raise TypeError("Thinking budgets require a tokenizer encode method")
        start_ids = tokenizer.encode(thinking_start_token, add_special_tokens=False)
        if not start_ids:
            raise ValueError(
                "The thinking start marker must encode to at least one token"
            )
        thinking_start_token_id = start_ids[-1]
        prompt_preopens_thinking = thinking_start_token_id in array_int_list(
            input_ids.flatten()
        )
        tokenizer.thinking_budget_criteria = ThinkingBudgetCriteria(
            tokenizer=tokenizer,
            thinking_budget=thinking_budget,
            thinking_end_token=thinking_end_token,
            thinking_start_token=thinking_start_token,
            enable_thinking=enable_thinking,
            prompt_preopens_thinking=prompt_preopens_thinking,
        )
        kwargs["thinking_budget_criteria"] = tokenizer.thinking_budget_criteria
    else:
        tokenizer.thinking_budget_criteria = None

    # Ensure we have a prompt_cache we can track for reuse.
    if "prompt_cache" not in kwargs:
        language_model = model.language_model
        created_cache = cache.make_prompt_cache(
            language_model,
            max_kv_size=kwargs.get("max_kv_size", None),
        )
        if not is_cache_entry_list(created_cache):
            raise TypeError(
                "Generation requires cache entries exposing state and metadata"
            )
        kwargs["prompt_cache"] = created_cache
    tracked_cache = kwargs["prompt_cache"]

    total_prompt_tokens = reused_prefix_len + input_ids.size

    with wired_limit(model, [generation_stream]):
        detokenizer = make_streaming_detokenizer(processor)
        thinking_criteria = getattr(tokenizer, "thinking_budget_criteria", None)
        for key in kwargs:
            if key in {"input_ids", "pixel_values", "mask", "verbose"}:
                raise TypeError(
                    "mlx_vlm.generate.ar.generate_step() "
                    f"got multiple values for keyword argument '{key}'"
                )
        step_kwargs = cast(GenerateStepKwargs, kwargs)
        gen = generate_step(
            input_ids,
            model,
            pixel_values,
            mask,
            verbose=verbose,
            **step_kwargs,
        )
        tic = time.perf_counter()

        generated_tokens: list[int] = []
        finish_reason: Optional[str] = None
        prompt_tps = 0.0
        token: int | mx.array | None = None
        logprobs: mx.array | None = None
        n = -1
        for n, (token, logprobs) in enumerate(gen):
            # stream_generate decodes one sequence; batched token arrays are
            # handled by BatchGenerator rather than a scalar detokenizer.
            assert isinstance(token, int)
            if n == 0:
                prompt_time = time.perf_counter() - tic
                prompt_tps = total_prompt_tokens / prompt_time
                tic = time.perf_counter()

            generated_tokens.append(token)

            # Check thinking budget and force token if needed
            if thinking_criteria is not None:
                thinking_criteria(token)

            # Stop generation if the token is in the eos_token_ids
            if tokenizer.stopping_criteria(token):
                finish_reason = "stop"
                break

            detokenizer.add_token(token, skip_special_token_ids=skip_special_token_ids)

            # Yield the last segment if streaming
            yield GenerationResult(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                prompt_tokens=total_prompt_tokens,
                generation_tokens=n + 1,
                total_tokens=total_prompt_tokens + n + 1,
                prompt_tps=prompt_tps,
                generation_tps=(n + 1) / (time.perf_counter() - tic),
                peak_memory=mx.get_peak_memory() / 1e9,
                cached_tokens=reused_prefix_len,
            )
        else:
            # generate_step exhausted its budget without stopping_criteria firing.
            finish_reason = "length"

        if not generated_tokens:
            prompt_time = time.perf_counter() - tic
            prompt_tps = total_prompt_tokens / prompt_time if prompt_time > 0 else 0.0
            yield GenerationResult(
                text="",
                token=None,
                logprobs=None,
                prompt_tokens=total_prompt_tokens,
                generation_tokens=0,
                total_tokens=total_prompt_tokens,
                prompt_tps=prompt_tps,
                generation_tps=0.0,
                peak_memory=mx.get_peak_memory() / 1e9,
                cached_tokens=reused_prefix_len,
                finish_reason="length",
                token_ids=[],
            )
            return

        detokenizer.finalize()
        yield GenerationResult(
            text=detokenizer.last_segment,
            token=token,
            logprobs=logprobs,
            prompt_tokens=total_prompt_tokens,
            generation_tokens=n + 1,
            total_tokens=total_prompt_tokens + n + 1,
            prompt_tps=prompt_tps,
            generation_tps=(n + 1) / (time.perf_counter() - tic),
            peak_memory=mx.get_peak_memory() / 1e9,
            cached_tokens=reused_prefix_len,
            finish_reason=finish_reason,
            token_ids=[int(t) for t in generated_tokens],
        )

        # Save cache state for potential reuse on next turn
        all_ids: Optional[List[int]] = None
        if prompt_cache_state is not None:
            all_ids = full_input_ids_list + [t for t in generated_tokens]
            assert tracked_cache is not None
            prompt_cache_state.update(all_ids, tracked_cache)

        # Cleanup after generation
        mx.clear_cache()


def generate(
    model: GenerationModelLike,
    processor: ProcessorLike | TokenizerLike | PreTrainedTokenizer,
    prompt: str,
    image: ImageSource | list[ImageSource] | list[str] | None = None,
    verbose: bool = False,
    **kwargs: Unpack[GenerateTextKwargs],
) -> GenerationResult:
    """Generate text and return the accumulated response and generation metadata.

    Args:
       model: The model used by the selected generation path.
       processor: The media processor or tokenizer for the model.
       prompt: The text prompt.
       image: Optional image input or inputs.
       verbose: Print the prompt, generated text and timing information.
       **kwargs: Options declared by ``GenerateTextKwargs``. The autoregressive
           path defaults to 2048 output tokens when ``max_tokens`` is omitted;
           generation uses the autoregressive medical path.
    """

    if verbose:
        print("=" * 10)
        files: list[object] = []
        if image is not None:
            if not isinstance(image, Iterable):
                raise TypeError(f"'{type(image).__name__}' object is not iterable")
            files.extend(cast(Iterable[object], image))

        print(f"Files: {files}", "\n")

        print("Prompt:", prompt)

    text = ""
    last_response = None

    eos_tokens = kwargs.get("eos_tokens", None)
    stopping_criteria = kwargs.get("stopping_criteria", None)

    # Get the tokenizer
    tokenizer = resolve_tokenizer(processor)

    # Add custom EOS tokens to the stopping criteria
    if eos_tokens is not None:
        tokenizer.stopping_criteria.add_eos_token_ids(eos_tokens)

    # Use custom stopping criteria
    elif stopping_criteria is not None:
        if isinstance(stopping_criteria, StoppingCriteria) or callable(
            stopping_criteria
        ):
            tokenizer.stopping_criteria = stopping_criteria
        else:
            raise ValueError(
                "stopping_criteria must be an instance of StoppingCriteria or a callable"
            )
    else:
        config = model.config
        tokenizer.stopping_criteria.reset(config.eos_token_id)

    for response in stream_generate(
        model, processor, prompt, image, verbose=verbose, **kwargs
    ):
        if verbose and not response.text_already_printed:
            print(response.text, end="", flush=True)
        text += response.text
        last_response = response

    clean_output = getattr(processor, "clean_output", None)
    if callable(clean_output):
        cleaned = clean_output(text)
        if not isinstance(cleaned, str):
            raise TypeError("processor.clean_output must return text")
        text = cleaned

    if last_response is None:
        return GenerationResult(text=text, peak_memory=mx.get_peak_memory() / 1e9)

    if verbose:
        print("\n" + "=" * 10)
        if len(text) == 0:
            print("No text generated for this prompt")
        print(
            f"Prompt: {last_response.prompt_tokens} tokens, "
            f"{last_response.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"Generation: {last_response.generation_tokens} tokens, "
            f"{last_response.generation_tps:.3f} tokens-per-sec"
        )
        print(f"Peak memory: {last_response.peak_memory:.3f} GB")

    return GenerationResult(
        text=text,
        token=last_response.token,
        token_ids=last_response.token_ids,
        logprobs=last_response.logprobs,
        prompt_tokens=last_response.prompt_tokens,
        generation_tokens=last_response.generation_tokens,
        total_tokens=last_response.total_tokens,
        prompt_tps=last_response.prompt_tps,
        generation_tps=last_response.generation_tps,
        peak_memory=last_response.peak_memory,
        cached_tokens=last_response.cached_tokens,
        finish_reason=last_response.finish_reason,
    )
