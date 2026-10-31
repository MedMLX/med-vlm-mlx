"""Typed option dictionaries and tokenizer protocols for generation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, Protocol, TypeIs
from typing import Unpack as Unpack
from typing import runtime_checkable

import mlx.core as mx
from typing_extensions import ReadOnly, TypedDict

from mlx_vlm.contracts import (
    CacheEntry,
    InputTensor,
    ModelArgument,
)
from mlx_vlm.tokenizer_utils import StreamingDetokenizer

if TYPE_CHECKING:

    from mlx_vlm.generate.common import PromptCacheState
    from mlx_vlm.utils import ThinkingBudgetCriteria
    from mlx_vlm.vision_cache import VisionFeatureCache

__all__ = ["ProcessorLike", "Unpack"]


@runtime_checkable
class StopCriteria(Protocol):
    @property
    def eos_token_ids(self) -> list[int]: ...

    def reset(self, eos_token_ids: int | list[int] | None = None) -> None: ...

    def add_eos_token_ids(
        self,
        new_eos_token_ids: int | str | list[int] | list[str] | set[int] | None = None,
    ) -> None: ...

    def __call__(self, token: int | mx.array, /) -> bool: ...


@runtime_checkable
class TokenizerLike(Protocol):
    """Installed generation hooks, also usable with precomputed token IDs."""

    stopping_criteria: StopCriteria
    thinking_budget_criteria: ThinkingBudgetCriteria | None


@runtime_checkable
class TokenEncoder(Protocol):
    """Encoding interface for resolving thinking marker token IDs."""

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]: ...


@runtime_checkable
class ProcessorLike(Protocol):
    """Processor interface used by the generation entry points."""

    @property
    def tokenizer(self) -> TokenizerLike:
        """Return the text tokenizer carrying installed generation hooks."""
        ...

    @property
    def detokenizer(self) -> StreamingDetokenizer:
        """Return the streaming detokenizer for incremental output."""
        ...


def resolve_tokenizer(processor: object) -> TokenizerLike:
    """Resolve a text processor after its generation hooks have been installed."""
    tokenizer = getattr(processor, "tokenizer", processor)
    # The thinking hook is optional until the first generation request.
    if not hasattr(tokenizer, "thinking_budget_criteria"):
        setattr(tokenizer, "thinking_budget_criteria", None)
    if not isinstance(tokenizer, TokenizerLike):
        raise TypeError(
            "Text generation requires a tokenizer with installed generation hooks"
        )
    if not callable(tokenizer.stopping_criteria):
        raise TypeError("Tokenizer stopping_criteria must be callable")
    return tokenizer


class ProcessorForwardOptions(
    TypedDict, total=False, extra_items=ReadOnly[ModelArgument]
):
    padding: bool
    padding_side: str
    pad_to_uniform_size: bool
    return_tensors: str


class GenerationForwardOptions(TypedDict, total=False):
    repetition_penalty: float | None
    repetition_context_size: int | None
    presence_penalty: float | None
    presence_context_size: int | None
    frequency_penalty: float | None
    frequency_context_size: int | None
    min_p: float
    logit_bias: dict[int, float] | None
    prompt_cache: list[CacheEntry] | None
    max_kv_size: int | None
    kv_bits: float | None
    kv_group_size: int
    quantized_kv_start: int
    sampler: Callable[[mx.array], mx.array] | None
    prefill_step_size: int | None
    resize_shape: tuple[int, int] | None
    eos_tokens: list[int] | list[str] | None
    stopping_criteria: StopCriteria
    thinking_budget: int | None
    thinking_end_token: str
    thinking_start_token: str | None
    enable_thinking: bool
    vision_cache: VisionFeatureCache | None
    prompt_cache_state: PromptCacheState | None
    seed: int | None
    top_n_sigma: float
    p_less: bool
    typical_p: float
    cached_image_features: mx.array | Sequence[mx.array] | None
    attention_mask_4d: mx.array | None
    inputs_embeds: mx.array | None
    thinking_budget_criteria: ThinkingBudgetCriteria | None


class GenerationDecodingOptions(GenerationForwardOptions, total=False):
    temperature: float
    top_p: float
    top_k: int
    skip_special_tokens: bool


class GenerationOptionsBase(GenerationDecodingOptions, total=False):
    input_ids: mx.array
    pixel_values: InputTensor | list[InputTensor] | list[list[InputTensor]] | None
    mask: mx.array | None


class GenerateStepKwargs(
    GenerationDecodingOptions, total=False, extra_items=ReadOnly[ModelArgument]
):
    max_tokens: int
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] | None


class GenerationOptions(GenerationOptionsBase, total=False):
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] | None


class BatchGenerationOptions(GenerationOptionsBase, total=False):
    logits_processors: (
        list[Callable[[mx.array, mx.array], mx.array]]
        | list[list[Callable[[mx.array, mx.array], mx.array]] | None]
        | None
    )


def is_logits_processor_list(
    values: (
        list[Callable[[mx.array, mx.array], mx.array]]
        | list[list[Callable[[mx.array, mx.array], mx.array]] | None]
    ),
) -> TypeIs[list[Callable[[mx.array, mx.array], mx.array]]]:
    """Distinguish shared callbacks from a list of per-row callback lists."""
    return all(callable(value) for value in values)


def is_per_row_logits_processor_list(
    values: (
        list[Callable[[mx.array, mx.array], mx.array]]
        | list[list[Callable[[mx.array, mx.array], mx.array]] | None]
    ),
) -> TypeIs[list[list[Callable[[mx.array, mx.array], mx.array]] | None]]:
    return all(value is None or isinstance(value, list) for value in values)


class GenerateTextKwargs(
    GenerationOptions, total=False, extra_items=ReadOnly[ModelArgument]
):
    max_tokens: int


class StreamGenerateKwargs(
    GenerateTextKwargs, total=False, extra_items=ReadOnly[ModelArgument]
):
    verbose: bool
