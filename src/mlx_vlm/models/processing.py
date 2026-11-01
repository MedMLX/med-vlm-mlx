"""Validated interfaces to Transformers' dynamically typed processor results."""

from pathlib import Path
from typing import Protocol, cast

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from mlx_vlm.contracts import (
    InputTensor,
    JSONValue,
    ProcessedInputs,
    ProcessorValue,
    require_processed_inputs,
)


class _Tokenize(Protocol):
    def __call__(
        self, text: str | list[str] | None = None, **kwargs: ProcessorValue
    ) -> object: ...


class _LoadTokenizer(Protocol):
    def __call__(
        self, path: str | Path, **kwargs: ProcessorValue
    ) -> PreTrainedTokenizerBase: ...


def tokenize_inputs(
    tokenizer: PreTrainedTokenizerBase,
    text: str | list[str] | None,
    kwargs: dict[str, ProcessorValue],
) -> ProcessedInputs:
    """Validate the array/list payload returned by the installed tokenizer."""
    # Transformers 5 leaves BatchEncoding.data dynamic; validate its values instead
    # of carrying those dependency types into native model inputs.
    encoded = cast(_Tokenize, tokenizer)(text, **kwargs)
    return require_processed_inputs(getattr(encoded, "data", encoded))


def load_tokenizer(
    path: str | Path, kwargs: dict[str, ProcessorValue]
) -> PreTrainedTokenizerBase:
    """Load the verified AutoTokenizer interface with its concrete base contract."""
    # The installed factory accepts a checkpoint path plus tokenizer load kwargs.
    return cast(_LoadTokenizer, getattr(AutoTokenizer, "from_pretrained"))(
        path, **kwargs
    )


type DecodeIds = int | list[int] | list[list[int]] | InputTensor


class DecodeTokens(Protocol):
    """Single-sequence decode signature."""

    def __call__(
        self,
        token_ids: DecodeIds,
        skip_special_tokens: bool = False,
        **kwargs: JSONValue,
    ) -> str | list[str]:
        """Decode token IDs into one string, or one string per batch row."""
        ...


class BatchDecodeTokens(Protocol):
    """Batched decode signature."""

    def __call__(
        self,
        sequences: list[int] | list[list[int]] | InputTensor,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs: JSONValue,
    ) -> list[str]:
        """Decode a batch of token sequences into one string per row."""
        ...
