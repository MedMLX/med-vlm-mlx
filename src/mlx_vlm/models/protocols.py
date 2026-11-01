"""Structural interfaces at MLX's dynamically registered model boundary."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.contracts import (
    CacheComponent,
    CacheFactory,
    CacheModel,
    InputTensor,
    LayeredModel,
    ModelArgument,
)

if TYPE_CHECKING:
    from mlx_vlm.models.base import InputEmbeddingsFeatures, LanguageModelOutput


class GenerationConfigLike(Protocol):
    """Checkpoint configuration fields read by generation."""

    @property
    def model_type(self) -> str:
        """Return the architecture name, such as ``"gemma3"``."""
        ...

    @property
    def eos_token_id(self) -> int | list[int] | None:
        """Return the configured end-of-sequence ids, if any."""
        ...


@runtime_checkable
class LanguageModelLike(Protocol):
    """Forward and cache layout shared by the shipped autoregressive decoders."""

    @property
    def layers(self) -> Sequence[nn.Module]:
        """Return the decoder layers that own one cache entry each."""
        ...

    def __call__(
        self,
        inputs: mx.array | None = None,
        /,
        *,
        cache: Sequence[CacheComponent] | None = None,
        **kwargs: ModelArgument,
    ) -> LanguageModelOutput:
        """Run a decoder step and return logits for the retained token positions."""
        ...


@runtime_checkable
class GenerationModelLike(Protocol):
    """Multimodal model interface used by generation and batch scheduling."""

    def __call__(
        self,
        inputs: mx.array | None = None,
        /,
        *,
        cache: Sequence[CacheComponent] | None = None,
        **kwargs: ModelArgument,
    ) -> LanguageModelOutput:
        """Run the language model step with optional prompt embeddings."""
        ...

    @property
    def config(self) -> GenerationConfigLike:
        """Return the checkpoint configuration."""
        ...

    @property
    def language_model(self) -> LanguageModelLike:
        """Return the autoregressive decoder."""
        ...

    def get_input_embeddings(
        self,
        input_ids: mx.array | None,
        pixel_values: (
            InputTensor | list[InputTensor] | list[list[InputTensor]] | None
        ) = None,
        *,
        mask: mx.array | None = None,
        **kwargs: ModelArgument,
    ) -> InputEmbeddingsFeatures:
        """Merge image features into token embeddings and return the 4D mask."""
        ...


def require_generation_model(value: object) -> GenerationModelLike:
    """Validate dynamically registered MLX modules before narrowing their API.

    MLX stores child modules and config in its dictionary; runtime protocol
    checks use static lookup and therefore miss those registered attributes.
    """
    if not all(
        callable(getattr(value, name, None))
        for name in ("get_input_embeddings", "__call__")
    ):
        raise TypeError(
            "Text generation requires a callable model and embedding interface"
        )
    config: object = getattr(value, "config", None)
    if not isinstance(getattr(config, "model_type", None), str):
        raise TypeError("Text generation requires a config with a model_type")
    require_language_model(getattr(value, "language_model", None))
    return cast(GenerationModelLike, value)


def require_language_model(value: object) -> LanguageModelLike:
    """Validate the nested decoder using MLX's dynamic attribute lookup."""
    if not callable(value):
        raise TypeError("Text generation requires a callable language model")
    layers: object = getattr(value, "layers", None)
    if not isinstance(layers, Sequence):
        raise TypeError("Text generation requires language-model cache layers")
    return cast(LanguageModelLike, value)


def generation_cache_layers(
    model: GenerationModelLike | LanguageModelLike,
) -> Sequence[object]:
    """Read the cache layout without requiring a decoder forward call."""
    decoder: object = getattr(model, "language_model", model)
    layers: object = getattr(decoder, "layers", None)
    if not isinstance(layers, Sequence):
        raise TypeError("Cache construction requires language-model layers")
    return cast(Sequence[object], layers)


def generation_cache_model(
    model: GenerationModelLike | LanguageModelLike,
) -> CacheModel:
    """Preserve outer cache factories; otherwise resolve the decoder's layers."""
    if isinstance(model, CacheFactory):
        return model
    generation_cache_layers(model)
    # Only the layers sequence is used by the default cache constructor.
    return cast(LayeredModel, getattr(model, "language_model", model))
