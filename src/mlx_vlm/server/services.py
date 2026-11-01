"""Typed application services shared by the compatible text HTTP routes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from mlx_vlm.contracts import ImageSource
from mlx_vlm.generate.types import GenerateTextKwargs, ProcessorLike, TokenizerLike
from mlx_vlm.models.protocols import GenerationConfigLike, GenerationModelLike
from mlx_vlm.server.schemas import ChatLogprobContent

if TYPE_CHECKING:
    from mlx_vlm.server.generation import GenerationArguments


type LoadedModel = tuple[
    GenerationModelLike, ProcessorLike | TokenizerLike, GenerationConfigLike
]


class CachedModelLoader(Protocol):
    """Load or reuse the model tuple for a checkpoint path."""

    def __call__(
        self,
        model_path: str,
        *,
        model_kind: str = "auto",
    ) -> LoadedModel:
        """Return the model, processor and config for ``model_path``."""
        ...


class ContextPreflight(Protocol):
    """Validate that a prompt fits the loaded model's context budget."""

    def __call__(
        self,
        *,
        endpoint: str,
        model: str,
        prompt: str,
        images: list[ImageSource] | None = None,
        args: GenerationArguments,
    ) -> Awaitable[None]:
        """Return an awaitable that raises when the request exceeds the budget."""
        ...


@runtime_checkable
class TokenDecoder(Protocol):
    """Decode token IDs with the model's tokenizer."""

    def decode(self, token_ids: list[int]) -> str:
        """Decode one token sequence into text."""
        ...


class LogprobFormatter(Protocol):
    """Render one token's log probability for a chat response."""

    def __call__(
        self,
        tokenizer: TokenDecoder,
        token_id: int,
        logprob: float,
        top_logprobs: list[tuple[int, float]] | None = None,
        top_k: int = 0,
    ) -> ChatLogprobContent:
        """Return the token text, its log probability and the top alternatives."""
        ...


@dataclass(frozen=True)
class ProtocolServices:
    """Application-owned callbacks; canonical generation functions are imported directly."""

    get_cached_model: CachedModelLoader
    preflight_stream_context_budget: ContextPreflight
    count_thinking_tag_tokens: Callable[[str, str | None, str | None], int]
    make_logprob_content: LogprobFormatter


def generation_kwargs(
    args: GenerationArguments, overrides: GenerateTextKwargs | None = None
) -> GenerateTextKwargs:
    """Merge request options with the current runtime caches before keyword expansion."""
    from mlx_vlm.server.runtime import runtime

    kwargs = args.to_generate_kwargs()
    kwargs["vision_cache"] = runtime.model_cache.for_kind().get("vision_cache")
    if overrides:
        kwargs.update(overrides)
    return kwargs
