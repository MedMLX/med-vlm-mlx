"""Configuration dataclasses for Gemma3 / MedGemma checkpoints."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isqrt
from typing import List, Optional, Self, cast

from mlx_vlm.contracts import RopeScaling
from mlx_vlm.models.base import BaseModelConfig


def _validate_image_token_count(count: object) -> None:
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("mm_tokens_per_image must be a positive perfect square")
    if isqrt(count) ** 2 != count:
        raise ValueError("mm_tokens_per_image must be a positive perfect square")


@dataclass
class TextConfig(BaseModelConfig):
    """Gemma3 decoder settings.

    Attention scores are scaled by ``query_pre_attn_scalar ** -0.5``. Every
    ``sliding_window_pattern``-th layer is global; the others attend within
    ``sliding_window`` tokens. ``mm_tokens_per_image`` is the number of language
    tokens each image occupies and must be a positive square.
    """

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int = 8
    head_dim: int = 256
    rms_norm_eps: float = 1.0e-6
    vocab_size: int = 262208
    num_key_value_heads: int = 4
    rope_global_base_freq: float = 1_000_000.0
    rope_local_base_freq: float = 10_000.0
    rope_traditional: bool = False
    query_pre_attn_scalar: float = 256
    sliding_window: int = 1024
    rope_scaling: RopeScaling | None = None
    mm_tokens_per_image: int = 256
    sliding_window_pattern: int = 6
    max_position_embeddings: int = 4096

    def __post_init__(self) -> None:
        """Reject an image token count that is not a positive perfect square."""
        _validate_image_token_count(self.mm_tokens_per_image)


@dataclass
class VisionConfig(BaseModelConfig):
    """SigLIP vision tower settings; images are square ``image_size`` pixels."""

    model_type: str
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    patch_size: int
    image_size: int = 224
    num_channels: int = 3
    layer_norm_eps: float = 1e-6


@dataclass
class ModelConfig(BaseModelConfig):
    """Top-level Gemma3 config; ``image_token_index`` is the image placeholder ID."""

    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    vocab_size: int = 257152
    ignore_index: int = -100
    image_token_index: int = 262144
    eos_token_id: Optional[List[int]] = None

    @classmethod
    def from_dict(cls, params: Mapping[str, object]) -> Self:
        """Validate MedGemma configuration, accepting the upstream top-level token count.

        The local text config also accepts that setting. Explicit top-level and
        nested counts must agree; the input mapping is not modified.
        """
        from pydantic import TypeAdapter

        normalized = dict(params)
        raw_text = params.get("text_config")
        if isinstance(raw_text, TextConfig):
            _validate_image_token_count(raw_text.mm_tokens_per_image)
        elif isinstance(raw_text, Mapping):
            raw_text_mapping = cast(Mapping[str, object], raw_text)
            if "mm_tokens_per_image" in raw_text_mapping:
                _validate_image_token_count(raw_text_mapping["mm_tokens_per_image"])
        if "mm_tokens_per_image" in params:
            count = params["mm_tokens_per_image"]
            _validate_image_token_count(count)
            if isinstance(raw_text, TextConfig):
                if raw_text.mm_tokens_per_image != count:
                    raise ValueError("conflicting mm_tokens_per_image values")
            elif isinstance(raw_text, Mapping):
                text_config = dict(cast(Mapping[str, object], raw_text))
                if (
                    "mm_tokens_per_image" in text_config
                    and text_config["mm_tokens_per_image"] != count
                ):
                    raise ValueError("conflicting mm_tokens_per_image values")
                text_config["mm_tokens_per_image"] = count
                normalized["text_config"] = text_config
            else:
                raise ValueError("text_config is required for mm_tokens_per_image")
        return TypeAdapter(cls).validate_python(normalized)
