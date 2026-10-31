"""Autoregressive text generation for MedGemma."""

from mlx_vlm.generate.ar import (
    BatchGenerator,
    BatchResponse,
    BatchStats,
    PromptProcessingBatch,
    batch_generate,
    generate_step,
)
from mlx_vlm.generate.cli import (
    DEFAULT_MODEL_PATH,
    DEFAULT_SEED,
    DEFAULT_THINKING_END_TOKEN,
    DEFAULT_THINKING_START_TOKEN,
    main,
    parse_arguments,
)
from mlx_vlm.generate.common import (
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_QUANTIZED_KV_START,
    DEFAULT_REPETITION_CONTEXT_SIZE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    GenerationResult,
    PromptCacheState,
)
from mlx_vlm.generate.dispatch import generate, stream_generate

__all__ = [
    "BatchGenerator",
    "BatchResponse",
    "BatchStats",
    "GenerationResult",
    "PromptCacheState",
    "PromptProcessingBatch",
    "batch_generate",
    "generate",
    "generate_step",
    "main",
    "parse_arguments",
    "stream_generate",
    "DEFAULT_KV_GROUP_SIZE",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_PREFILL_STEP_SIZE",
    "DEFAULT_QUANTIZED_KV_START",
    "DEFAULT_REPETITION_CONTEXT_SIZE",
    "DEFAULT_SEED",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_THINKING_END_TOKEN",
    "DEFAULT_THINKING_START_TOKEN",
    "DEFAULT_TOP_P",
]
