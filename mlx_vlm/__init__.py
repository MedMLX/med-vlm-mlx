"""MedGemma multimodal inference on MLX."""

import atexit
import os

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from mlx_vlm._stream_cleanup import clear_mlx_streams
from mlx_vlm.convert import convert
from mlx_vlm.generate import (
    BatchResponse,
    BatchStats,
    GenerationResult,
    PromptCacheState,
    batch_generate,
    generate,
    stream_generate,
)
from mlx_vlm.prompt_utils import apply_chat_template, get_message_json
from mlx_vlm.utils import load, prepare_inputs, process_image
from mlx_vlm.version import __version__
from mlx_vlm.vision_cache import VisionFeatureCache

# MLX streams are thread-local and must be released before interpreter teardown.
atexit.register(clear_mlx_streams)

__all__ = [
    "BatchResponse",
    "BatchStats",
    "GenerationResult",
    "PromptCacheState",
    "VisionFeatureCache",
    "__version__",
    "apply_chat_template",
    "batch_generate",
    "convert",
    "generate",
    "get_message_json",
    "load",
    "prepare_inputs",
    "process_image",
    "stream_generate",
]
