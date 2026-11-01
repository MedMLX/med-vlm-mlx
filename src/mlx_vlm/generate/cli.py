"""Text generation from medical image and text prompts on Metal."""

from __future__ import annotations

import argparse
import math
from typing import cast

import mlx.core as mx

from mlx_vlm.contracts import JSONObject
from mlx_vlm.generate.common import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
from mlx_vlm.generate.dispatch import generate
from mlx_vlm.generate.types import GenerateTextKwargs, ProcessorLike
from mlx_vlm.models.protocols import require_generation_model
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

DEFAULT_MODEL_PATH = "mlx-community/medgemma-1.5-4b-it-4bit"
DEFAULT_SEED = 0
DEFAULT_THINKING_START_TOKEN = "<think>"
DEFAULT_THINKING_END_TOKEN = "</think>"


class GenerationCLIArguments(argparse.Namespace):
    """Validated parser values for the medical text-generation entry point."""

    model: str
    revision: str | None
    image: list[str] | None
    prompt: list[str]
    system: str | None
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    seed: int
    prefill_step_size: int | None
    verbose: bool


def positive_int(value: str) -> int:
    """Parse a positive token count before a checkpoint is loaded."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_arguments(argv: list[str] | None = None) -> GenerationCLIArguments:
    """Parse the supported medical inputs and autoregressive sampling controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--revision", help="Hugging Face checkpoint revision.")
    parser.add_argument("--image", nargs="+", help="Image paths or URLs.")
    parser.add_argument("--prompt", nargs="+", required=True)
    parser.add_argument("--system", help="Optional system instruction.")
    parser.add_argument("--max-tokens", type=positive_int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--prefill-step-size", type=positive_int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv, namespace=GenerationCLIArguments())
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be nonnegative and finite")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.top_k < 0:
        parser.error("--top-k must be nonnegative")
    return args


def main() -> None:
    """Load a maintained architecture and generate text on the Metal device."""
    args = parse_arguments()
    if not mx.metal.is_available():
        raise RuntimeError("Medical inference requires an Apple Silicon Mac with Metal")
    mx.set_default_device(mx.gpu)
    mx.random.seed(args.seed)
    loaded_model, loaded_processor = load(
        args.model,
        revision=args.revision,
        trust_remote_code=False,
    )
    model = require_generation_model(loaded_model)
    processor = cast(ProcessorLike, loaded_processor)
    messages: list[JSONObject] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": " ".join(args.prompt)})
    prompt = apply_chat_template(
        processor,
        model.config,
        messages,
        num_images=len(args.image or []),
    )
    if not isinstance(prompt, str):
        raise TypeError("The medical chat template must return text")
    options: GenerateTextKwargs = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    if args.prefill_step_size is not None:
        options["prefill_step_size"] = args.prefill_step_size
    result = generate(
        model,
        processor,
        prompt,
        image=args.image,
        verbose=args.verbose,
        **options,
    )
    if not args.verbose:
        print(result.text)
