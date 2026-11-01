"""Start the medical text-generation HTTP service on Metal."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlx.core as mx
import uvicorn

from mlx_vlm.server.model_discovery import MODEL_PATHS_ENV
from mlx_vlm.server.runtime import runtime
from mlx_vlm.server.runtime_config import RuntimeConfig


class ServerArguments(argparse.Namespace):
    """Parsed server command-line arguments."""

    host: str
    port: int
    model: str | None
    model_dir: list[str] | None
    api_key: str | None
    max_tokens: int
    prefill_step_size: int
    kv_bits: int | None
    log_level: str


def parse_arguments(argv: list[str] | None = None) -> ServerArguments:
    """Parse medical model selection, networking and cache controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", help="Preload a MedGemma checkpoint.")
    parser.add_argument("--model-dir", action="append")
    parser.add_argument("--api-key", help="Bearer token for API requests.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--kv-bits", type=int, choices=(2, 4, 8))
    parser.add_argument(
        "--log-level", default="info", choices=("debug", "info", "warning", "error")
    )
    args = parser.parse_args(argv, namespace=ServerArguments())
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    if args.max_tokens <= 0 or args.prefill_step_size <= 0:
        parser.error("token counts must be positive")
    return args


def configure_server(args: ServerArguments) -> None:
    """Apply startup configuration before the model worker is created."""
    if runtime.model_cache:
        raise RuntimeError("Cannot reconfigure a server with a loaded model")
    values = {
        "MLX_VLM_PRELOAD_MODEL": args.model,
        "MLX_VLM_SERVER_API_KEY": args.api_key,
        "MLX_VLM_MAX_TOKENS": str(args.max_tokens),
        "PREFILL_STEP_SIZE": str(args.prefill_step_size),
        "KV_BITS": str(args.kv_bits) if args.kv_bits is not None else None,
    }
    if args.model_dir is not None:
        values[MODEL_PATHS_ENV] = os.pathsep.join(
            str(Path(p).expanduser().resolve()) for p in args.model_dir
        )
    for name, value in values.items():
        if value is not None:
            os.environ[name] = value
    runtime.config = RuntimeConfig.from_env()


def main() -> None:
    """Run a single service process with explicit Metal execution."""
    args = parse_arguments()
    if not mx.metal.is_available():
        raise RuntimeError(
            "The medical server requires an Apple Silicon Mac with Metal"
        )
    mx.set_default_device(mx.gpu)
    configure_server(args)
    uvicorn.run(
        "mlx_vlm.server.app:app",
        host=args.host,
        port=args.port,
        workers=1,
        log_level=args.log_level,
        server_header=False,
    )
