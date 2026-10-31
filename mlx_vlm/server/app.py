"""FastAPI application exposing the medical text-generation HTTP API."""

import asyncio
import gc
import logging
import os
import secrets
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, List, Optional, Tuple, TypedDict

import mlx.core as mx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import scan_cache_dir
from huggingface_hub.errors import CacheNotFound
from pydantic import TypeAdapter
from starlette.requests import HTTPConnection
from starlette.responses import Response

from mlx_vlm.contracts import (
    ImageSource,
    JSONObject,
    JSONValue,
)
from mlx_vlm.server.generation import (
    GenerationArguments,
    PromptTooLongError,
    ResponseGenerator,
    ServerMetricsStore,
    get_configured_context_limit,
    get_kv_group_size,
    get_quantized_kv_bits,
    get_quantized_kv_start,
    get_top_logprobs_k,
)
from mlx_vlm.server.model_discovery import MODEL_PATHS_ENV, discover_models
from mlx_vlm.server.openai import register_routes as register_openai_routes
from mlx_vlm.server.runtime import ModelCacheRegistry, metrics_store, runtime
from mlx_vlm.server.schemas import ChatLogprobContent, ModelsResponse, TopLogprob
from mlx_vlm.server.services import (
    LoadedModel,
    ProtocolServices,
    TokenDecoder,
)
from mlx_vlm.server.state_types import ModelResources
from mlx_vlm.version import __version__
from mlx_vlm.vision_cache import VisionFeatureCache

DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8080
SERVER_API_KEY_ENV = "MLX_VLM_SERVER_API_KEY"

logger = logging.getLogger("mlx_vlm.server")


def _server_api_key() -> Optional[str]:
    key = os.environ.get(SERVER_API_KEY_ENV)
    return key if key else None


def _require_management_api_key(request: HTTPConnection) -> None:
    api_key = _server_api_key()
    if api_key is None:
        return

    expected = f"Bearer {api_key}"
    supplied = request.headers.get("Authorization", "")
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


class ServedModel(TypedDict):
    """Model listing entry served by the models endpoints."""

    id: str
    object: str
    created: int
    loaded: bool


def _model_info(model_id: str, created: int, *, loaded: bool = False) -> ServedModel:
    model_id = str(model_id)
    return {
        "id": model_id,
        "object": "model",
        "created": created,
        "loaded": loaded,
    }


def _served_model_entries() -> list[ServedModel]:
    created = int(time.time())
    models: dict[str, ServedModel] = {}
    for _, cache in _model_cache_registry().items():
        model_id = cache.get("model_path")
        if not model_id:
            continue
        models[model_id] = _model_info(model_id, created, loaded=True)
    return list(models.values())


def _model_cache_registry() -> ModelCacheRegistry:
    return runtime.model_cache


def _server_runtime_snapshot() -> JSONObject:
    registry = _model_cache_registry()
    default_cache = registry.for_kind("text_generation")
    config = default_cache.get("config")
    text_config = getattr(config, "text_config", None)
    native_context_size = getattr(text_config, "max_position_embeddings", None)
    configured_context_limit = get_configured_context_limit()
    effective_context_limit = (
        min(native_context_size, configured_context_limit)
        if native_context_size is not None and configured_context_limit is not None
        else configured_context_limit or native_context_size
    )
    queue_depth = 0
    if runtime.response_generator is not None and hasattr(
        runtime.response_generator, "requests"
    ):
        try:
            queue_depth = runtime.response_generator.requests.qsize()
        except Exception:
            queue_depth = 0

    return TypeAdapter(JSONObject).validate_python(
        {
            "loaded_model": default_cache.get("model_path", None),
            "loaded_models": {
                group: {
                    "model": cache.get("model_path"),
                    "model_kind": cache.get("model_kind"),
                }
                for group, cache in registry.items()
            },
            "model_kind": default_cache.get("model_kind", "text_generation"),
            "loaded_context_size": native_context_size,
            "configured_context_limit": configured_context_limit,
            "effective_context_limit": effective_context_limit,
            "continuous_batching_enabled": runtime.response_generator is not None,
            "request_queue_depth": queue_depth,
            "preload_failures": dict(runtime.preload_failures),
        },
        strict=True,
    )


async def _preflight_stream_context_budget(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    images: Optional[list[ImageSource]] = None,
    args: GenerationArguments,
):
    """Reject over-budget streaming requests before the HTTP stream starts."""
    if runtime.response_generator is None:
        return
    try:
        await asyncio.to_thread(
            runtime.response_generator.validate_context_budget,
            prompt,
            images=images,
            args=args,
        )
    except PromptTooLongError as e:
        metrics_store().record_failure(
            endpoint=endpoint,
            model=model,
            stream=True,
            error=str(e),
        )
        mx.clear_cache()
        gc.collect()
        raise HTTPException(status_code=400, detail=str(e))


def _count_thinking_tag_tokens(
    text: str,
    thinking_start_token: Optional[str] = None,
    thinking_end_token: Optional[str] = None,
) -> int:
    """Count tokens consumed by thinking tags (excluded from completion_tokens)."""
    count = 0
    if (
        thinking_start_token
        and thinking_end_token
        and thinking_start_token in text
        and thinking_end_token in text
    ):
        return 2
    # <|channel>thought (2 tokens) + <channel|> (1 token) + EOS (1 token)
    if "<|channel>thought" in text and "<channel|>" in text:
        count = 4
    elif "<think>" in text and "</think>" in text:
        count = 2  # <think> and </think> are 1 token each typically
    return count


def _decode_token(
    tokenizer: TokenDecoder, token_id: int
) -> Tuple[str, Optional[List[int]]]:
    """Decode a single token id to its string + UTF-8 bytes."""
    try:
        text = tokenizer.decode([int(token_id)])
    except Exception:
        text = ""
    try:
        token_bytes = list(text.encode("utf-8"))
    except Exception:
        token_bytes = None
    return text, token_bytes


def _make_logprob_content(
    tokenizer: TokenDecoder,
    token_id: int,
    logprob: float,
    top_logprobs: Optional[List[Tuple[int, float]]] = None,
    top_k: int = 0,
) -> "ChatLogprobContent":
    """Build an OpenAI-style logprob entry for a single token."""
    token_text, token_bytes = _decode_token(tokenizer, token_id)
    top_list: List[TopLogprob] = []
    if top_k > 0 and top_logprobs:
        for tid, lp in top_logprobs[:top_k]:
            t_text, t_bytes = _decode_token(tokenizer, tid)
            top_list.append(TopLogprob(token=t_text, logprob=float(lp), bytes=t_bytes))
    return ChatLogprobContent(
        token=token_text,
        logprob=float(logprob),
        bytes=token_bytes,
        top_logprobs=top_list,
    )


# Shared mutable server runtime state.
runtime.metrics = ServerMetricsStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optionally preload a model and request worker shutdown on server exit."""
    model_path = os.environ.pop("MLX_VLM_PRELOAD_MODEL", None)
    if model_path:
        await asyncio.to_thread(get_cached_model, model_path)
    try:
        yield
    finally:
        await asyncio.to_thread(unload_model_sync)


app = FastAPI(
    title="MedMLX Medical VLM Inference API",
    description="Text generation with MedGemma on Apple Silicon.",
    version=__version__,
    lifespan=lifespan,
)
inference_router = APIRouter(
    dependencies=[Depends(_require_management_api_key)],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_IMAGES = 10  # Maximum number of images to process at once


def _unload_model_cache_group(cache_group: str) -> bool:
    registry = _model_cache_registry()
    cache = registry.for_kind(cache_group)
    if not cache:
        return False

    logger.info(
        "Unloading %s model: %s",
        cache_group,
        cache.get("model_path"),
    )

    response_generator = cache.get("response_generator")
    if response_generator is not None:
        logger.info("Stopping response generator.")
        response_generator.stop_and_join()
        if runtime.response_generator is response_generator:
            runtime.response_generator = None

    if "vision_cache" in cache:
        cache["vision_cache"].clear()

    registry.pop(cache_group)
    gc.collect()
    mx.clear_cache()
    return True


def _create_response_generator(
    model_path: str, vision_cache: VisionFeatureCache
) -> ResponseGenerator:
    """Create the worker with the configured medical cache policy."""
    cfg = runtime.config
    kv_bits = cfg.kv_bits if cfg.kv_bits is not None else get_quantized_kv_bits()
    group_size = (
        cfg.kv_group_size if cfg.kv_group_size is not None else get_kv_group_size()
    )
    quantized_start = (
        cfg.quantized_kv_start
        if cfg.quantized_kv_start is not None
        else get_quantized_kv_start()
    )
    return ResponseGenerator(
        model_path=model_path,
        vision_cache=vision_cache,
        kv_bits=kv_bits,
        kv_group_size=group_size,
        quantized_kv_start=quantized_start,
        top_logprobs_k=get_top_logprobs_k(),
    )


def get_cached_model(
    model_path: str,
    *,
    model_kind: str = "auto",
) -> LoadedModel:
    """Load or reuse one medical text-generation worker and its model state."""
    if model_kind not in {"auto", "text_generation"}:
        raise HTTPException(
            status_code=400, detail="Only medical text generation is supported"
        )
    registry = _model_cache_registry()
    cached = registry.for_kind("text_generation")
    cache_key = (
        model_path,
        "text_generation",
        runtime.config.fingerprint(kinds={"text_generation"}),
    )
    if cached and cached.get("cache_key") == cache_key:
        runtime.response_generator = cached.get("response_generator")
        if "model" not in cached or "processor" not in cached or "config" not in cached:
            raise RuntimeError("Cached medical model resources are incomplete")
        return cached["model"], cached["processor"], cached["config"]
    if cached:
        _unload_model_cache_group("text_generation")
    vision_cache = VisionFeatureCache(max_size=runtime.config.vision_cache_size)
    worker = _create_response_generator(model_path, vision_cache)
    ready = False
    try:
        model, processor, config = worker.wait_until_ready()
        ready = True
    finally:
        if not ready:
            worker.stop_and_join()
            vision_cache.clear()
    resources: ModelResources = {
        "cache_key": cache_key,
        "model_path": model_path,
        "model": model,
        "processor": processor,
        "config": config,
        "vision_cache": vision_cache,
        "model_kind": "text_generation",
        "response_generator": worker,
    }
    registry.set("text_generation", resources)
    runtime.response_generator = worker
    return model, processor, config


# Synchronous unload function for internal use
def unload_model_sync() -> bool:
    """Request worker shutdown and remove model-specific cache registrations.

    The worker may continue draining after its bounded join timeout.
    """
    unloaded = False
    for group, _ in list(_model_cache_registry().items()):
        unloaded = _unload_model_cache_group(group) or unloaded
    runtime.response_generator = None
    return unloaded


_protocol_deps = ProtocolServices(
    get_cached_model=get_cached_model,
    preflight_stream_context_budget=_preflight_stream_context_budget,
    count_thinking_tag_tokens=_count_thinking_tag_tokens,
    make_logprob_content=_make_logprob_content,
)
register_openai_routes(inference_router, _protocol_deps)


@inference_router.get("/models", response_model=ModelsResponse)
@inference_router.get(
    "/v1/models",
    response_model=ModelsResponse,
    include_in_schema=False,
)
def models_endpoint(
    model_dir: Annotated[
        Optional[List[str]],
        Query(
            description=(
                "Additional model folder or parent containing model folders on the "
                "server filesystem. Repeat for multiple paths. Applies only to this "
                "request, in addition to the cache and configured model directories."
            )
        ),
    ] = None,
):
    """
    Return cached and loaded models, indicating which are loaded in this process.

    Inspect the shared cache, configured directories, and request-specific paths.
    """
    models = {model["id"]: model for model in _served_model_entries()}

    try:
        hf_cache_info = scan_cache_dir()
    except CacheNotFound:
        hf_cache_info = None

    loaded_paths: set[Path] = set()
    for model_id in models:
        path = Path(model_id).expanduser()
        try:
            if path.is_dir():
                loaded_paths.add(path.resolve())
        except (OSError, RuntimeError):
            continue
    paths = [p for p in os.environ.get(MODEL_PATHS_ENV, "").split(os.pathsep) if p]
    paths.extend(p for p in model_dir or [] if p)
    for model in discover_models(hf_cache_info, paths):
        if model["id"] not in models and model["path"] not in loaded_paths:
            models[model["id"]] = _model_info(model["id"], model["created"])

    return {
        "object": "list",
        "data": sorted(models.values(), key=lambda model: model["id"].lower()),
    }


app.include_router(inference_router)


# MLX_VLM API endpoints


@app.middleware("http")
async def add_server_header(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Add the server name and version header to every response."""
    response = await call_next(request)
    response.headers["Server"] = f"mlx_vlm/{__version__}"
    return response


@app.get("/health")
async def health_check(request: Request):
    """
    Check if the server is healthy and what model is loaded.
    """
    _require_management_api_key(request)
    runtime = _server_runtime_snapshot()
    return {
        "status": "healthy",
        "loaded_model": runtime["loaded_model"],
        "loaded_models": runtime["loaded_models"],
        "loaded_context_size": runtime["loaded_context_size"],
        "configured_context_limit": runtime["configured_context_limit"],
        "effective_context_limit": runtime["effective_context_limit"],
        "continuous_batching_enabled": runtime["continuous_batching_enabled"],
    }


@app.get("/metrics")
@app.get("/v1/metrics", include_in_schema=False)
async def metrics_endpoint(request: Request):
    """Return request metrics with the runtime snapshot under ``server``."""
    _require_management_api_key(request)
    payload = TypeAdapter(JSONObject).validate_python(
        metrics_store().snapshot(), strict=True
    )
    payload["server"] = _server_runtime_snapshot()
    return payload


def _settings_operation(payload: JSONObject):
    if "op" in payload or "values" in payload:
        op = payload.get("op", "merge")
        if op not in ("merge", "replace"):
            raise HTTPException(status_code=400, detail=f"unsupported op {op!r}")
        values = payload.get("values")
        if not isinstance(values, dict):
            raise HTTPException(status_code=400, detail="values must be a JSON object")
        return op, values
    return "merge", payload


@app.get("/v1/settings")
@app.get("/settings", include_in_schema=False)
async def get_runtime_settings(request: Request):
    """Return the setting schema, current values and cache fingerprint."""
    _require_management_api_key(request)
    cfg = runtime.config
    return {
        "schema": cfg.schema(),
        "current": cfg.current(),
        "fingerprint": cfg.fingerprint(),
    }


@app.patch("/v1/settings")
@app.patch("/settings", include_in_schema=False)
async def update_runtime_settings(request: Request):
    """Validate and apply a settings patch.

    Rejects a non-object body or an unknown operation with ``400``; accepted and
    rejected values, the reload kinds and the new fingerprint are returned.
    """
    _require_management_api_key(request)
    payload = TypeAdapter(JSONValue).validate_python(await request.json(), strict=True)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    op, values = _settings_operation(payload)
    before = runtime.config.current()
    applied, rejected = runtime.config.apply_changes(values, op=op)
    current = runtime.config.current()
    changed = {name: value for name, value in current.items() if before[name] != value}
    return {
        "op": op,
        "applied": applied,
        "rejected": rejected,
        "reload_kinds": sorted(runtime.config.reload_kinds(changed)),
        "fingerprint": runtime.config.fingerprint(),
        "current": current,
    }


@app.post("/unload")
async def unload_model_endpoint(request: Request):
    """
    Unload the currently loaded model from memory.
    """
    _require_management_api_key(request)
    snapshot = _server_runtime_snapshot()
    unloaded_info = {
        "model_name": snapshot["loaded_model"],
        "models": snapshot["loaded_models"],
    }

    if not unload_model_sync():  # Use the synchronous unload function
        return {"status": "no_model_loaded", "message": "No model is currently loaded"}

    return {
        "status": "success",
        "message": "Model unloaded successfully",
        "unloaded": unloaded_info,
    }


def main():
    """Run the server command line."""
    from mlx_vlm.server.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
