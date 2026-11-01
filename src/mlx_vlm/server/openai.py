"""OpenAI-compatible chat completion routes for the medical server."""

import asyncio
import gc
import json
import logging
import time
import uuid
from collections.abc import Callable, Sequence
from typing import List, Optional, Tuple

import mlx.core as mx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, TypeAdapter

from mlx_vlm.contracts import (
    ImageSource,
    JSONObject,
    JSONValue,
)
from mlx_vlm.generate import generate, stream_generate
from mlx_vlm.generate.types import GenerateTextKwargs, resolve_tokenizer
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.server.generation import (
    GenerationMetrics,
    PromptTooLongError,
    StreamingToken,
    build_metrics_envelope,
)
from mlx_vlm.server.request_normalization import (
    as_plain_dict,
    build_gen_args,
    chat_message_to_prompt,
    normalize_instruction_messages,
)
from mlx_vlm.server.runtime import metrics_store, runtime
from mlx_vlm.server.schemas import (
    ChatChoice,
    ChatLogprobs,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatStreamChoice,
    ChatStreamChunk,
    GenerationTimings,
    StreamingTimings,
    UsageStats,
)
from mlx_vlm.server.services import (
    CachedModelLoader,
    ContextPreflight,
    LogprobFormatter,
    ProtocolServices,
    TokenDecoder,
    generation_kwargs,
)
from mlx_vlm.server.thinking import (
    finish_content_streams,
    make_response_stream_state,
    prompt_has_open_thinking,
    split_thinking,
)

logger = logging.getLogger("mlx_vlm.server")

get_cached_model: CachedModelLoader
_preflight_stream_context_budget: ContextPreflight
_count_thinking_tag_tokens: Callable[[str, str | None, str | None], int]
_make_logprob_content: LogprobFormatter
_MISSING_INPUT_DETAIL = (
    "Request must include at least one non-empty message content or media input."
)


def _has_non_empty_text(value: JSONValue) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _content_has_effective_input(content: JSONValue) -> bool:
    if _has_non_empty_text(content):
        return True
    if content is None:
        return False
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in ("text", "input_text", "output_text"):
                if _has_non_empty_text(item.get("text")) or _has_non_empty_text(
                    item.get("content")
                ):
                    return True
            elif item_type in ("image_url", "input_image"):
                image = item.get("image_url") or item.get("file_id")
                if isinstance(image, dict):
                    image = image.get("url")
                if image:
                    return True
            elif _content_has_effective_input(item.get("content")):
                return True
        return False
    if isinstance(content, dict):
        return _content_has_effective_input([content])
    return bool(str(content).strip())


def _message_has_effective_input(message: JSONValue | BaseModel) -> bool:
    message = as_plain_dict(message)
    if not isinstance(message, dict):
        return False
    return (
        _content_has_effective_input(message.get("content"))
        or _has_non_empty_text(message.get("reasoning_content"))
        or _has_non_empty_text(message.get("reasoning"))
    )


def _ensure_effective_input(
    messages: Sequence[JSONObject | BaseModel],
    *,
    images: Sequence[ImageSource] | None = None,
) -> None:
    if any(image for image in (images or [])):
        return
    if any(_message_has_effective_input(message) for message in messages or []):
        return
    raise HTTPException(status_code=400, detail=_MISSING_INPUT_DETAIL)


def _final_chat_chunk(
    request_id: str,
    model: str,
    finish_reason: str,
    predicted_per_second: Optional[float] = None,
) -> ChatStreamChunk:
    return ChatStreamChunk(
        id=request_id,
        created=int(time.time()),
        model=model,
        timings=StreamingTimings(predicted_per_second=predicted_per_second),
        choices=[
            ChatStreamChoice(
                finish_reason=finish_reason,
                delta=ChatMessage(role="assistant"),
            )
        ],
    )


def _chat_usage_chunk(
    request_id: str,
    model: str,
    metrics: GenerationMetrics,
    prompt_tokens: int,
    output_tokens: int,
) -> ChatStreamChunk:
    return ChatStreamChunk(
        id=request_id,
        created=int(time.time()),
        model=model,
        usage=UsageStats.from_metrics(metrics, prompt_tokens, output_tokens),
        choices=[],
        timings=GenerationTimings.from_metrics(metrics, prompt_tokens, output_tokens),
    )


def register_routes(app: APIRouter, deps: ProtocolServices) -> None:
    """Bind the chat completion endpoints and install application services."""
    global get_cached_model
    global _preflight_stream_context_budget
    global _count_thinking_tag_tokens, _make_logprob_content

    get_cached_model = deps.get_cached_model
    _preflight_stream_context_budget = deps.preflight_stream_context_budget
    _count_thinking_tag_tokens = deps.count_thinking_tag_tokens
    _make_logprob_content = deps.make_logprob_content

    app.post("/chat/completions", response_model=None)(chat_completions_endpoint)
    app.post("/v1/chat/completions", response_model=None, include_in_schema=False)(
        chat_completions_endpoint
    )


# OpenAI-compatible endpoints


async def chat_completions_endpoint(request: ChatRequest, http_request: Request):
    """Generate from chat messages and optional images, synchronously or as SSE.

    Message instructions are normalized for the model's chat template.
    Successful streams emit a final
    finish-reason chunk, an optional usage chunk when requested, then ``[DONE]``;
    stream errors are emitted as an error data event after the response starts.
    """

    request_start = time.perf_counter()
    try:
        kwargs: GenerateTextKwargs = {}

        if request.resize_shape is not None:
            if len(request.resize_shape) not in [1, 2]:
                raise HTTPException(
                    status_code=400,
                    detail="resize_shape must contain exactly two integers (height, width)",
                )
            kwargs["resize_shape"] = (
                (request.resize_shape[0], request.resize_shape[0])
                if len(request.resize_shape) == 1
                else (request.resize_shape[0], request.resize_shape[1])
            )

        images: list[ImageSource] = []
        processed_messages: list[JSONObject] = []
        for message in request.messages:
            if isinstance(message.content, list):
                if message.role == "user":
                    for raw_item in message.content:
                        item = TypeAdapter(JSONObject).validate_python(
                            raw_item, strict=True
                        )
                        item_type = item.get("type")
                        if item_type == "input_image":
                            images.append(
                                TypeAdapter(str).validate_python(
                                    item["image_url"], strict=True
                                )
                            )
                        elif item_type == "image_url":
                            image_url = TypeAdapter(JSONObject).validate_python(
                                item["image_url"], strict=True
                            )
                            images.append(
                                TypeAdapter(str).validate_python(
                                    image_url["url"], strict=True
                                )
                            )
            processed_messages.append(
                chat_message_to_prompt(message.model_dump(exclude_none=True))
            )

        normalize_instruction_messages(processed_messages)
        _ensure_effective_input(processed_messages, images=images)

        model, processor, config = get_cached_model(request.model)

        try:
            gen_args = build_gen_args(request, processor)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        formatted_prompt = apply_chat_template(
            processor,
            config,
            processed_messages,
            num_images=len(images),
            **gen_args.to_template_kwargs(),
        )

        if not isinstance(formatted_prompt, str):
            raise TypeError("Chat template must return text")

        logger.debug(
            "chat/completions request: model=%s images=%d "
            "max_tokens=%s temp=%s stream=%s",
            request.model,
            len(images),
            gen_args.max_tokens,
            gen_args.temperature,
            request.stream,
        )

        if request.stream:
            # Streaming response using ResponseGenerator for continuous batching
            metrics_store().begin_request(
                endpoint="/chat/completions",
                model=request.model,
                stream=True,
            )
            await _preflight_stream_context_budget(
                endpoint="/chat/completions",
                model=request.model,
                prompt=formatted_prompt,
                images=list[ImageSource](images) if images else None,
                args=gen_args,
            )

            async def stream_generator():
                token_iterator = None
                token_iter = None  # For ResponseGenerator cleanup
                metrics_finalized = False
                metrics = GenerationMetrics()
                finish_reason = None
                emit_usage = bool(
                    request.stream_options and request.stream_options.include_usage
                )
                try:
                    output_tokens = 0
                    full_output = ""
                    output_text = ""
                    stream_prompt_tokens = 0

                    # Use ResponseGenerator if available, otherwise fall back to stream_generate
                    if (response_generator := runtime.response_generator) is not None:
                        # generate() does blocking Queue.get — run off event loop
                        ctx, token_iter = await asyncio.to_thread(
                            response_generator.generate,
                            formatted_prompt,
                            list[ImageSource](images) if images else None,
                            args=gen_args,
                        )

                        stream_prompt_tokens = ctx.prompt_tokens
                        output_tokens = 0
                        request_id = f"chatcmpl-{uuid.uuid4()}"
                        thinking_state = make_response_stream_state(
                            processor,
                            prompt_has_open_thinking(
                                formatted_prompt,
                                gen_args.enable_thinking,
                                gen_args.thinking_start_token,
                                gen_args.thinking_end_token,
                            ),
                            gen_args.thinking_start_token,
                            gen_args.thinking_end_token,
                            prompt_prefix=formatted_prompt,
                        )
                        full_output = ""

                        def _next_token() -> StreamingToken | None:
                            try:
                                return next(token_iter)
                            except StopIteration:
                                return None

                        while True:
                            token = await asyncio.to_thread(_next_token)
                            if token is None:
                                break
                            output_tokens += getattr(token, "token_count", 1)
                            full_output += token.text
                            chunk_rate = metrics.record_chunk(token)

                            # Detect thinking boundaries
                            thinking_delta = thinking_state.feed(
                                token.text, last=bool(token.finish_reason)
                            )
                            delta_reasoning = thinking_delta.reasoning
                            delta_content = thinking_delta.content

                            chunk_logprobs = None
                            if (
                                request.logprobs
                                and token.finish_reason != "stop"
                                and token.logprobs is not None
                            ):
                                tokenizer = response_generator.tokenizer
                                if not isinstance(tokenizer, TokenDecoder):
                                    raise TypeError(
                                        "Log probabilities require a tokenizer with decode()"
                                    )
                                req_top_k = int(request.top_logprobs or 0)
                                chunk_logprobs = ChatLogprobs(
                                    content=[
                                        _make_logprob_content(
                                            tokenizer,
                                            token.token,
                                            token.logprobs,
                                            top_logprobs=token.top_logprobs,
                                            top_k=req_top_k,
                                        )
                                    ]
                                )

                            # Skip empty deltas (e.g. tokens consumed as thinking markers)
                            has_payload = (
                                bool(delta_content)
                                or bool(delta_reasoning)
                                or chunk_logprobs is not None
                            )
                            if has_payload:
                                choices = [
                                    ChatStreamChoice(
                                        delta=ChatMessage(
                                            role="assistant",
                                            content=delta_content,
                                            reasoning=delta_reasoning,
                                        ),
                                        logprobs=chunk_logprobs,
                                    )
                                ]
                                chunk_data = ChatStreamChunk(
                                    id=request_id,
                                    created=int(time.time()),
                                    model=request.model,
                                    choices=choices,
                                    timings=StreamingTimings(
                                        predicted_per_second=chunk_rate
                                    ),
                                )

                                yield f"data: {chunk_data.model_dump_json()}\n\n"

                            if token.finish_reason:
                                finish_reason = token.finish_reason
                                break

                        tail_reasoning, tail = finish_content_streams(thinking_state)
                        if tail or tail_reasoning:
                            chunk_data = ChatStreamChunk(
                                id=request_id,
                                created=int(time.time()),
                                model=request.model,
                                choices=[
                                    ChatStreamChoice(
                                        delta=ChatMessage(
                                            role="assistant",
                                            content=tail,
                                            reasoning=tail_reasoning,
                                        )
                                    )
                                ],
                                timings=StreamingTimings(
                                    predicted_per_second=metrics.rate
                                ),
                            )
                            yield f"data: {chunk_data.model_dump_json()}\n\n"

                        finish_reason = finish_reason or "stop"
                        chunk_data = _final_chat_chunk(
                            request_id,
                            request.model,
                            finish_reason,
                            metrics.rate,
                        )
                        yield f"data: {chunk_data.model_dump_json()}\n\n"
                        if emit_usage:
                            chunk_data = _chat_usage_chunk(
                                request_id,
                                request.model,
                                metrics,
                                ctx.prompt_tokens,
                                output_tokens,
                            )
                            yield f"data: {chunk_data.model_dump_json()}\n\n"
                    else:
                        # Fallback to stream_generate
                        token_iterator = stream_generate(
                            model=model,
                            processor=processor,
                            prompt=formatted_prompt,
                            image=images,
                            **generation_kwargs(gen_args, kwargs),
                        )

                        request_id = f"chatcmpl-{uuid.uuid4()}"
                        output_text = ""
                        thinking_state = make_response_stream_state(
                            processor,
                            prompt_has_open_thinking(
                                formatted_prompt,
                                gen_args.enable_thinking,
                                gen_args.thinking_start_token,
                                gen_args.thinking_end_token,
                            ),
                            gen_args.thinking_start_token,
                            gen_args.thinking_end_token,
                            prompt_prefix=formatted_prompt,
                        )
                        for chunk in token_iterator:

                            output_text += chunk.text
                            stream_prompt_tokens = chunk.prompt_tokens
                            output_tokens = chunk.generation_tokens
                            chunk_rate = metrics.record_chunk(chunk)
                            chunk_finish = getattr(chunk, "finish_reason", None)
                            if chunk_finish is not None:
                                finish_reason = chunk_finish

                            thinking_delta = thinking_state.feed(
                                chunk.text, last=bool(chunk_finish)
                            )
                            delta_content = thinking_delta.content
                            if delta_content or thinking_delta.reasoning:
                                choices = [
                                    ChatStreamChoice(
                                        delta=ChatMessage(
                                            role="assistant",
                                            content=delta_content,
                                            reasoning=thinking_delta.reasoning,
                                        )
                                    )
                                ]
                                chunk_data = ChatStreamChunk(
                                    id=request_id,
                                    created=int(time.time()),
                                    model=request.model,
                                    choices=choices,
                                    timings=StreamingTimings(
                                        predicted_per_second=chunk_rate
                                    ),
                                )

                                yield f"data: {chunk_data.model_dump_json()}\n\n"
                                await asyncio.sleep(0.01)

                        tail_reasoning, tail = finish_content_streams(thinking_state)
                        if tail or tail_reasoning:
                            chunk_data = ChatStreamChunk(
                                id=request_id,
                                created=int(time.time()),
                                model=request.model,
                                choices=[
                                    ChatStreamChoice(
                                        delta=ChatMessage(
                                            role="assistant",
                                            content=tail,
                                            reasoning=tail_reasoning,
                                        )
                                    )
                                ],
                                timings=StreamingTimings(
                                    predicted_per_second=metrics.rate
                                ),
                            )
                            yield f"data: {chunk_data.model_dump_json()}\n\n"

                        finish_reason = finish_reason or "stop"
                        chunk_data = _final_chat_chunk(
                            request_id,
                            request.model,
                            finish_reason,
                            metrics.rate,
                        )
                        yield f"data: {chunk_data.model_dump_json()}\n\n"
                        if emit_usage:
                            chunk_data = _chat_usage_chunk(
                                request_id,
                                request.model,
                                metrics,
                                stream_prompt_tokens,
                                output_tokens,
                            )
                            yield f"data: {chunk_data.model_dump_json()}\n\n"

                    metrics_text = full_output or output_text
                    completion_tokens = max(
                        0,
                        output_tokens
                        - _count_thinking_tag_tokens(
                            metrics_text,
                            gen_args.thinking_start_token,
                            gen_args.thinking_end_token,
                        ),
                    )
                    envelope = build_metrics_envelope(
                        endpoint="/chat/completions",
                        model=request.model,
                        stream=True,
                        backend=(
                            "continuous_batching"
                            if runtime.response_generator is not None
                            else "generate"
                        ),
                        prompt_tokens=stream_prompt_tokens,
                        completion_tokens=completion_tokens,
                        generated_tokens=output_tokens,
                        request_elapsed_s=time.perf_counter() - request_start,
                        request_started_s=request_start,
                        token_times=metrics.token_times,
                        prompt_tps=metrics.prompt_tps,
                        generation_tps=metrics.generation_tps,
                        peak_memory_gb=metrics.peak_memory or None,
                        finish_reason=finish_reason,
                        image_count=len(images),
                        thinking_enabled=bool(gen_args.enable_thinking),
                    )
                    metrics_store().record_success(envelope)
                    metrics_finalized = True

                    # Signal stream end
                    yield "data: [DONE]\n\n"

                    elapsed = time.perf_counter() - request_start
                    logger.debug(
                        "chat/completions stream done: tokens=%d total_time=%.2fs",
                        output_tokens,
                        elapsed,
                    )

                except Exception as e:
                    if not metrics_finalized:
                        metrics_store().record_failure(
                            endpoint="/chat/completions",
                            model=request.model,
                            stream=True,
                            error=str(e),
                        )
                        metrics_finalized = True
                    logger.exception("Chat completion stream generation failed: %s", e)
                    error_data = json.dumps({"error": str(e)})
                    yield f"data: {error_data}\n\n"

                finally:
                    # Close the token iterator to trigger cleanup (important for ResponseGenerator)
                    if token_iter is not None:
                        try:
                            token_iter.close()
                        except Exception:
                            pass
                    if not metrics_finalized:
                        metrics_store().record_failure(
                            endpoint="/chat/completions",
                            model=request.model,
                            stream=True,
                            error="stream_closed_before_completion",
                        )
                    logger.debug("Chat completion stream closed.")

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        else:
            # Non-streaming response
            metrics_store().begin_request(
                endpoint="/chat/completions",
                model=request.model,
                stream=False,
            )
            try:
                full_text = ""
                prompt_tokens = 0
                output_tokens = 0
                metrics = GenerationMetrics()
                finish_reason = None

                collected_logprobs: List[
                    Tuple[int, float, Optional[List[Tuple[int, float]]]]
                ] = []

                if (response_generator := runtime.response_generator) is not None:

                    def _blocking_generate():
                        metrics = GenerationMetrics()
                        logprobs: List[
                            Tuple[int, float, Optional[List[Tuple[int, float]]]]
                        ] = []
                        text = ""
                        pt = gt = 0
                        fr = None
                        ctx, token_iter = response_generator.generate(
                            prompt=formatted_prompt,
                            images=list[ImageSource](images) if images else None,
                            args=gen_args,
                        )
                        pt = ctx.prompt_tokens
                        for token in token_iter:
                            text += token.text
                            gt += getattr(token, "token_count", 1)
                            metrics.record_chunk(token)
                            if (
                                request.logprobs
                                and token.finish_reason != "stop"
                                and token.logprobs is not None
                            ):
                                logprobs.append(
                                    (token.token, token.logprobs, token.top_logprobs)
                                )
                            if token.finish_reason:
                                fr = token.finish_reason
                                break
                        try:
                            token_iter.close()
                        except Exception:
                            pass
                        return pt, text, gt, fr, metrics, logprobs

                    (
                        prompt_tokens,
                        full_text,
                        output_tokens,
                        finish_reason,
                        metrics,
                        collected_logprobs,
                    ) = await asyncio.to_thread(_blocking_generate)
                else:
                    gen_result = generate(
                        model=model,
                        processor=processor,
                        prompt=formatted_prompt,
                        image=images,
                        verbose=logger.isEnabledFor(logging.DEBUG),
                        **generation_kwargs(gen_args, kwargs),
                    )
                    full_text = gen_result.text
                    prompt_tokens = gen_result.prompt_tokens
                    output_tokens = gen_result.generation_tokens
                    metrics.record_result(gen_result)
                    finish_reason = getattr(gen_result, "finish_reason", None) or "stop"

                mx.clear_cache()
                gc.collect()

                reasoning, content = split_thinking(
                    full_text,
                    gen_args.thinking_start_token,
                    gen_args.thinking_end_token,
                    prompt_has_open_thinking(
                        formatted_prompt,
                        gen_args.enable_thinking,
                        gen_args.thinking_start_token,
                        gen_args.thinking_end_token,
                    ),
                    processor=processor,
                    prompt_prefix=formatted_prompt,
                )

                # Count raw generated tokens minus thinking tag tokens
                completion_tokens = output_tokens - _count_thinking_tag_tokens(
                    full_text,
                    gen_args.thinking_start_token,
                    gen_args.thinking_end_token,
                )

                usage_stats = UsageStats.from_metrics(
                    metrics, prompt_tokens, completion_tokens
                )

                response_logprobs = None
                if request.logprobs and collected_logprobs:
                    tokenizer = resolve_tokenizer(processor)
                    if not isinstance(tokenizer, TokenDecoder):
                        raise TypeError(
                            "Log probabilities require a tokenizer with decode()"
                        )
                    req_top_k = int(request.top_logprobs or 0)
                    response_logprobs = ChatLogprobs(
                        content=[
                            _make_logprob_content(
                                tokenizer,
                                tid,
                                lp,
                                top_logprobs=top_lps,
                                top_k=req_top_k,
                            )
                            for tid, lp, top_lps in collected_logprobs
                        ]
                    )

                choices = [
                    ChatChoice(
                        finish_reason=finish_reason or "stop",
                        message=ChatMessage(
                            role="assistant",
                            content=content if content else None,
                            reasoning=reasoning,
                        ),
                        logprobs=response_logprobs,
                    )
                ]
                result = ChatResponse(
                    id=f"chatcmpl-{uuid.uuid4()}",
                    created=int(time.time()),
                    model=request.model,
                    usage=usage_stats,
                    choices=choices,
                    timings=GenerationTimings.from_metrics(
                        metrics, prompt_tokens, output_tokens
                    ),
                )

                elapsed = time.perf_counter() - request_start
                logger.debug(
                    "chat/completions done: prompt_tokens=%d completion_tokens=%d "
                    "total_time=%.2fs peak_memory=%.2fGB",
                    prompt_tokens,
                    completion_tokens,
                    elapsed,
                    metrics.peak_memory,
                )
                if logger.isEnabledFor(logging.DEBUG):
                    resp_text = content or ""
                    logger.debug(
                        "  response: %s",
                        resp_text[:200] + ("..." if len(resp_text) > 200 else ""),
                    )

                envelope = build_metrics_envelope(
                    endpoint="/chat/completions",
                    model=request.model,
                    stream=False,
                    backend=(
                        "continuous_batching"
                        if runtime.response_generator is not None
                        else "generate"
                    ),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    generated_tokens=output_tokens,
                    request_elapsed_s=elapsed,
                    request_started_s=request_start,
                    token_times=metrics.token_times,
                    prompt_tps=metrics.prompt_tps,
                    generation_tps=metrics.generation_tps,
                    peak_memory_gb=metrics.peak_memory or None,
                    finish_reason=finish_reason or "stop",
                    image_count=len(images),
                    thinking_enabled=bool(gen_args.enable_thinking),
                )
                metrics_store().record_success(envelope)

                return result

            except PromptTooLongError as e:
                metrics_store().record_failure(
                    endpoint="/chat/completions",
                    model=request.model,
                    stream=False,
                    error=str(e),
                )
                mx.clear_cache()
                gc.collect()
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                metrics_store().record_failure(
                    endpoint="/chat/completions",
                    model=request.model,
                    stream=False,
                    error=str(e),
                )
                logger.exception("Chat completion generation failed: %s", e)
                mx.clear_cache()
                gc.collect()
                raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions (like model loading failure)
        raise http_exc
    except Exception as e:
        # Catch unexpected errors
        logger.exception("Unexpected error in /chat/completions endpoint: %s", e)
        mx.clear_cache()
        gc.collect()
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )
