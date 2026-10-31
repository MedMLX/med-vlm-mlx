"""Host-only request contracts without model loading."""

from dataclasses import dataclass
from queue import Queue
from threading import Lock, Thread
from typing import Callable, Protocol, cast

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from mlx_vlm.contracts import JSONObject, array_int_list
from mlx_vlm.sample_utils import make_sampler
from mlx_vlm.server.app import app
from mlx_vlm.server.generation import (
    GenerationArguments,
    QueuedGenerationRequest,
    ResponseGenerator,
)
from mlx_vlm.server.request_normalization import build_gen_args
from mlx_vlm.server.schemas import ChatRequest
from mlx_vlm.server.thinking import make_response_stream_state, split_thinking


@dataclass
class SamplingConfig:
    temperature: float | str = 0.8
    top_p: float = 0.9
    top_k: int = 10


@dataclass
class ConfiguredProcessor:
    config: SamplingConfig


class PositionedSampler(Protocol):
    def sample_target(
        self, logprobs: mx.array, *, row_ids: list[int], positions: list[int]
    ) -> mx.array: ...


def worker_sampler(args: GenerationArguments) -> Callable[[mx.array], mx.array] | None:
    worker = ResponseGenerator.__new__(ResponseGenerator)
    factory = cast(
        Callable[[GenerationArguments], Callable[[mx.array], mx.array] | None],
        getattr(worker, "_make_sampler"),
    )
    return factory(args)


def test_generation_arguments_preserve_explicit_falsy_overrides() -> None:
    payload: dict[str, object] = {
        "model": "medical",
        "temperature": 0.0,
        "top_p": 0.0,
        "top_k": 0,
        "enable_thinking": False,
    }
    request = ChatRequest.model_validate(
        {**payload, "messages": [{"role": "user", "content": "Describe"}]}
    )
    args = build_gen_args(request, ConfiguredProcessor(SamplingConfig()))
    assert (args.temperature, args.top_p, args.top_k, args.enable_thinking) == (
        0.0,
        0.0,
        0,
        False,
    )


@pytest.mark.parametrize("top_n_sigma", [0.0, 2.0])
def test_chat_min_p_reaches_worker_sampler(top_n_sigma: float) -> None:
    request = ChatRequest.model_validate(
        {
            "model": "medical",
            "messages": [{"role": "user", "content": "Describe"}],
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 1.0,
            "top_n_sigma": top_n_sigma,
            "seed": 7,
        }
    )
    args = build_gen_args(request, ConfiguredProcessor(SamplingConfig()))
    sampler = worker_sampler(args)
    assert sampler is not None
    mx.random.seed(7)
    logits = mx.broadcast_to(mx.array([[0.0, -0.01]]), (32, 2))
    assert sampler(logits).tolist() == [0] * 32


def test_seeded_min_p_sampler_is_position_deterministic() -> None:
    request = ChatRequest.model_validate(
        {
            "model": "medical",
            "messages": [{"role": "user", "content": "Describe"}],
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.2,
            "seed": 7,
        }
    )
    args = build_gen_args(request, ConfiguredProcessor(SamplingConfig()))
    sampler = worker_sampler(args)
    assert sampler is not None
    positioned = cast(PositionedSampler, sampler)
    logits = mx.broadcast_to(mx.array([[0.0, -0.1]]), (8, 2))
    sample = positioned.sample_target(logits, row_ids=list(range(8)), positions=[3] * 8)
    mx.random.seed(123)
    repeated = positioned.sample_target(
        logits, row_ids=list(range(8)), positions=[3] * 8
    )
    assert sample.tolist() == repeated.tolist()


def test_seeded_advanced_filters_match_direct_sampler_and_ignore_global_rng() -> None:
    request = ChatRequest.model_validate(
        {
            "model": "medical",
            "messages": [{"role": "user", "content": "Describe"}],
            "temperature": 0.9,
            "top_p": 0.95,
            "top_k": 3,
            "min_p": 0.1,
            "top_n_sigma": 2.0,
            "p_less": True,
            "typical_p": 0.95,
            "seed": 19,
        }
    )
    args = build_gen_args(request, ConfiguredProcessor(SamplingConfig()))
    sampler = worker_sampler(args)
    assert sampler is not None
    logits = mx.broadcast_to(mx.array([[0.0, -0.1, -0.3, -0.7]]), (32, 4))
    direct = make_sampler(
        temp=0.9,
        top_p=0.95,
        top_k=3,
        min_p=0.1,
        top_n_sigma=2.0,
        p_less=True,
        typical_p=0.95,
    )
    mx.random.seed(123)
    expected = direct(logits).tolist()
    mx.random.seed(123)
    assert sampler(logits).tolist() == expected
    request = request.model_copy(
        update={"top_n_sigma": 0.0, "p_less": False, "min_p": 0.0, "top_k": 0}
    )
    typical_sampler = worker_sampler(
        build_gen_args(request, ConfiguredProcessor(SamplingConfig()))
    )
    assert typical_sampler is not None
    positioned = cast(PositionedSampler, typical_sampler)
    row_ids = list(range(32))
    positions = [1] * 32
    mx.random.seed(1)
    first = array_int_list(
        positioned.sample_target(logits, row_ids=row_ids, positions=positions)
    )
    mx.random.seed(2)
    second = array_int_list(
        positioned.sample_target(logits, row_ids=row_ids, positions=positions)
    )
    assert first == second
    assert len(set(first)) > 1


def test_worker_admission_keeps_incompatible_sampling_requests_in_fifo_order() -> None:
    worker = ResponseGenerator.__new__(ResponseGenerator)
    worker.requests = Queue()
    setattr(worker, "_stop", False)
    setattr(worker, "_deferred_request", None)
    collect = cast(
        Callable[..., tuple[list[QueuedGenerationRequest], bool]],
        getattr(worker, "_collect_pending_requests"),
    )

    def request(seed: int, logprobs: bool = False) -> QueuedGenerationRequest:
        return QueuedGenerationRequest(
            rqueue=Queue(),
            raw_inputs={"input_ids": mx.array([[1]])},
            prompt_tokens=1,
            args=GenerationArguments(seed=seed, logprobs=logprobs),
        )

    first, peer, different_seed, later_peer, different_logs = (
        request(7),
        request(7),
        request(19),
        request(7),
        request(7, True),
    )
    for item in (first, peer, different_seed, later_peer, different_logs):
        worker.requests.put(item)
    admitted, stopped = collect(active=False, idle_timeout=0.01)
    assert admitted == [first, peer] and not stopped
    admitted, stopped = collect(
        active=True, sampling_key=first.args.batch_sampling_key()
    )
    assert admitted == [] and not stopped
    for item in (different_seed, later_peer, different_logs):
        admitted, stopped = collect(active=False, idle_timeout=0.01)
        assert admitted == [item] and not stopped
    setattr(worker, "_stop", True)
    worker.requests.put(None)
    admitted, stopped = collect(active=False, idle_timeout=0.01)
    assert admitted == [] and stopped


@pytest.mark.parametrize("admit_before_stop", [True, False])
def test_worker_shutdown_orders_admission_and_rejects_late_requests(
    admit_before_stop: bool,
) -> None:
    worker = ResponseGenerator.__new__(ResponseGenerator)
    worker.requests = Queue()
    setattr(worker, "_stop", False)
    setattr(worker, "_requests_lock", Lock())
    thread = Thread(target=lambda: None)
    thread.start()
    setattr(worker, "_thread", thread)
    enqueue = cast(
        Callable[[QueuedGenerationRequest], None], getattr(worker, "_enqueue_request")
    )
    request = QueuedGenerationRequest(
        rqueue=Queue(),
        raw_inputs={"input_ids": mx.array([[1]])},
        prompt_tokens=1,
        args=GenerationArguments(),
    )
    if admit_before_stop:
        enqueue(request)
    worker.stop_and_join()
    worker.stop_and_join()
    with pytest.raises(RuntimeError, match="shutting down"):
        enqueue(request)
    if admit_before_stop:
        assert worker.requests.get_nowait() is request
    assert worker.requests.get_nowait() is None
    assert worker.requests.empty()


def test_generation_arguments_reject_malformed_model_default() -> None:
    request = ChatRequest.model_validate(
        {"model": "medical", "messages": [{"role": "user", "content": "Describe"}]}
    )
    with pytest.raises(ValidationError, match="float"):
        build_gen_args(request, ConfiguredProcessor(SamplingConfig(temperature="0.8")))


@pytest.mark.parametrize(
    ("schema", "body", "match"),
    [
        (ChatRequest, {"tools": [{"type": "function"}]}, "tools is unsupported"),
        (ChatRequest, {"tool_choice": "auto"}, "tool_choice is unsupported"),
        (
            ChatRequest,
            {"messages": [{"role": "tool", "content": "42", "tool_call_id": "c"}]},
            "Tool calling",
        ),
        (
            ChatRequest,
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "video_url", "video_url": "a.mp4"}],
                    }
                ]
            },
            "Video inputs",
        ),
        (
            ChatRequest,
            {"context_management": [{"type": "compaction"}]},
            "context_management is unsupported",
        ),
        (ChatRequest, {"adapter_path": "lora"}, "adapter_path is unsupported"),
        (
            ChatRequest,
            {"response_format": {"type": "json_object"}},
            "response_format is unsupported",
        ),
    ],
)
def test_requests_reject_removed_features_instead_of_ignoring_them(
    schema: type[ChatRequest],
    body: JSONObject,
    match: str,
) -> None:
    base: JSONObject = {
        "model": "medical",
        "messages": [{"role": "user", "content": "Describe"}],
    }
    with pytest.raises(ValidationError, match=match):
        schema.model_validate({**base, **body})


@pytest.mark.parametrize("starts_in_thinking", [False, True])
def test_streamed_thinking_markers_are_removed_across_chunks(
    starts_in_thinking: bool,
) -> None:
    state = make_response_stream_state(None, enable_thinking=starts_in_thinking)
    reasoning: list[str] = []
    content: list[str] = []
    for chunk in ("<th", "ink>secret</thi", "nk>answer"):
        delta = state.feed(chunk)
        reasoning.append(delta.reasoning or "")
        content.append(delta.content or "")
    final = state.feed("", last=True)
    reasoning.append(final.reasoning or "")
    content.append(final.content or "")
    assert "".join(reasoning) == "secret"
    assert "".join(content) == "answer"


def test_tokenizer_response_parser_uses_open_prompt_prefix() -> None:
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer.from_str(
        '{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],'
        '"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,'
        '"model":{"type":"WordLevel","vocab":{"[UNK]":0},"unk_token":"[UNK]"}}'
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    setattr(
        tokenizer,
        "response_template",
        {
            "start_anchor": "<assistant>",
            "fields": {
                "reasoning": {"open": "<think>", "close": "</think>"},
                "content": {},
            },
        },
    )
    prompt = "<assistant><think>"
    state = make_response_stream_state(tokenizer, True, prompt_prefix=prompt)
    pieces = [state.feed(chunk) for chunk in ("secret</thi", "nk>answer")]
    final = state.feed("", last=True)
    assert "".join(piece.reasoning or "" for piece in [*pieces, final]) == "secret"
    assert "".join(piece.content or "" for piece in [*pieces, final]) == "answer"
    assert split_thinking(
        "secret</think>answer", processor=tokenizer, prompt_prefix=prompt
    ) == ("secret", "answer")
    ordinary = make_response_stream_state(tokenizer, prompt_prefix="<assistant>")
    delta = ordinary.feed("ordinary answer", last=True)
    assert delta.reasoning is None and delta.content == "ordinary answer"


def test_streamed_thinking_keeps_unmatched_marker_prefix_as_text() -> None:
    state = make_response_stream_state(None, enable_thinking=True)
    pieces = [state.feed(chunk) for chunk in ("text <th", "ere</thi", "nk>done")]
    assert "".join(piece.reasoning or "" for piece in pieces) == "text <there"
    assert "".join(piece.content or "" for piece in pieces) == "done"


def test_audio_request_is_rejected_before_model_loading() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/chat/completions",
            json={
                "model": "not-a-checkpoint",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": "invalid", "format": "wav"},
                            }
                        ],
                    }
                ],
            },
        )
    assert response.status_code == 422
    assert "Audio inputs" in response.text
