"""Separate streamed and final reasoning from user-visible chat content."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, TypedDict, cast

from mlx_vlm.contracts import is_json_object

logger = logging.getLogger("mlx_vlm.server")

_CONTENT_MARKERS = ("<|START_TEXT|>", "<|END_TEXT|>")


class ResponseParseEvent(TypedDict, total=False):
    """Fields consumed from Transformers' response-template stream events."""

    type: str
    field: str
    text: str


class ResponseParserLike(Protocol):
    """Streaming parser interface of a tokenizer response template."""

    def feed(self, text: str) -> list[ResponseParseEvent]:
        """Consume text and return the parser events it emits."""
        ...

    def finalize(self) -> tuple[object, list[ResponseParseEvent]]:
        """Return the parsed result and events emitted at end of stream."""
        ...


class _ResponseParserFactory(Protocol):
    def get_response_parser(self, *, prefix: str) -> ResponseParserLike: ...


class _ResponseTextParser(Protocol):
    def parse_response(self, text: str, *, prefix: str) -> object: ...


def _strip_content_markers(text: str) -> str:
    for marker in _CONTENT_MARKERS:
        text = text.replace(marker, "")
    return text


@dataclass
class ThinkingStreamDelta:
    """One stream step's reasoning and content, plus the closing marker flag."""

    reasoning: Optional[str] = None
    content: Optional[str] = None
    thinking_closed: bool = False


class ThinkingStreamState:
    """Split streamed thinking delimiters from user-visible content.

    Partial opening and closing markers are buffered across chunks until the
    first reasoning segment closes. Subsequent text is treated as content;
    leading newlines immediately following that closing marker are dropped.
    """

    _DEFAULT_OPEN_CLOSE_MARKERS = (
        ("<|channel>thought", "<channel|>"),
        ("<think>", "</think>"),
        ("<|START_THINKING|>", "<|END_THINKING|>"),
    )

    def __init__(
        self,
        enable_thinking: bool = False,
        thinking_start_token: Optional[str] = None,
        thinking_end_token: Optional[str] = None,
    ):
        """Configure markers and whether output already starts inside reasoning."""
        self.open_close_markers = self.build_open_close_markers(
            thinking_start_token, thinking_end_token
        )
        self.open_markers = tuple(marker for marker, _ in self.open_close_markers)
        self.close_markers = tuple(marker for _, marker in self.open_close_markers)
        self.in_thinking = bool(enable_thinking)
        self.thinking_done = False
        # Newlines right after the close marker are dropped, even when they
        # arrive in later chunks than the marker itself.
        self.strip_leading_newlines = False
        self.buffer = ""
        self.finished = False

    def feed(self, text: str, last: bool = False) -> ThinkingStreamDelta:
        """Split ``text`` into reasoning and content deltas.

        A trailing partial marker is held until the next chunk; ``last``
        releases the remaining buffer into the region it belongs to.
        """
        self.finished = self.finished or last
        self.buffer += text or ""
        reasoning: list[str] = []
        content: list[str] = []
        thinking_closed = False

        while self.buffer:
            if self.in_thinking:
                idx, marker = self._find_first(self.buffer, self.close_markers)
                if idx < 0:
                    emit, self.buffer = self._split_partial(
                        self.buffer, self.close_markers + self.open_markers
                    )
                    emit = self._strip_open_marker(emit)
                    if emit:
                        reasoning.append(emit)
                    break

                before = self._strip_open_marker(self.buffer[:idx])
                if before:
                    reasoning.append(before)

                self.buffer = self.buffer[idx + len(marker) :]
                self.in_thinking = False
                self.thinking_done = True
                self.strip_leading_newlines = True
                thinking_closed = True
                continue

            if self.thinking_done:
                if self.strip_leading_newlines:
                    self.buffer = self.buffer.lstrip("\n")
                    if not self.buffer:
                        break
                    self.strip_leading_newlines = False
                emit, self.buffer = self._split_partial(self.buffer, _CONTENT_MARKERS)
                emit = _strip_content_markers(emit)
                if emit:
                    content.append(emit)
                break

            idx, marker = self._find_first(self.buffer, self.open_markers)
            if idx < 0:
                emit, self.buffer = self._split_partial(self.buffer, self.open_markers)
                emit = _strip_content_markers(emit)
                if emit:
                    content.append(emit)
                break

            if idx:
                emit = _strip_content_markers(self.buffer[:idx])
                if emit:
                    content.append(emit)

            self.buffer = self.buffer[idx + len(marker) :].lstrip("\n")
            self.in_thinking = True

        if last and self.buffer:
            held, self.buffer = self.buffer, ""
            if self.in_thinking:
                reasoning.append(self._strip_open_marker(held))
            else:
                content.append(_strip_content_markers(held))

        return ThinkingStreamDelta(
            reasoning="".join(reasoning) or None,
            content="".join(content) or None,
            thinking_closed=thinking_closed,
        )

    @classmethod
    def build_open_close_markers(
        cls,
        thinking_start_token: Optional[str],
        thinking_end_token: Optional[str],
    ) -> Tuple[Tuple[str, str], ...]:
        """Return the requested marker pair, then the built-in pairs, deduplicated."""
        markers: list[tuple[str, str]] = []
        if thinking_start_token and thinking_end_token:
            markers.append((thinking_start_token, thinking_end_token))
        for marker_pair in cls._DEFAULT_OPEN_CLOSE_MARKERS:
            if marker_pair not in markers:
                markers.append(marker_pair)
        return tuple(markers)

    @staticmethod
    def _find_first(text: str, markers: Tuple[str, ...]) -> Tuple[int, str]:
        found_idx = -1
        found_marker = ""
        for marker in markers:
            idx = text.find(marker)
            if idx >= 0 and (found_idx < 0 or idx < found_idx):
                found_idx = idx
                found_marker = marker
        return found_idx, found_marker

    @staticmethod
    def _split_partial(text: str, markers: Tuple[str, ...]) -> Tuple[str, str]:
        hold = 0
        for marker in markers:
            max_len = min(len(marker) - 1, len(text))
            for length in range(max_len, 0, -1):
                if text.endswith(marker[:length]):
                    hold = max(hold, length)
                    break
        if hold:
            return text[:-hold], text[-hold:]
        return text, ""

    def _strip_open_marker(self, text: str) -> str:
        for marker in self.open_markers:
            if marker in text:
                before, after = text.split(marker, 1)
                return before + after.lstrip("\n")
        return text


class ResponseTemplateStreamState:
    """Adapt a Transformers response-template parser to server stream deltas."""

    def __init__(self, parser: ResponseParserLike):
        """Wrap a parser created from the prompt's response template."""
        self.parser = parser
        self.finished = False

    def feed(self, text: str, last: bool = False) -> ThinkingStreamDelta:
        """Forward text to the template parser and map its events to deltas.

        ``region_chunk`` events for reasoning or content become text segments;
        a reasoning ``region_close`` sets ``thinking_closed``.
        """
        self.finished = self.finished or last
        reasoning: list[str] = []
        content: list[str] = []
        thinking_closed = False
        events = self.parser.feed(text or "")
        if last:
            _, final_events = self.parser.finalize()
            events.extend(final_events)

        for event in events:
            event_type = event.get("type")
            field = event.get("field")
            if event_type == "region_close" and field in (
                "reasoning",
                "reasoning_content",
            ):
                thinking_closed = True
            if event_type != "region_chunk":
                continue
            chunk = event.get("text")
            if not chunk:
                continue
            if field in ("reasoning", "reasoning_content"):
                reasoning.append(chunk)
            elif field == "content":
                content.append(chunk)
        return ThinkingStreamDelta(
            reasoning="".join(reasoning) or None,
            content="".join(content) or None,
            thinking_closed=thinking_closed,
        )


def _response_template_tokenizer(processor: object) -> object | None:
    if processor is None:
        return None
    tokenizer: object = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "response_template", None) is None:
        return None
    return tokenizer


def make_response_stream_state(
    processor: object,
    enable_thinking: bool = False,
    thinking_start_token: Optional[str] = None,
    thinking_end_token: Optional[str] = None,
    *,
    prompt_prefix: str = "",
):
    """Create a tokenizer parser using the rendered prompt, or a marker fallback.

    Tokenizer templates can open a reasoning region in the prompt itself, so
    callers should pass the actual rendered prompt as ``prompt_prefix``.
    """
    tokenizer = _response_template_tokenizer(processor)
    if tokenizer is not None and hasattr(tokenizer, "get_response_parser"):
        try:
            # Verified against Transformers' get_response_parser(prefix=...) API.
            return ResponseTemplateStreamState(
                cast(_ResponseParserFactory, tokenizer).get_response_parser(
                    prefix=prompt_prefix
                )
            )
        except (AttributeError, TypeError, ValueError):
            logger.debug(
                "Falling back from tokenizer response-template parser", exc_info=True
            )
    return ThinkingStreamState(
        enable_thinking,
        thinking_start_token,
        thinking_end_token,
    )


def prompt_has_open_thinking(
    prompt: object,
    enable_thinking: bool = False,
    thinking_start_token: Optional[str] = None,
    thinking_end_token: Optional[str] = None,
) -> bool:
    """Return whether generation starts inside a prompt-opened thinking block."""
    if not isinstance(prompt, str):
        return False

    stripped_prompt = prompt.rstrip()
    for start_marker, _ in ThinkingStreamState.build_open_close_markers(
        thinking_start_token, thinking_end_token
    ):
        if stripped_prompt.endswith(start_marker):
            return True
    return False


def finish_content_streams(
    thinking_state: ThinkingStreamState | ResponseTemplateStreamState,
):
    """``(reasoning, content)`` still held when a stream ends.

    Callers finalize the state on a token with a finish reason; a token
    iterator can also just stop. A finished state is not fed again (a
    tokenizer response parser refuses a second finalization).
    """
    if thinking_state.finished:
        return None, None
    delta = thinking_state.feed("", last=True)
    return delta.reasoning, delta.content


def _clean_reasoning(reasoning: str, start_marker: str) -> str:
    reasoning = reasoning.replace(start_marker, "")
    if start_marker == "<|channel>thought":
        reasoning = reasoning.removeprefix("thought")
    return reasoning.strip()


def split_thinking(
    text: str,
    thinking_start_token: Optional[str] = None,
    thinking_end_token: Optional[str] = None,
    starts_in_thinking: bool = False,
    processor: object = None,
    *,
    prompt_prefix: str = "",
) -> Tuple[Optional[str], str]:
    """Separate reasoning from content using the tokenizer template when available.

    ``prompt_prefix`` supplies the rendered prompt's parser context; the marker
    fallback uses ``starts_in_thinking`` when the prompt leaves reasoning open.
    """
    if not text:
        return None, text

    tokenizer = _response_template_tokenizer(processor)
    if tokenizer is not None and hasattr(tokenizer, "parse_response"):
        try:
            parsed = cast(_ResponseTextParser, tokenizer).parse_response(
                text, prefix=prompt_prefix
            )
            if is_json_object(parsed) and (
                "content" in parsed
                or "reasoning" in parsed
                or "reasoning_content" in parsed
            ):
                reasoning = parsed.get("reasoning_content") or parsed.get("reasoning")
                content = parsed.get("content")
                if reasoning is None or isinstance(reasoning, str):
                    if content is None or isinstance(content, str):
                        return (
                            reasoning.strip() if reasoning else None,
                            content.strip() if content else "",
                        )
        except (AttributeError, TypeError, ValueError):
            logger.debug(
                "Falling back from tokenizer response-template parser", exc_info=True
            )

    for start_marker, end_marker in ThinkingStreamState.build_open_close_markers(
        thinking_start_token, thinking_end_token
    ):
        start = text.find(start_marker)
        end = text.find(end_marker, start if start >= 0 else 0)
        if start >= 0 and start < end:
            reasoning = text[start + len(start_marker) : end].strip()
            content = _strip_content_markers(
                text[:start] + text[end + len(end_marker) :]
            ).strip()
            return reasoning or None, content

        if end_marker in text:
            reasoning, content = text.split(end_marker, 1)
            reasoning = _clean_reasoning(reasoning, start_marker)
            return reasoning or None, _strip_content_markers(content).strip()

        if start_marker in text:
            reasoning = _clean_reasoning(text, start_marker)
            return reasoning or None, ""

    if starts_in_thinking:
        reasoning = _strip_content_markers(text).strip()
        return reasoning or None, ""

    return None, _strip_content_markers(text).strip()
