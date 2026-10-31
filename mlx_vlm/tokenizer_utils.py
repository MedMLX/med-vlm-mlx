"""Incremental detokenizers that stream decoded text from generated tokens."""

from __future__ import annotations

import codecs
import json
from collections.abc import Callable, Container, Mapping
from copy import copy
from functools import partial
from json import JSONDecodeError
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol, Self, cast

from transformers import PreTrainedTokenizerBase

from mlx_vlm.contracts import JSONObject, JSONValue, is_json_object

if TYPE_CHECKING:
    from mlx_vlm.generate.types import ProcessorLike, TokenizerLike

REPLACEMENT_CHAR = "\ufffd"


class TextTokenizer(Protocol):
    """Token vocabulary and flat-ID decode interface used by streaming."""

    @property
    def vocab(self) -> dict[str, int]:
        """Return the token-to-ID vocabulary."""
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Decode a flat list of token IDs into text."""
        ...


def require_text_tokenizer(value: object) -> TextTokenizer:
    """Validate the flat-token decode and vocabulary API used by streaming."""
    vocabulary: object = getattr(value, "vocab", None)
    if not isinstance(vocabulary, Mapping) or not callable(
        getattr(value, "decode", None)
    ):
        raise TypeError("Streaming requires a tokenizer with vocabulary and decode")
    entries = cast(Mapping[object, object], vocabulary)
    if not all(
        isinstance(key, str) and isinstance(token, int)
        for key, token in entries.items()
    ):
        raise TypeError("Tokenizer vocabulary must map strings to integer IDs")
    # For a flat list of IDs, the installed tokenizer decode API returns one string.
    return cast(TextTokenizer, value)


def _token_map(tokenizer: TextTokenizer) -> list[str]:
    mapping: list[str | None] = [None] * len(tokenizer.vocab)
    for token, index in tokenizer.vocab.items():
        if index < 0 or index >= len(mapping):
            raise ValueError("Streaming vocabulary IDs must be contiguous from zero")
        mapping[index] = token
    if any(token is None for token in mapping):
        raise ValueError("Streaming vocabulary has missing token IDs")
    return [token for token in mapping if token is not None]


def _remove_space(x: str) -> str:
    if x and x[0] == " ":
        return x[1:]
    return x


class StreamingDetokenizer:
    """The streaming detokenizer interface so that we can detokenize one token at a time.

    Example usage is as follows:

        detokenizer = ...

        # Reset the tokenizer state
        detokenizer.reset()

        for token in generate(...):
            detokenizer.add_token(token.item())

            # Contains the whole text so far. Some tokens may not be included
            # since it contains whole words usually.
            detokenizer.text

            # Contains the printable segment (usually a word) since the last
            # time it was accessed
            detokenizer.last_segment

            # Contains all the tokens added so far
            detokenizer.tokens

        # Make sure that we detokenize any remaining tokens
        detokenizer.finalize()

        # Now detokenizer.text should match tokenizer.decode(detokenizer.tokens)
    """

    offset: int
    __slots__ = ("offset",)

    @property
    def text(self) -> str:
        """Return all text decoded so far."""
        raise NotImplementedError()

    @property
    def tokens(self) -> list[int]:
        """Return the token IDs added so far."""
        raise NotImplementedError()

    def reset(self) -> None:
        """Clear all buffered tokens and decoded text."""
        raise NotImplementedError()

    def add_token(
        self, token: int, skip_special_token_ids: Container[int] = ()
    ) -> None:
        """Append one token, ignoring IDs listed in ``skip_special_token_ids``."""
        raise NotImplementedError()

    def finalize(self) -> None:
        """Flush buffered tokens into the final text."""
        raise NotImplementedError()

    @property
    def last_segment(self) -> str:
        """Return the last segment of readable text since last time this property was accessed."""
        text = self.text
        if text and text[-1] != REPLACEMENT_CHAR:
            segment = text[self.offset :]
            self.offset = len(text)
            return segment
        return ""


class NaiveStreamingDetokenizer(StreamingDetokenizer):
    """NaiveStreamingDetokenizer relies on the underlying tokenizer
    implementation and should work with every tokenizer.

    Its complexity is O(T^2) where T is the longest line since it will
    repeatedly detokenize the same tokens until a new line is generated.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase | TextTokenizer) -> None:
        """Wrap a tokenizer that decodes every buffered token list directly."""
        self._tokenizer = require_text_tokenizer(tokenizer)
        self._tokenizer.decode([0])
        self.reset()

    def __copy__(self) -> Self:
        """Return an independent detokenizer over the same tokenizer."""
        return type(self)(self._tokenizer)

    def reset(self) -> None:
        """Clear buffered and committed tokens."""
        self.offset = 0
        self._tokens: list[int] = []
        self._text = ""
        self._current_tokens: list[int] = []
        self._current_text = ""

    def add_token(
        self, token: int, skip_special_token_ids: Container[int] = ()
    ) -> None:
        """Buffer one token until a line break is decoded."""
        if token in skip_special_token_ids:
            return
        self._current_tokens.append(token)

    def finalize(self) -> None:
        """Commit buffered tokens and append their decoded text."""
        self._tokens.extend(self._current_tokens)
        self._text += self._tokenizer.decode(self._current_tokens)
        self._current_tokens: list[int] = []
        self._current_text = ""

    @property
    def text(self) -> str:
        """Return committed text plus the current line's decoding."""
        if self._current_tokens:
            self._current_text = self._tokenizer.decode(self._current_tokens)
        if self._current_text and self._current_text[-1] == "\n":
            self._tokens.extend(self._current_tokens)
            self._text += self._current_text
            self._current_tokens.clear()
            self._current_text = ""
        return self._text + self._current_text

    @property
    def tokens(self) -> list[int]:
        """Return token IDs committed to the decoded text."""
        return self._tokens


class SPMStreamingDetokenizer(StreamingDetokenizer):
    """A streaming detokenizer for SPM models.

    It adds tokens to the text if the next token starts with the special SPM
    underscore which results in linear complexity.

    Handles UTF-8 byte tokens (like <0xE5><0xA4><0xA2>) by accumulating them
    and decoding as UTF-8 bytes, which is necessary for multi-byte characters
    like Chinese that may not be in the vocabulary.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase | TextTokenizer,
        trim_space: bool = True,
    ) -> None:
        """Build the SPM token map and mark byte-fallback vocabulary entries."""
        tokenizer = require_text_tokenizer(tokenizer)
        self.trim_space = trim_space

        # Extract the tokens in a list from id to text
        self.tokenmap = _token_map(tokenizer)
        self.is_byte_token = [False] * len(tokenizer.vocab)
        self.byte_value = [0] * len(tokenizer.vocab)

        for value, tokenid in tokenizer.vocab.items():
            self.tokenmap[tokenid] = value
            # Mark byte tokens and store their byte value
            if value.startswith("<0x") and len(value) >= 6 and value[5] == ">":
                self.is_byte_token[tokenid] = True
                self.byte_value[tokenid] = int(value[3:5], 16)

        self.reset()

    @property
    def text(self) -> str:
        """Return text committed from complete word pieces."""
        return self._text

    @property
    def tokens(self) -> list[int]:
        """Return token IDs added since the last reset."""
        return self._tokens

    def reset(self) -> None:
        """Clear unfinished text, pending bytes and token history."""
        self.offset = 0
        self._unflushed = ""
        self._byte_buffer = bytearray()
        self._text = ""
        self._tokens: list[int] = []

    def _flush_bytes(self) -> None:
        """Decode accumulated bytes as UTF-8 and append to unflushed text."""
        if self._byte_buffer:
            try:
                decoded = self._byte_buffer.decode("utf-8")
                self._unflushed += decoded
            except UnicodeDecodeError:
                # If decoding fails, use replacement character
                self._unflushed += self._byte_buffer.decode("utf-8", errors="replace")
            self._byte_buffer = bytearray()

    def add_token(
        self, token: int, skip_special_token_ids: Container[int] = ()
    ) -> None:
        """Flush pending pieces before a new word and buffer byte tokens.

        A token beginning with the SPM word marker commits the unflushed text, so
        text is emitted one word at a time; byte-fallback tokens accumulate until
        they form valid UTF-8.
        """
        if token in skip_special_token_ids:
            return

        if self.is_byte_token[token]:
            # Accumulate byte tokens
            self._byte_buffer.append(self.byte_value[token])
            return

        # Flush any accumulated bytes before processing regular token
        self._flush_bytes()

        v = self.tokenmap[token]
        if v and v[0] == "\u2581":
            if self._text or not self.trim_space:
                self._text += self._unflushed.replace("\u2581", " ")
            else:
                self._text = _remove_space(self._unflushed.replace("\u2581", " "))
            self._unflushed = v
        else:
            self._unflushed += v

    def finalize(self) -> None:
        """Flush accumulated bytes and the final word piece into the text."""
        # Flush any remaining bytes
        self._flush_bytes()

        if self._text or not self.trim_space:
            self._text += self._unflushed.replace("\u2581", " ")
        else:
            self._text = _remove_space(self._unflushed.replace("\u2581", " "))
        self._unflushed = ""


class BPEStreamingDetokenizer(StreamingDetokenizer):
    """A streaming detokenizer for OpenAI style BPE models.

    It adds tokens to the text if the next token starts with a space similar to
    the SPM detokenizer.
    """

    byte_decoder: dict[str, int] = {}

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase | TextTokenizer,
        trim_space: bool = False,
    ) -> None:
        """Build the BPE token map and GPT-2 byte decoder."""
        tokenizer = require_text_tokenizer(tokenizer)
        self.trim_space = trim_space

        # Extract the tokens in a list from id to text
        self.tokenmap = _token_map(tokenizer)
        for value, tokenid in tokenizer.vocab.items():
            self.tokenmap[tokenid] = value

        self.reset()

        # Make the BPE byte decoder from
        # https://github.com/openai/gpt-2/blob/master/src/encoder.py
        self.make_byte_decoder()

    @property
    def text(self) -> str:
        """Return text committed from complete word pieces."""
        return self._text

    @property
    def tokens(self) -> list[int]:
        """Return token IDs added since the last reset."""
        return self._tokens

    def reset(self) -> None:
        """Clear unfinished text and token history."""
        self.offset = 0
        self._unflushed = ""
        self._text = ""
        self._tokens: list[int] = []

    def add_token(
        self, token: int, skip_special_token_ids: Container[int] = ()
    ) -> None:
        """Commit pending bytes when a token starts with a space byte."""
        if token in skip_special_token_ids:
            return
        v = self.tokenmap[token]
        # if the token starts with space
        if self.byte_decoder[v[0]] == 32:
            current_text = bytearray(
                self.byte_decoder[c] for c in self._unflushed
            ).decode("utf-8", errors="replace")
            if self._text or not self.trim_space:
                self._text += current_text
            else:
                self._text += _remove_space(current_text)
            self._unflushed = v
        else:
            self._unflushed += v

    def finalize(self) -> None:
        """Decode the remaining byte tokens into the final text."""
        try:
            current_text = bytearray(
                self.byte_decoder[c] for c in self._unflushed
            ).decode("utf-8", errors="ignore")
            if self._text or not self.trim_space:
                self._text += current_text
            else:
                self._text += _remove_space(current_text)
        except (UnicodeDecodeError, KeyError):
            print(f"Warning: could not decode bytes: {self._unflushed}")
        self._unflushed = ""

    @classmethod
    def make_byte_decoder(cls) -> None:
        """See https://github.com/openai/gpt-2/blob/master/src/encoder.py for the rationale."""
        if cls.byte_decoder:
            return

        char_to_bytes: dict[str, int] = {}
        limits = [
            0,
            ord("!"),
            ord("~") + 1,
            ord("¡"),
            ord("¬") + 1,
            ord("®"),
            ord("ÿ") + 1,
        ]
        n = 0
        for i, (start, stop) in enumerate(zip(limits, limits[1:])):
            if i % 2 == 0:
                for b in range(start, stop):
                    char_to_bytes[chr(2**8 + n)] = b
                    n += 1
            else:
                for b in range(start, stop):
                    char_to_bytes[chr(b)] = b
        cls.byte_decoder = char_to_bytes


class ServerTokenStreamer:
    """Emit server text deltas per token while buffering incomplete UTF-8."""

    def __init__(self, tokenizer: object, detokenizer: StreamingDetokenizer) -> None:
        """Select incremental decoding for the wrapped detokenizer type."""
        self._tokenizer = require_text_tokenizer(tokenizer)
        self._detokenizer = detokenizer
        self._decoder: codecs.IncrementalDecoder | None = None
        self._started = False
        self._tokenmap: list[str] | None = None
        self._mode = "delegate"
        self._is_byte_token: list[bool] = []
        self._byte_value: list[int] = []
        self.byte_decoder: dict[str, int] = {}
        self._trim_space = False

        if isinstance(detokenizer, SPMStreamingDetokenizer):
            self._mode = "spm"
            self._tokenmap = detokenizer.tokenmap
            self._trim_space = detokenizer.trim_space
            self._is_byte_token = detokenizer.is_byte_token
            self._byte_value = detokenizer.byte_value
            self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        elif isinstance(detokenizer, BPEStreamingDetokenizer):
            self._mode = "bpe"
            self._tokenmap = detokenizer.tokenmap
            self._trim_space = detokenizer.trim_space
            self.byte_decoder = detokenizer.byte_decoder
            self._decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def _emit_text(self, text: str) -> str:
        if text and not self._started and self._trim_space and text[0] == " ":
            text = text[1:]
        if text:
            self._started = True
        return text

    def _fallback_decode(self, token: int) -> str:
        return self._tokenizer.decode([token])

    def _spm_text_for_token(self, token: int) -> str:
        assert self._decoder is not None and self._tokenmap is not None
        try:
            if self._is_byte_token[token]:
                byte_value = self._byte_value[token]
                return self._decoder.decode(bytes((byte_value,)), final=False)
            value = self._tokenmap[token]
        except (IndexError, TypeError):
            return self._fallback_decode(token)

        return value.replace("\u2581", " ")

    def _bpe_text_for_token(self, token: int) -> str:
        assert self._decoder is not None and self._tokenmap is not None
        try:
            value = self._tokenmap[token]
        except (IndexError, TypeError):
            return self._fallback_decode(token)

        try:
            token_bytes = bytes(self.byte_decoder[c] for c in value)
        except KeyError:
            return self._fallback_decode(token)
        return self._decoder.decode(token_bytes, final=False)

    def _decode_token(self, token: int) -> str:
        if self._mode == "spm":
            return self._emit_text(self._spm_text_for_token(token))
        if self._mode == "bpe":
            return self._emit_text(self._bpe_text_for_token(token))
        return self._fallback_advance(token, finish_reason=None)

    def _fallback_advance(self, token: int, finish_reason: Optional[str]) -> str:
        if finish_reason == "stop":
            self._detokenizer.finalize()
        else:
            self._detokenizer.add_token(token)
            if finish_reason is not None:
                self._detokenizer.finalize()
        return self._detokenizer.last_segment

    def advance(self, token: int, finish_reason: Optional[str]) -> str:
        """Return the text delta for one token and finish reason.

        A ``"stop"`` finish reason suppresses the token itself; any non-``None``
        reason flushes the incremental UTF-8 decoder afterwards.
        """
        if self._mode == "delegate":
            return self._fallback_advance(token, finish_reason)

        assert self._decoder is not None
        parts: list[str] = []
        if finish_reason != "stop":
            parts.append(self._decode_token(token))
        if finish_reason is not None:
            parts.append(self._emit_text(self._decoder.decode(b"", final=True)))
        return "".join(parts)

    def finalize(self) -> str:
        """Flush pending decoder state and return the remaining text."""
        if self._mode == "delegate":
            self._detokenizer.finalize()
            return self._detokenizer.last_segment
        assert self._decoder is not None
        return self._emit_text(self._decoder.decode(b"", final=True))


def make_streaming_detokenizer(
    processor: ProcessorLike | TokenizerLike | PreTrainedTokenizerBase,
) -> StreamingDetokenizer:
    """Return an isolated, reset streaming detokenizer for a processor."""
    # load_processor installs this hook on HF processors and bare tokenizers;
    # their stubs do not declare the attached streaming detokenizer.
    detokenizer = copy(cast(StreamingDetokenizer, getattr(processor, "detokenizer")))
    detokenizer.reset()
    return detokenizer


def _match(a: JSONValue, b: JSONValue) -> bool:
    if type(a) != type(b):
        return False
    if isinstance(a, dict) and isinstance(b, dict):
        return len(a) == len(b) and all(k in b and _match(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_match(ai, bi) for ai, bi in zip(a, b))

    return a == b


def _is_spm_decoder(decoder: JSONValue) -> bool:
    _target_description: JSONObject = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0},
        ],
    }
    return _match(_target_description, decoder)


def _is_spm_decoder_no_space(decoder: JSONValue) -> bool:
    _target_description: JSONObject = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
        ],
    }
    return _match(_target_description, decoder)


def _is_bpe_decoder(decoder: JSONValue) -> bool:
    return isinstance(decoder, dict) and decoder.get("type", None) == "ByteLevel"


def load_detokenizer_factory(
    model_path: Path,
) -> Callable[[PreTrainedTokenizerBase | TextTokenizer], StreamingDetokenizer]:
    """Select the streaming codec declared by a local tokenizer JSON file."""
    detokenizer_class: Callable[
        [PreTrainedTokenizerBase | TextTokenizer], StreamingDetokenizer
    ] = NaiveStreamingDetokenizer

    tokenizer_file = model_path / "tokenizer.json"
    if tokenizer_file.exists():
        with open(tokenizer_file, "r") as f:
            try:
                tokenizer_content: object = json.load(f)
            except JSONDecodeError as e:
                raise JSONDecodeError("Failed to parse tokenizer.json", e.doc, e.pos)

        if not is_json_object(tokenizer_content):
            raise ValueError("tokenizer.json must contain a JSON object")
        if "decoder" in tokenizer_content:
            if _is_spm_decoder(tokenizer_content["decoder"]):
                detokenizer_class = SPMStreamingDetokenizer
            elif _is_spm_decoder_no_space(tokenizer_content["decoder"]):
                detokenizer_class = partial(SPMStreamingDetokenizer, trim_space=False)
            elif _is_bpe_decoder(tokenizer_content["decoder"]):
                detokenizer_class = BPEStreamingDetokenizer

    return detokenizer_class
