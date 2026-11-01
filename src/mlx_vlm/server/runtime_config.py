"""Server settings that can be changed at runtime, without a restart."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    TypedDict,
    cast,
)

from mlx_vlm.contracts import JSONObject, JSONValue

type SettingValue = str | int | float | bool | None
type KnobType = Literal[
    "bool", "int", "int_or_none", "float_or_none", "str", "str_or_none"
]


class KnobSpec(TypedDict):
    """Internal specification of one runtime setting."""

    type: KnobType
    default: SettingValue
    reload_kinds: tuple[str, ...]
    allowed: list[str] | None
    help: str


class KnobSchema(TypedDict):
    """JSON-serializable view of one runtime setting."""

    name: str
    type: KnobType
    default: SettingValue
    reload_kinds: list[str]
    allowed: list[str] | None
    help: str


class KnobRejection(TypedDict):
    """Reason a requested setting change was not applied."""

    name: str
    reason: str


# The built-in constructors ARE the numeric validation boundary. They accept
# raw decoded JSON here so invalid containers/null retain Python's original
# TypeError text; on success their results are always int/float, respectively.
_json_int = cast(Callable[[JSONValue], int], int)
_json_float = cast(Callable[[JSONValue], float], float)

TEXT_KINDS: Tuple[str, ...] = ("text_generation",)

DEFAULT_TOKEN_QUEUE_TIMEOUT = 600.0

logger = logging.getLogger("mlx_vlm.server")

# name, type, default, reload kinds, allowed (None = free), help
KNOBS: Tuple[
    Tuple[str, KnobType, SettingValue, Tuple[str, ...], Optional[Tuple[str, ...]], str],
    ...,
] = (
    (
        "kv_bits",
        "float_or_none",
        None,
        TEXT_KINDS,
        None,
        "KV bits; null = model default.",
    ),
    (
        "kv_group_size",
        "int_or_none",
        None,
        TEXT_KINDS,
        None,
        "KV quantization group size.",
    ),
    (
        "quantized_kv_start",
        "int_or_none",
        None,
        TEXT_KINDS,
        None,
        "Quantized KV start layer.",
    ),
    (
        "max_kv_size",
        "int_or_none",
        None,
        TEXT_KINDS,
        None,
        "Requested context budget (tokens).",
    ),
    (
        "token_queue_timeout",
        "float_or_none",
        DEFAULT_TOKEN_QUEUE_TIMEOUT,
        TEXT_KINDS,
        None,
        "Token queue wait timeout in seconds; null disables the timeout.",
    ),
    (
        "vision_cache_size",
        "int",
        20,
        TEXT_KINDS,
        None,
        "Vision feature cache capacity.",
    ),
)

_KNOB_SPEC: Dict[str, KnobSpec] = {
    name: {
        "type": kind,
        "default": default,
        "reload_kinds": kinds,
        "allowed": list(allowed) if allowed is not None else None,
        "help": help_text,
    }
    for (name, kind, default, kinds, allowed, help_text) in KNOBS
}

# Knobs that are naturally live (applied per request) and must never trigger a
# model reload, so they are excluded from the cache-key fingerprint.
_LIVE_KNOBS: Tuple[str, ...] = ("max_kv_size", "token_queue_timeout")


def _env_float(
    name: str, default: Optional[float], *, zero_is_none: bool = True
) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    value = _json_float(raw)
    return default if zero_is_none and value == 0 else value


def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.environ.get(name)
    return _json_int(raw) if raw else default


def _env_token_queue_timeout() -> Optional[float]:
    raw = os.environ.get("MLX_VLM_TOKEN_QUEUE_TIMEOUT", "")
    if raw == "":
        return DEFAULT_TOKEN_QUEUE_TIMEOUT
    try:
        timeout = float(raw)
    except ValueError:
        logger.warning(
            "Invalid MLX_VLM_TOKEN_QUEUE_TIMEOUT=%r; falling back to %ss.",
            raw,
            DEFAULT_TOKEN_QUEUE_TIMEOUT,
        )
        return DEFAULT_TOKEN_QUEUE_TIMEOUT
    return timeout if timeout > 0 else None


@dataclass
class RuntimeConfig:
    """Runtime-tunable generation and cache settings guarded by a lock.

    Values read from the environment at startup are retained as per-setting
    defaults so ``apply_changes(op="replace")`` can restore them.
    """

    kv_bits: Optional[float] = None
    kv_group_size: Optional[int] = None
    quantized_kv_start: Optional[int] = None
    max_kv_size: Optional[int] = None
    token_queue_timeout: Optional[float] = DEFAULT_TOKEN_QUEUE_TIMEOUT
    vision_cache_size: int = 20

    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )
    _env_defaults: Dict[str, SettingValue] = field(
        default_factory=dict[str, SettingValue], repr=False, compare=False
    )

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        """Read settings from their environment variables.

        The resolved values also become the defaults restored by a later
        ``replace`` update.
        """
        cfg = cls(
            kv_bits=_env_float("KV_BITS", None),
            kv_group_size=_env_int("KV_GROUP_SIZE", None),
            quantized_kv_start=_env_int("QUANTIZED_KV_START", None),
            max_kv_size=_env_int("MAX_KV_SIZE", None),
            token_queue_timeout=_env_token_queue_timeout(),
            vision_cache_size=int(os.environ.get("MLX_VLM_VISION_CACHE_SIZE", "20")),
        )
        cfg._env_defaults = {name: getattr(cfg, name) for name in _KNOB_SPEC}
        return cfg

    def schema(self) -> List[KnobSchema]:
        """Return each setting's type, default, allowed values and reload kinds."""
        return [
            {
                "name": name,
                "type": spec["type"],
                "default": spec["default"],
                "reload_kinds": list(spec["reload_kinds"]),
                "allowed": spec["allowed"],
                "help": spec["help"],
            }
            for name, spec in _KNOB_SPEC.items()
        ]

    def current(self) -> Dict[str, SettingValue]:
        """Return a snapshot of the current setting values."""
        with self._lock:
            return {name: getattr(self, name) for name in _KNOB_SPEC}

    def apply_changes(
        self, payload: JSONObject, op: str = "merge"
    ) -> Tuple[Dict[str, SettingValue], List[KnobRejection]]:
        """Apply requested settings and report accepted and rejected entries.

        ``op="replace"`` first restores the startup environment values. Unknown
        names and values failing type or range validation are rejected without
        changing the others.
        """
        if op == "replace":
            with self._lock:
                for name in _KNOB_SPEC:
                    restored = self._env_defaults.get(name, _KNOB_SPEC[name]["default"])
                    setattr(self, name, restored)
        return self._apply_validated(payload)

    def _apply_validated(
        self, payload: JSONObject
    ) -> Tuple[Dict[str, SettingValue], List[KnobRejection]]:
        applied: Dict[str, SettingValue] = {}
        rejected: List[KnobRejection] = []
        with self._lock:
            for name, raw in payload.items():
                if name not in _KNOB_SPEC:
                    rejected.append({"name": name, "reason": "unknown knob"})
                    continue
                spec = _KNOB_SPEC[name]
                try:
                    value = _coerce(spec["type"], raw, spec.get("allowed"))
                    if (
                        name == "token_queue_timeout"
                        and isinstance(value, (int, float))
                        and value < 0
                    ):
                        value = None
                except (TypeError, ValueError, OverflowError) as exc:
                    rejected.append({"name": name, "reason": str(exc)})
                    continue
                setattr(self, name, value)
                applied[name] = value
        return applied, rejected

    def _in_cache_key(self, name: str) -> bool:
        # Callers must hold self._lock. A knob outside the cache key cannot
        # change the fingerprint, so changing it never reloads a model.
        if name in _LIVE_KNOBS:
            return False
        return True

    def reload_kinds(self, applied: Dict[str, SettingValue]) -> Set[str]:
        """Return the model kinds that must reload after ``applied`` changes."""
        kinds: Set[str] = set()
        with self._lock:
            for name in applied:
                if not self._in_cache_key(name):
                    continue
                kinds.update(_KNOB_SPEC[name]["reload_kinds"])
        return kinds

    def fingerprint(self, kinds: Optional[Iterable[str]] = None) -> str:
        """Hash the cache-affecting settings for ``kinds``, or for every kind.

        Settings applied live per request are excluded, so changing them never
        produces a new cache key.
        """
        kind_set = set(kinds) if kinds is not None else None
        items: List[Tuple[str, SettingValue]] = []
        with self._lock:
            for name in _KNOB_SPEC:
                if not self._in_cache_key(name):
                    continue
                spec = _KNOB_SPEC[name]
                if kind_set is not None and not (set(spec["reload_kinds"]) & kind_set):
                    continue
                value = getattr(self, name)
                items.append((name, value))
        blob = json.dumps(sorted(items), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


def _coerce(
    kind: str, raw: JSONValue, allowed: Optional[List[str]] = None
) -> SettingValue:
    if kind == "bool":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return bool(raw)
        if isinstance(raw, str):
            lowered = raw.strip().lower()
            if lowered in ("1", "true", "yes", "on"):
                return True
            if lowered in ("0", "false", "no", "off", ""):
                return False
        raise ValueError(f"expected a boolean, got {raw!r}")
    if kind == "int":
        return _json_int(raw)
    if kind == "int_or_none":
        if raw is None or raw == "":
            return None
        return _json_int(raw)
    if kind == "float_or_none":
        if raw is None or raw == "":
            return None
        value = _json_float(raw)
        return None if value == 0 else value
    if kind == "str":
        if raw is None:
            raise ValueError("expected a string, got null")
        value = str(raw)
    elif kind == "str_or_none":
        if raw is None or raw == "":
            return None
        value = str(raw)
    else:
        raise ValueError(f"unsupported knob type {kind!r}")
    if allowed is not None and value not in allowed:
        raise ValueError(f"expected one of {allowed}, got {value!r}")
    return value
