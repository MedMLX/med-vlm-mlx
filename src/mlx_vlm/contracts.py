"""Value contracts shared by loaders, processors, and cache persistence.

JSON describes serialized configuration. State trees additionally carry native
arrays without converting their dtype or evaluating lazy MLX computations.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Protocol,
    TypedDict,
    TypeIs,
    cast,
    runtime_checkable,
)

import mlx.core as mx
import numpy as np
from PIL import Image


class _Evaluation(Protocol):
    def __call__(self, *values: object) -> None: ...


def evaluate(*values: object) -> None:
    """Evaluate array leaves; MLX ignores non-array leaves in nested containers."""
    # MLX 0.32.3 documents this API but leaves its variadic arguments untyped.
    cast(_Evaluation, getattr(mx, "eval"))(*values)


def async_evaluate(*values: object) -> None:
    """Schedule evaluation using the same tree contract as ``mlx.core.eval``."""
    cast(_Evaluation, getattr(mx, "async_eval"))(*values)


class _Einsum(Protocol):
    def __call__(self, subscripts: str, *operands: mx.array) -> mx.array: ...


def einsum(subscripts: str, *operands: mx.array) -> mx.array:
    """Contract array operands through MLX's documented variadic API."""
    return cast(_Einsum, getattr(mx, "einsum"))(subscripts, *operands)


def array_depends(
    value: mx.array, dependencies: mx.array | Sequence[mx.array]
) -> mx.array:
    """Keep an array value while adding native graph evaluation dependencies."""
    native = cast(
        Callable[[mx.array, mx.array | Sequence[mx.array]], mx.array],
        getattr(mx, "depends"),
    )
    return native(value, dependencies)


class _SafetensorsWriter(Protocol):
    def __call__(
        self,
        file: str | Path,
        arrays: dict[str, mx.array],
        metadata: dict[str, str] | None = None,
    ) -> None: ...


def save_safetensors(
    file: str | Path,
    arrays: dict[str, mx.array],
    metadata: dict[str, str] | None = None,
) -> None:
    """Write arrays and string metadata through MLX's verified file API."""
    cast(_SafetensorsWriter, getattr(mx, "save_safetensors"))(file, arrays, metadata)


if TYPE_CHECKING:
    from torch import Tensor as TorchTensor

    from mlx_vlm.generate.common import PromptCacheState
    from mlx_vlm.generate.types import StopCriteria
    from mlx_vlm.utils import ThinkingBudgetCriteria
    from mlx_vlm.vision_cache import VisionFeatureCache

type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[
    str, JSONValue
]
type JSONObject = dict[str, JSONValue]
type NumpyScalar = (
    np.bool_
    | np.int8
    | np.int16
    | np.int32
    | np.int64
    | np.uint8
    | np.uint16
    | np.uint32
    | np.uint64
    | np.float16
    | np.float32
    | np.float64
    | np.longdouble
    | np.complex64
    | np.complex128
    | np.clongdouble
    | np.str_
    | np.bytes_
)
type NumericArray = np.ndarray[
    tuple[int, ...],
    np.dtype[
        np.float16
        | np.float32
        | np.float64
        | np.int8
        | np.int16
        | np.int32
        | np.int64
        | np.uint8
        | np.uint16
        | np.uint32
        | np.uint64
        | np.bool_
    ],
]


type FloatArray = np.ndarray[
    tuple[int, ...], np.dtype[np.float16 | np.float32 | np.float64]
]
type NumpyArray = np.ndarray[tuple[int, ...], np.dtype[NumpyScalar]]


def numpy_to_mlx(value: NumpyArray, *, dtype: mx.Dtype | None = None) -> mx.array:
    """Convert a numeric NumPy tensor through MLX's native ndarray boundary.

    NumPy implements DLPack with a broader stream annotation than MLX's stub;
    the native constructor accepts these arrays directly. Dtype conversion is
    left to MLX unless ``dtype`` is supplied explicitly.
    """
    if value.dtype.kind not in "buifc":
        raise TypeError(f"Expected a numeric NumPy tensor, received {value.dtype}")
    return mx.array(cast("mx.DLPackCompatible", value), dtype=dtype)


type ImageSource = str | Path | Image.Image | NumpyArray | mx.array
type InputTensor = mx.array | NumpyArray | TorchTensor
type ProcessedInputs = dict[str, ProcessorValue]
type StateTree = (
    mx.array
    | NumpyScalar
    | str
    | int
    | float
    | bool
    | None
    | list[StateTree]
    | tuple[StateTree, ...]
    | dict[Hashable, StateTree]
)
# Cache implementations own their mutable containers. Traversal accepts a
# covariant view; detached snapshots reconstruct the concrete StateTree above.
type StateView = (
    mx.array
    | NumpyScalar
    | str
    | int
    | float
    | bool
    | None
    | Sequence[StateView]
    | Mapping[Hashable, StateView]
)
type KVPair = tuple[mx.array, mx.array]
type QuantizedTensor = tuple[mx.array, mx.array, mx.array]
type KVBuffer = mx.array | QuantizedTensor
type AttentionMask = mx.array | str | None

# Numeric constructors validate arbitrary ingress values and either return the
# stated scalar type or raise their native exception. Keep that error behavior
# at JSON/config boundaries instead of coercing through str or hiding errors.
parse_int = cast(Callable[[object], int], int)
parse_float = cast(Callable[[object], float], float)
parse_path = cast(Callable[[object], Path], Path)


def is_json_value(value: object) -> TypeIs[JSONValue]:
    """Narrow raw decoder/plugin input without coercing keys or scalar types."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return True
    if isinstance(value, list):
        return all(is_json_value(item) for item in cast(list[object], value))
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and is_json_value(item)
            for key, item in cast(dict[object, object], value).items()
        )
    return False


def is_json_object(value: object) -> TypeIs[JSONObject]:
    """Narrow a JSON object without coercing its keys or entries."""
    return isinstance(value, dict) and is_json_value(cast(dict[object, object], value))


@runtime_checkable
class CacheEntry(Protocol):
    """The state round trip required by prompt caching, including custom caches."""

    @property
    def state(self) -> StateView:
        """Return the layer state as native arrays plus JSON scalars."""
        ...

    @state.setter
    def state(self, value: StateTree, /) -> None:
        """Restore the layer state from a snapshot."""
        ...

    @property
    def meta_state(self) -> StateView:
        """Return serializable cache metadata such as offsets and positions."""
        ...

    @meta_state.setter
    def meta_state(self, value: StateTree, /) -> None:
        """Restore cache metadata from a snapshot."""
        ...


type CacheComponent = CacheEntry | tuple[CacheComponent, ...]


class AttentionCache(CacheEntry, Protocol):
    """Dense or quantized attention cache; offsets may be per batch row."""

    @property
    def offset(self) -> int | mx.array:
        """Return the number of cached tokens, per batch row when batched."""
        ...

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[KVBuffer, KVBuffer]:
        """Append one step of keys and values and return the full cache buffers."""
        ...


class LayeredModel(Protocol):
    """Transformer stack exposing its decoder layers."""

    @property
    def layers(self) -> Sequence[object]:
        """Return the decoder layers in execution order."""
        ...


@runtime_checkable
class CacheFactory(Protocol):
    """Model that constructs its own prompt cache."""

    def make_cache(self) -> list[CacheComponent]:
        """Return one empty cache component per attention layer."""
        ...


type CacheModel = LayeredModel | CacheFactory


class KVQuantConfig(TypedDict):
    """Uniform KV-cache quantization options accepted by MLX caches."""

    bits: int
    group_size: int


class RopeScaling(TypedDict, total=False):
    """Gemma3 ``rope_scaling`` keys read by ``initialize_rope``."""

    type: str
    rope_type: str
    factor: float


class QuantizationOptions(TypedDict, total=False):
    """Per-module quantization parameters plus conversion-only fallbacks."""

    group_size: int
    bits: int
    mode: str
    scheme: str
    quantize_input: bool
    fallback_group_size: int | None


type QuantizationConfig = dict[str, int | float | str | bool | QuantizationOptions]


type ProcessorValue = InputTensor | JSONValue | list[InputTensor] | list[
    list[InputTensor]
]


def is_processor_value(value: object) -> TypeIs[ProcessorValue]:
    """Validate processor leaves, including image tensor batches."""
    return is_json_value(value) or is_pixel_values(value)


def require_processed_inputs(value: object) -> ProcessedInputs:
    """Validate a third-party processor result without coercing its values."""
    if not isinstance(value, dict):
        raise TypeError("Processor output must be a dictionary")
    items = cast(dict[object, object], value)
    for key, item in items.items():
        if not isinstance(key, str) or not is_processor_value(item):
            raise TypeError(f"Unsupported processor output entry: {key!r}")
    return cast(ProcessedInputs, items)


type ModelArgument = (
    StateView
    | ProcessorValue
    | dict[str, KVPair]
    | Sequence[CacheComponent]
    | VisionFeatureCache
    | PromptCacheState
    | StopCriteria
    | ThinkingBudgetCriteria
    | Callable[[mx.array], mx.array]
    | list[Callable[[mx.array, mx.array], mx.array]]
    | ImageSource
    | Sequence[ImageSource]
    | Sequence[mx.array]
    | list[list[Callable[[mx.array, mx.array], mx.array]] | None]
)
type PromptInputs = dict[str, ModelArgument]
# Module.update() represents parameter children as maps and indexed lists.
type ParameterNode = mx.array | dict[str, ParameterNode] | list[ParameterNode]
type ParameterTree = dict[str, ParameterNode]


def is_int_list(value: object) -> TypeIs[list[int]]:
    """Recognize integer token/offset vectors without coercing their elements."""
    return isinstance(value, list) and all(
        isinstance(item, int) for item in cast(list[object], value)
    )


def array_int_list(value: mx.array) -> list[int]:
    """Read an integer vector from MLX's shape-dependent ``tolist`` boundary."""
    result: object = value.tolist()
    if not is_int_list(result):
        raise TypeError("Expected a one-dimensional integer token or offset array")
    return result


def is_input_tensor(value: object) -> TypeIs[InputTensor]:
    """Narrow native processor tensors without importing optional PyTorch."""
    import sys

    if isinstance(value, mx.array):
        return True
    if isinstance(value, np.ndarray):
        return (
            cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], value).dtype.kind
            in "buifcSU"
        )
    tensor_type = getattr(sys.modules.get("torch"), "Tensor", None)
    return isinstance(tensor_type, type) and isinstance(value, tensor_type)


def is_pixel_values(
    value: object,
) -> TypeIs[InputTensor | list[InputTensor] | list[list[InputTensor]] | None]:
    """Processors emit one tensor, a crop list, or a batch of crop lists."""
    if value is None or is_input_tensor(value):
        return True
    if not isinstance(value, list):
        return False
    if all(is_input_tensor(item) for item in cast(list[object], value)):
        return True
    return all(
        isinstance(batch, list)
        and all(is_input_tensor(item) for item in cast(list[object], batch))
        for batch in cast(list[object], value)
    )


def is_str_list(value: object) -> TypeIs[list[str]]:
    """Recognize text lists without coercing their elements."""
    return isinstance(value, list) and all(
        isinstance(item, str) for item in cast(list[object], value)
    )


def array_int_rows(value: mx.array) -> list[list[int]]:
    """Read a batched integer tensor without changing row widths or values."""
    result: object = value.tolist()
    if isinstance(result, list) and all(
        is_int_list(row) for row in cast(list[object], result)
    ):
        return [row for row in cast(list[object], result) if is_int_list(row)]
    raise TypeError("Expected a two-dimensional integer token array")


def _is_number_list(value: object) -> TypeIs[list[int | float | complex]]:
    return isinstance(value, list) and all(
        isinstance(item, (int, float, complex)) for item in cast(list[object], value)
    )


def array_number_rows(value: mx.array) -> list[list[int | float | complex]]:
    """Read numeric mask values without converting their dtype or truth values."""
    result: object = value.tolist()
    if isinstance(result, list) and all(
        _is_number_list(row) for row in cast(list[object], result)
    ):
        return [row for row in cast(list[object], result) if _is_number_list(row)]
    raise TypeError("Expected a two-dimensional numeric mask array")


def is_cache_entry_list(value: object) -> TypeIs[list[CacheEntry]]:
    """Recognize lists whose entries implement the cache state round trip."""
    return isinstance(value, list) and all(
        isinstance(item, CacheEntry) for item in cast(list[object], value)
    )
