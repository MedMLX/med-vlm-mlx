"""Used MLX 0.32.3 interfaces, checked against the installed layer sources."""

from collections.abc import Callable, ItemsView, Iterator, Sequence
from typing import Literal, Self, TypedDict, TypeIs

from mlx.core import array

type ParameterNode = array | dict[str, ParameterNode] | list[ParameterNode]
type Parameters = dict[str, ParameterNode]
type ModuleNode = Module | dict[str, ModuleNode] | list[ModuleNode]

class Module:
    def __init__(self) -> None: ...
    def __contains__(self, key: object) -> bool: ...
    def __getitem__(self, key: str) -> object: ...
    def __setitem__(self, key: str, value: object) -> None: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def items(self) -> ItemsView[str, object]: ...
    @staticmethod
    def is_module(value: object) -> TypeIs[Module]: ...
    @property
    def state(self) -> Self: ...
    @property
    def training(self) -> bool: ...
    def parameters(self) -> Parameters: ...
    def trainable_parameters(self) -> Parameters: ...
    def update(self, parameters: Parameters, strict: bool = True) -> Self: ...
    def load_weights(
        self, file_or_weights: str | list[tuple[str, array]], strict: bool = True
    ) -> Self: ...
    def save_weights(self, file: str) -> None: ...
    def eval(self) -> Self: ...
    def train(self, mode: bool = True) -> Self: ...
    def freeze(
        self,
        *,
        recurse: bool = True,
        keys: str | list[str] | None = None,
        strict: bool = False,
    ) -> Self: ...
    def named_modules(self) -> list[tuple[str, Module]]: ...
    def children(self) -> dict[str, ModuleNode]: ...
    def leaf_modules(self) -> dict[str, ModuleNode]: ...
    def update_modules(
        self, modules: dict[str, ModuleNode], strict: bool = True
    ) -> Self: ...
    def apply(
        self,
        map_fn: Callable[[array], array],
        filter_fn: Callable[[Module, str, object], bool] | None = None,
    ) -> Self: ...

class Identity(Module):
    def __init__(self, *args: object, **kwargs: object) -> None: ...
    def __call__(self, x: array) -> array: ...

class GELU(Module):
    def __init__(self, approx: str = "none") -> None: ...
    def __call__(self, x: array) -> array: ...

class LayerNorm(Module):
    weight: array
    bias: array
    def __init__(
        self, dims: int, eps: float = 1e-5, affine: bool = True, bias: bool = True
    ) -> None: ...
    def __call__(self, x: array) -> array: ...

class RoPE(Module):
    def __init__(
        self,
        dims: int,
        traditional: bool = False,
        base: float = 10000,
        scale: float = 1.0,
    ) -> None: ...
    def __call__(self, x: array, offset: int | array = 0) -> array: ...

class Conv2d(Module):
    weight: array
    bias: array
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None: ...
    def __call__(self, x: array) -> array: ...

class Conv3d(Module):
    weight: array
    bias: array
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        bias: bool = True,
    ) -> None: ...
    def __call__(self, x: array) -> array: ...

class Linear(Module):
    weight: array
    bias: array
    def __init__(
        self, input_dims: int, output_dims: int, bias: bool = True
    ) -> None: ...
    def __call__(self, x: array) -> array: ...
    def to_quantized(
        self,
        group_size: int | None = None,
        bits: int | None = None,
        mode: str = "affine",
        quantize_input: bool = False,
    ) -> QuantizedLinear | QQLinear: ...

class Embedding(Module):
    weight: array
    def __init__(self, num_embeddings: int, dims: int) -> None: ...
    def __call__(self, x: array) -> array: ...
    def as_linear(self, x: array) -> array: ...
    def to_quantized(
        self,
        group_size: int | None = None,
        bits: int | None = None,
        mode: str = "affine",
        quantize_input: bool = False,
    ) -> QuantizedEmbedding: ...

class QuantizedLinear(Module):
    weight: array
    scales: array
    biases: array | None
    bias: array
    group_size: int
    bits: int
    mode: str
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int | None = None,
        bits: int | None = None,
        mode: str = "affine",
    ) -> None: ...
    def __call__(self, x: array) -> array: ...
    @classmethod
    def from_linear(
        cls,
        linear_layer: Module,
        group_size: int | None = None,
        bits: int | None = None,
        mode: str = "affine",
    ) -> Self: ...

class QuantizedEmbedding(Module):
    weight: array
    scales: array
    biases: array | None
    group_size: int
    bits: int
    mode: str
    def __init__(
        self,
        num_embeddings: int,
        dims: int,
        group_size: int | None = None,
        bits: int | None = None,
        mode: str = "affine",
    ) -> None: ...
    def __call__(self, x: array) -> array: ...
    def as_linear(self, x: array) -> array: ...
    @classmethod
    def from_embedding(
        cls,
        embedding_layer: Module,
        group_size: int | None = None,
        bits: int | None = None,
        mode: str = "affine",
    ) -> Self: ...

class QQLinear(Module):
    weight: array
    scales: array
    group_size: int
    bits: int
    mode: str
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        group_size: int | None = None,
        bits: int | None = None,
        mode: str = "nvfp4",
    ) -> None: ...
    def quantize(self) -> None: ...
    def __call__(self, x: array) -> array: ...

class RMSNorm(Module):
    weight: array
    eps: float
    def __init__(self, dims: int, eps: float = 1e-5) -> None: ...
    def __call__(self, x: array) -> array: ...

class AvgPool2d(Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] | None = 0,
    ) -> None: ...
    def __call__(self, x: array) -> array: ...

def gelu_approx(x: array) -> array: ...
def softplus(x: array) -> array: ...
def silu(x: array) -> array: ...

class QuantizeOptions(TypedDict, total=False):
    group_size: int
    bits: int
    mode: str
    quantize_input: bool

def quantize(
    model: Module,
    group_size: int | None = None,
    bits: int | None = None,
    *,
    mode: str = "affine",
    quantize_input: bool = False,
    class_predicate: Callable[[str, Module], bool | QuantizeOptions] | None = None,
) -> None: ...

class Upsample(Module):
    def __init__(
        self,
        scale_factor: float | tuple[float, ...],
        mode: Literal["nearest", "linear", "cubic"] = "nearest",
        align_corners: bool = False,
        antialias: bool = False,
    ) -> None: ...
    def __call__(self, x: array) -> array: ...
