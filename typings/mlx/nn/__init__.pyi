"""MLX 0.32.4 neural-network APIs used by med-vlm-mlx, from the installed source.

Covers python/mlx/nn/layers/{base,linear,embedding,positional_encoding,
normalization,activations,convolution,pooling,quantized}.py as this repository
calls them. Module is a partial interface: runtime modules still inherit from
the real MLX Module (a dict subclass), and Module.update returns the module
rather than None as dict.update does.
"""

from collections.abc import Callable, ItemsView, Iterator, Mapping, Sequence
from typing import Literal, Self

from mlx.core import array
from mlx.utils import Tree

type Parameters = dict[str, Tree[array]]
type ModuleTree = Tree[Module]
type Shape = int | Sequence[int]
# nn.quantize's predicate returns True, False or per-layer quantize kwargs.
type QuantizePredicate = Callable[[str, Module], bool | Mapping[str, object]]

class Module:
    def __init__(self) -> None: ...
    def __contains__(self, key: object) -> bool: ...
    def __getitem__(self, key: str) -> object: ...
    def __setitem__(self, key: str, value: object) -> None: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def items(self) -> ItemsView[str, object]: ...
    @property
    def training(self) -> bool: ...
    def parameters(self) -> Parameters: ...
    def children(self) -> dict[str, ModuleTree]: ...
    def leaf_modules(self) -> dict[str, ModuleTree]: ...
    def named_modules(self) -> list[tuple[str, Module]]: ...
    def update(self, parameters: Parameters, strict: bool = True) -> Self: ...
    def update_modules(
        self, modules: dict[str, ModuleTree], strict: bool = True
    ) -> Self: ...
    def load_weights(
        self, file_or_weights: str | list[tuple[str, array]], strict: bool = True
    ) -> Self: ...
    def eval(self) -> Self: ...
    @staticmethod
    def is_module(value: object) -> bool: ...

class Identity(Module):
    def __init__(self, *args: object, **kwargs: object) -> None: ...
    def __call__(self, x: array) -> array: ...

class Linear(Module):
    weight: array
    bias: array
    def __init__(
        self, input_dims: int, output_dims: int, bias: bool = True
    ) -> None: ...
    def __call__(self, x: array) -> array: ...

class Embedding(Module):
    weight: array
    def __init__(self, num_embeddings: int, dims: int) -> None: ...
    def __call__(self, x: array) -> array: ...
    def as_linear(self, x: array) -> array: ...

class RoPE(Module):
    dims: int
    traditional: bool
    base: float
    scale: float
    def __init__(
        self,
        dims: int,
        traditional: bool = False,
        base: float = 10000,
        scale: float = 1.0,
    ) -> None: ...
    # Forwarded unchanged to mx.fast.rope, which also takes per-row array offsets.
    def __call__(self, x: array, offset: int | array = 0) -> array: ...

class LayerNorm(Module):
    weight: array
    bias: array
    eps: float
    dims: int
    def __init__(
        self, dims: int, eps: float = 1e-5, affine: bool = True, bias: bool = True
    ) -> None: ...
    def __call__(self, x: array) -> array: ...

class GELU(Module):
    def __init__(
        self, approx: Literal["none", "precise", "tanh", "fast"] = "none"
    ) -> None: ...
    def __call__(self, x: array) -> array: ...

class Conv2d(Module):
    weight: array
    bias: array
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Shape,
        stride: Shape = 1,
        padding: Shape = 0,
        dilation: Shape = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None: ...
    def __call__(self, x: array) -> array: ...

class AvgPool2d(Module):
    def __init__(
        self, kernel_size: Shape, stride: Shape | None = None, padding: Shape = 0
    ) -> None: ...
    def __call__(self, x: array) -> array: ...

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
    def __call__(self, x: array) -> array: ...
    def quantize(self) -> None: ...

def quantize(
    model: Module,
    group_size: int | None = None,
    bits: int | None = None,
    *,
    mode: str = "affine",
    quantize_input: bool = False,
    class_predicate: QuantizePredicate | None = None,
) -> None: ...
def gelu(x: array) -> array: ...
def gelu_approx(x: array) -> array: ...
