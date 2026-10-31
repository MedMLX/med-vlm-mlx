"""Typed boundary for the native MLX custom-kernel factory.

MLX 0.32 declares the factory result as ``object`` in its stub. The protocol
below follows the native callable's documented signature; execution, stream
selection, compilation options, and array dtypes remain owned by MLX.
"""

from collections.abc import Sequence
from typing import Literal, Protocol, TypedDict, cast

import mlx.core as mx


class MetalKernel(Protocol):
    """Callable returned by ``mlx.core.fast.metal_kernel``."""

    def __call__(
        self,
        *,
        inputs: Sequence[mx.array | int | float | bool],
        output_shapes: Sequence[Sequence[int]],
        output_dtypes: Sequence[mx.Dtype],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        template: Sequence[tuple[str, bool | int | mx.Dtype]] | None = None,
        init_value: float | None = None,
        verbose: bool = False,
        stream: (
            mx.Stream | mx.ThreadLocalStream | mx.Device | mx.DeviceType | None
        ) = None,
    ) -> list[mx.array]:
        """Launch the kernel and return one materialized array per output."""
        ...


class MetalCompileOptions(TypedDict, total=False):
    """Math mode forwarded to the Metal kernel compiler."""

    math_mode: Literal["safe", "relaxed", "fast"]


def metal_kernel(
    name: str,
    input_names: Sequence[str],
    output_names: Sequence[str],
    source: str,
    header: str = "",
    ensure_row_contiguous: bool = True,
    atomic_outputs: bool = False,
    compile_options: MetalCompileOptions | None = None,
) -> MetalKernel:
    """Create a Metal kernel whose invocation takes named inputs and grid sizes."""
    # Trusted native dependency boundary: the factory returns the callable
    # documented by mlx.core.fast.metal_kernel, rather than an arbitrary object.
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name,
            input_names,
            output_names,
            source,
            header,
            ensure_row_contiguous,
            atomic_outputs,
            compile_options,
        ),
    )
