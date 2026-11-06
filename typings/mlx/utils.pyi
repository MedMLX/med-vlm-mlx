"""MLX 0.32.4 tree utilities used by med-vlm-mlx, from the installed mlx/utils.py.

MLX ships stubs for mlx.core only. These declarations cover the calls this
repository makes; they are not complete coverage of mlx.utils.
"""

from collections.abc import Callable
from typing import overload

type Tree[T] = T | list[Tree[T]] | tuple[Tree[T], ...] | dict[str, Tree[T]]

# A dict tree maps to a dict, as the runtime rebuilds each container type.
@overload
def tree_map[T, U](
    fn: Callable[[T], U],
    tree: dict[str, Tree[T]],
    *,
    is_leaf: Callable[[object], bool] | None = None,
) -> dict[str, Tree[U]]: ...
@overload
def tree_map[T, U](
    fn: Callable[[T], U],
    tree: Tree[T],
    *,
    is_leaf: Callable[[object], bool] | None = None,
) -> Tree[U]: ...
@overload
def tree_map_with_path[T, U](
    fn: Callable[[str, T], U],
    tree: dict[str, Tree[T]],
    *,
    is_leaf: Callable[[object], bool] | None = None,
    path: str | None = None,
) -> dict[str, Tree[U]]: ...
@overload
def tree_map_with_path[T, U](
    fn: Callable[[str, T], U],
    tree: Tree[T],
    *,
    is_leaf: Callable[[object], bool] | None = None,
    path: str | None = None,
) -> Tree[U]: ...
@overload
def tree_flatten[T](
    tree: Tree[T],
    prefix: str = "",
    is_leaf: Callable[[object], bool] | None = None,
    destination: None = None,
) -> list[tuple[str, T]]: ...
@overload
def tree_flatten[T](
    tree: Tree[T],
    prefix: str = "",
    is_leaf: Callable[[object], bool] | None = None,
    *,
    destination: dict[str, T],
) -> dict[str, T]: ...
def tree_unflatten[T](tree: list[tuple[str, T]] | dict[str, T]) -> Tree[T]: ...
def tree_reduce[T, A](
    fn: Callable[[A, T], A],
    tree: Tree[T] | object,
    initializer: A,
    is_leaf: Callable[[object], bool] | None = None,
) -> A: ...
