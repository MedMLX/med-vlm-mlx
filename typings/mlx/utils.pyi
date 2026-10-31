"""Used tree operations from MLX 0.32.3's installed mlx/utils.py."""

from collections.abc import Callable
from typing import overload

from mlx.core import array
from mlx.nn import Module, ModuleNode, Parameters
from typing_extensions import TypeIs

type Tree[T] = T | list[Tree[T]] | tuple[Tree[T], ...] | dict[str, Tree[T]]

@overload
def tree_flatten(
    tree: Parameters,
    prefix: str = "",
    is_leaf: None = None,
    destination: None = None,
) -> list[tuple[str, array]]: ...
@overload
def tree_flatten(
    tree: dict[str, ModuleNode],
    prefix: str = "",
    *,
    is_leaf: Callable[[object], TypeIs[Module]],
    destination: None = None,
) -> list[tuple[str, Module]]: ...
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
@overload
def tree_unflatten(
    tree: list[tuple[str, Module]] | dict[str, Module],
) -> ModuleNode: ...
@overload
def tree_unflatten[T](tree: list[tuple[str, T]] | dict[str, T]) -> Tree[T]: ...
@overload
def tree_map(
    fn: Callable[[Module], Module],
    tree: dict[str, ModuleNode],
    *,
    is_leaf: Callable[[object], TypeIs[Module]],
) -> dict[str, ModuleNode]: ...
@overload
def tree_map[T, U](
    fn: Callable[[T], U], tree: dict[str, Tree[T]]
) -> dict[str, Tree[U]]: ...
@overload
def tree_map[T, U](fn: Callable[[T], U], tree: Tree[T]) -> Tree[U]: ...
def tree_reduce[T](
    fn: Callable[[T, object], T],
    tree: object,
    initializer: T,
    is_leaf: Callable[[object], bool] | None = None,
) -> T: ...
@overload
def tree_map_with_path(
    fn: Callable[[str, array], array],
    tree: Parameters,
    *,
    is_leaf: None = None,
    path: str | None = None,
) -> Parameters: ...
@overload
def tree_map_with_path(
    fn: Callable[[str, Module], Module],
    tree: dict[str, ModuleNode],
    *,
    is_leaf: Callable[[object], TypeIs[Module]],
    path: str | None = None,
) -> dict[str, ModuleNode]: ...
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
    fn: Callable[[str | None, T], U],
    tree: Tree[T],
    *,
    is_leaf: Callable[[object], bool] | None = None,
    path: str | None = None,
) -> Tree[U]: ...
