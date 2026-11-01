"""Server runtime state: per-kind model caches and process-wide metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterator, Optional

from mlx_vlm.server.runtime_config import RuntimeConfig
from mlx_vlm.server.state_types import ModelResources

if TYPE_CHECKING:
    from mlx_vlm.server.generation import ResponseGenerator, ServerMetricsStore


class ModelCacheRegistry:
    """Map model kinds to their loaded resources."""

    def __init__(self, default_kind: str = "text_generation"):
        """Create an empty registry answering ``None`` kinds with ``default_kind``."""
        self.default_kind = default_kind
        self._caches: dict[str, ModelResources] = {}

    def _kind(self, kind: Optional[str] = None) -> str:
        return kind or self.default_kind

    def for_kind(self, kind: Optional[str] = None) -> ModelResources:
        """Return the resources for ``kind``, or an empty mapping when unloaded."""
        return self._caches.get(self._kind(kind), {})

    def set(self, kind: str, cache: ModelResources) -> ModelResources:
        """Store and return the resources for ``kind``."""
        self._caches[self._kind(kind)] = cache
        return cache

    def pop(self, kind: str) -> Optional[ModelResources]:
        """Remove and return the resources for ``kind``."""
        return self._caches.pop(self._kind(kind), None)

    def clear(self, kind: Optional[str] = None) -> None:
        """Drop one kind's resources, or every kind when ``kind`` is ``None``."""
        if kind is None:
            self._caches.clear()
        else:
            self._caches.pop(self._kind(kind), None)

    def __bool__(self) -> bool:
        """Report whether any model kind is loaded."""
        return bool(self._caches)

    def items(self) -> Iterator[tuple[str, ModelResources]]:
        """Iterate over loaded kinds and their resources."""
        return iter(self._caches.items())

    def values(self) -> Iterator[ModelResources]:
        """Iterate over loaded resources."""
        return iter(self._caches.values())


@dataclass
class ServerRuntime:
    """Process-wide server state shared by routes and the worker."""

    model_cache: ModelCacheRegistry = field(default_factory=ModelCacheRegistry)
    response_generator: Optional[ResponseGenerator] = None
    metrics: Optional[ServerMetricsStore] = None
    preload_failures: Dict[str, Dict[str, str]] = field(
        default_factory=dict[str, Dict[str, str]]
    )
    config: RuntimeConfig = field(default_factory=RuntimeConfig.from_env)


runtime = ServerRuntime()


def metrics_store() -> ServerMetricsStore:
    """Require the metrics store initialized by the application before serving requests."""
    if runtime.metrics is None:
        raise RuntimeError(
            "Server metrics are unavailable before application initialization"
        )
    return runtime.metrics
