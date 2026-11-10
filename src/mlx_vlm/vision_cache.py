"""Vision feature cache for multi-turn conversations.

Caches the output of vision_tower + embed_vision (projected image features
in language model space) keyed by image path or content hash, avoiding
expensive re-computation when the same image is discussed across turns.
"""

import hashlib
from collections import OrderedDict
from collections.abc import Hashable
from pathlib import Path
from typing import Optional, Sequence, Union

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_vlm.contracts import ImageSource

VisionFeatures = Union[mx.array, Sequence[mx.array]]


class VisionFeatureCache:
    """LRU cache for vision features projected into language model space.

    Cache keys preserve sequence boundaries and distinguish source types.
    Paths and URLs are keyed by their text, PIL images by RGB pixels and size,
    and arrays by dtype, shape and content. Cached values are one feature array
    or an ordered collection of per-image feature arrays after vision projection.
    File keys include their current version metadata. Generation keys the
    processed pixel tensors instead, so changes to source content or image
    preprocessing cannot reuse stale features. Namespaces separate model states.
    Feature arrays are borrowed and must remain unmodified while cached. Input
    collections are snapshotted as tuples; lookups return borrowed arrays too.

    Cleanup is handled by three mechanisms:
    - **LRU eviction**: oldest entry is dropped when max_size is exceeded.
    - **Model unload**: server calls clear() when the model is swapped.
    - **Process exit**: in-memory cache is freed automatically.

    Args:
        max_size: Maximum number of cached image features. Default 20.
        max_bytes: Maximum feature payload retained, in bytes. Default 256 MiB.
    """

    def __init__(self, max_size: int = 20, max_bytes: int = 256 * 1024 * 1024):
        """Bound retained feature payload by both entry count and bytes."""
        if max_size < 0 or max_bytes < 0:
            raise ValueError("Vision cache limits must be non-negative")
        self.max_size = max_size
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self._cache: OrderedDict[tuple[object, ...], VisionFeatures] = OrderedDict()

    def _make_key(
        self, image_source: ImageSource | Sequence[ImageSource]
    ) -> tuple[object, ...]:
        """Derive a structural key for supported image sources."""
        if isinstance(image_source, (str, Path)):
            path = Path(image_source)
            if path.is_file():
                stat = path.stat()
                return (
                    "file",
                    str(path.resolve()),
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                    stat.st_ino,
                )
            return ("source", str(image_source))
        if isinstance(image_source, (list, tuple)):
            return ("sequence", tuple(self._make_key(img) for img in image_source))
        if isinstance(image_source, Image.Image):
            rgb = image_source.convert("RGB")
            return ("image", rgb.size, hashlib.sha256(rgb.tobytes()).digest())
        if isinstance(image_source, mx.array):
            data = np.asarray(image_source.view(mx.uint8)).tobytes()
            dtype = str(image_source.dtype).removeprefix("mlx.core.")
        elif isinstance(image_source, np.ndarray):
            data = image_source.tobytes()
            dtype = image_source.dtype.name
        else:
            return ("sequence", tuple(self._make_key(img) for img in image_source))
        return (
            "array",
            tuple(image_source.shape),
            dtype,
            hashlib.sha256(data).digest(),
        )

    def get(
        self,
        image_source: ImageSource | Sequence[ImageSource],
        *,
        namespace: Hashable | None = None,
    ) -> Optional[VisionFeatures]:
        """Look up cached features. Returns None on miss."""
        key = (namespace, self._make_key(image_source))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(
        self,
        image_source: ImageSource | Sequence[ImageSource],
        features: VisionFeatures,
        *,
        namespace: Hashable | None = None,
    ) -> None:
        """Store features in the cache, evicting LRU if full."""
        features = features if isinstance(features, mx.array) else tuple(features)
        key = (namespace, self._make_key(image_source))
        previous = self._cache.pop(key, None)
        if previous is not None:
            self.current_bytes -= self._feature_bytes(previous)
        size = self._feature_bytes(features)
        if not self.max_size or size > self.max_bytes:
            return
        while self._cache and (
            len(self._cache) >= self.max_size
            or self.current_bytes + size > self.max_bytes
        ):
            _, evicted = self._cache.popitem(last=False)
            self.current_bytes -= self._feature_bytes(evicted)
        self._cache[key] = features
        self.current_bytes += size

    @staticmethod
    def _feature_bytes(features: VisionFeatures) -> int:
        """Count logical feature bytes without retaining encoder intermediates."""
        if isinstance(features, mx.array):
            return features.nbytes
        return sum(value.nbytes for value in features)

    def clear(self) -> None:
        """Clear all cached features."""
        self._cache.clear()
        self.current_bytes = 0

    def __len__(self) -> int:
        """Return the number of cached image features."""
        return len(self._cache)

    def __contains__(self, image_source: ImageSource | Sequence[ImageSource]) -> bool:
        """Report whether features for the source are cached."""
        key = (None, self._make_key(image_source))
        return key in self._cache
