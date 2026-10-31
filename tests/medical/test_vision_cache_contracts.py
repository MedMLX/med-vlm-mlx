"""Synthetic cache identity cases for supported image sources."""

from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_vlm.vision_cache import VisionFeatureCache


def test_sequence_keys_preserve_item_boundaries() -> None:
    cache = VisionFeatureCache()
    cache.put(["a|b", "c"], mx.array([1]))
    assert cache.get(["a", "b|c"]) is None
    assert cache.get(["a|b", "c"]) is not None
    assert cache.get(("a|b", "c")) is not None


def test_pil_keys_include_geometry_and_palette() -> None:
    cache = VisionFeatureCache()
    first = Image.frombytes("RGB", (1, 2), bytes(range(6)))
    second = Image.frombytes("RGB", (2, 1), bytes(range(6)))
    cache.put(first, mx.array([1]))
    assert cache.get(second) is None
    assert cache.get(first.copy()) is not None

    first = Image.new("P", (1, 1))
    second = first.copy()
    second.putpalette([255, 0, 0] + [0] * 765)
    cache.put(first, mx.array([2]))
    assert cache.get(second) is None


def test_array_keys_include_shape_dtype_and_content() -> None:
    cache = VisionFeatureCache()
    first = np.arange(4, dtype=np.uint8).reshape((2, 2))
    cache.put(first, mx.array([1]))
    assert cache.get(first.copy()) is not None
    assert cache.get(first.reshape((1, 4))) is None
    assert cache.get(first.view(np.uint16)) is None
    assert cache.get(mx.array(first)) is not None


def test_path_keys_compare_by_value() -> None:
    cache = VisionFeatureCache()
    cache.put(Path("images/scan.png"), mx.array([1]))
    assert cache.get(Path("images/scan.png")) is not None
    assert cache.get("images/scan.png") is not None
