"""Image input types shared by the native processor and input boundary."""

from collections.abc import Sequence
from pathlib import Path
from typing import TypeIs, cast

from PIL import Image

from mlx_vlm.contracts import ImageSource, InputTensor, is_input_tensor

type ImageBatch = ImageSource | InputTensor | list[ImageBatch] | tuple[ImageBatch, ...]


def is_image_batch(value: object) -> TypeIs[ImageBatch]:
    """Return whether a value is an image source or nested sequence of them."""
    if isinstance(value, (str, Path, Image.Image)) or is_input_tensor(value):
        return True
    if isinstance(value, (list, tuple)):
        return all(is_image_batch(item) for item in cast(Sequence[object], value))
    return False
