"""Public request defaults must survive the typing-only schema cleanup."""

from typing import Literal

import pytest
from pydantic import TypeAdapter

from mlx_vlm.server.schemas import ResponseInputImageParam


@pytest.mark.parametrize("detail", [None, "high", "low", "auto"])
def test_image_input_preserves_omitted_and_explicit_detail(
    detail: Literal["high", "low", "auto"] | None,
):
    payload: ResponseInputImageParam = {
        "type": "input_image",
        "image_url": "https://example.com/image.png",
    }
    if detail is not None:
        payload["detail"] = detail
    assert TypeAdapter(ResponseInputImageParam).validate_python(payload) == payload
