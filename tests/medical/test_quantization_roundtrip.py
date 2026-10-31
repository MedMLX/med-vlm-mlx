"""Tiny native quantization/dequantization roundtrip without pretrained weights."""

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.contracts import JSONObject, evaluate
from mlx_vlm.quant_utils import dequantize_model, quantize_model


class TinyProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection: nn.Linear | nn.QuantizedLinear = nn.Linear(64, 8, bias=False)

    def __call__(self, values: mx.array) -> mx.array:
        return self.projection(values)


def test_quantized_projection_dequantizes_without_changing_quantized_output() -> None:
    model = TinyProjection()
    config: JSONObject = {"model_type": "synthetic"}
    quantized, metadata = quantize_model(model, config, group_size=64, bits=4)
    assert quantized is model
    assert isinstance(model.projection, nn.QuantizedLinear)
    assert config == {"model_type": "synthetic"}
    assert metadata["quantization"] == {"group_size": 64, "bits": 4, "mode": "affine"}
    assert metadata["quantization_config"] == metadata["quantization"]
    inputs = mx.ones((1, 64))
    packed_output = model(inputs)
    evaluate(packed_output)
    restored = dequantize_model(model)
    assert restored is model
    assert isinstance(model.projection, nn.Linear)
    assert not isinstance(model.projection, nn.QuantizedLinear)
    dense_output = model(inputs)
    evaluate(dense_output)
    assert mx.allclose(packed_output, dense_output, atol=1e-5, rtol=1e-5).item()
