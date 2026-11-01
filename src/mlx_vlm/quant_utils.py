"""Weight quantization helpers adapted from mlx-lm 0.31.3.

The local conversion path does not import mlx-lm; selection, metadata and
supported module handling follow this implementation rather than promising
parity with every upstream version.
"""

import copy
from typing import Callable, Optional, Protocol, Tuple, Union, runtime_checkable

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_reduce, tree_unflatten

from mlx_vlm.contracts import JSONObject, QuantizationOptions

QUANTIZATION_MODE_DEFAULTS = {
    "affine": (64, 4),
    "mxfp4": (32, 4),
    "nvfp4": (16, 4),
    "mxfp8": (32, 8),
}


def get_quantization_params(
    group_size: Optional[int],
    bits: Optional[int],
    mode: str,
) -> QuantizationOptions:
    """Resolve mode defaults and fixed non-affine formats.

    Falsy ``group_size`` or ``bits`` values select defaults, including zero.
    The affine mode leaves other values to the downstream MLX operation.
    """
    if mode not in QUANTIZATION_MODE_DEFAULTS:
        raise ValueError(f"Unsupported quantization mode: {mode}")
    default_group_size, default_bits = QUANTIZATION_MODE_DEFAULTS[mode]
    resolved_group_size = group_size or default_group_size
    resolved_bits = bits or default_bits
    if mode != "affine" and (resolved_group_size, resolved_bits) != (
        default_group_size,
        default_bits,
    ):
        raise ValueError(
            f"{mode} requires group_size={default_group_size}, bits={default_bits}"
        )
    return {
        "group_size": resolved_group_size,
        "bits": resolved_bits,
        "mode": mode,
    }


@runtime_checkable
class _QuantizationPredicate(Protocol):
    def quant_predicate(
        self, path: str, module: nn.Module
    ) -> bool | QuantizationOptions: ...


def get_total_parameters(model: nn.Module) -> int:
    """Count parameters, expanding packed quantized weights to logical elements."""
    leaf_modules = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)

    def nparams(m: nn.Module) -> int:
        weight = getattr(m, "weight", None)
        bits = getattr(m, "bits", None)
        if isinstance(weight, mx.array) and isinstance(bits, int):
            if bits <= 0:
                raise ValueError("Quantized module bits must be positive")
            bias = getattr(m, "bias", None)
            if bias is not None and not isinstance(bias, mx.array):
                raise TypeError("Quantized module bias must be an MLX array")
            n = 0 if bias is None else bias.size
            return n + weight.size * 32 // bits
        return sum(v.size for _, v in tree_flatten(m.parameters()))

    return sum(nparams(m) for _, m in leaf_modules)


def compute_bits_per_weight(model: nn.Module) -> float:
    """Return stored bits per parameter across the whole model."""

    def add_bytes(acc: int, value: object) -> int:
        return acc + value.nbytes if isinstance(value, mx.array) else acc

    model_bytes = tree_reduce(add_bytes, model, 0)
    model_params = get_total_parameters(model)
    return model_bytes * 8 / model_params


def _quantization_json(options: QuantizationOptions) -> JSONObject:
    """Copy supported quantization fields into serializable model metadata."""
    fields = (
        ("group_size", options.get("group_size")),
        ("bits", options.get("bits")),
        ("mode", options.get("mode")),
        ("scheme", options.get("scheme")),
        ("quantize_input", options.get("quantize_input")),
    )
    return {key: value for key, value in fields if value is not None}


def quantize_model(
    model: nn.Module,
    config: JSONObject,
    group_size: Optional[int],
    bits: Optional[int],
    mode: str = "affine",
    quant_predicate: Optional[
        Callable[[str, nn.Module], Union[bool, QuantizationOptions]]
    ] = None,
) -> Tuple[nn.Module, JSONObject]:
    """
    Applies quantization to the model weights.

    Args:
        model (nn.Module): The model to be quantized.
        config (dict): Model configuration.
        group_size (Optional[int]): Group size for quantization.
        bits (Optional[int]): Bits per weight for quantization.
        mode (str): The quantization mode.
        quant_predicate (Callable): A callable that decides how to quantize
          each layer based on the path. Accepts the layer `path` and the
          `module`. Returns either a bool to signify quantize/no quantize or
          a dict of quantization parameters to pass to `to_quantized`.

    Returns:
        Tuple: The model (quantized in place) and a deep-copied config containing
            quantization metadata. The caller's original config is not modified.

    Modules without a quantization method or with incompatible input dimensions
    are skipped. A predicate may supply a compatible fallback group size.
    """

    quantized_config = copy.deepcopy(config)

    if quant_predicate is None and isinstance(model, _QuantizationPredicate):
        quant_predicate = model.quant_predicate
    quant_params = get_quantization_params(group_size, bits, mode)
    group_size = quant_params.get("group_size")
    bits = quant_params.get("bits")
    if group_size is None or bits is None:
        raise ValueError("Resolved quantization requires group_size and bits")
    resolved_group_size = group_size
    if "quantization" in quantized_config:
        # If the model is already partially quantized, return params so that
        # the config is set on a per-layer basis
        fine_grained_config = True
        quantization = quantized_config["quantization"]
        if not isinstance(quantization, dict):
            raise ValueError("Model quantization metadata must be a JSON object")
    else:
        fine_grained_config = False
        quantization = _quantization_json(quant_params)
        quantized_config["quantization"] = quantization

    def wrapped_predicate(path: str, module: nn.Module) -> bool | QuantizationOptions:
        weight = getattr(module, "weight", None)
        if not hasattr(module, "to_quantized") or not isinstance(weight, mx.array):
            return False

        input_dims = weight.shape[-1]
        bool_or_params: bool | QuantizationOptions = True
        default_group_is_compatible = input_dims % resolved_group_size == 0
        if not default_group_is_compatible:
            if quant_predicate is None:
                return False
            bool_or_params = quant_predicate(path, module)
            if not (
                isinstance(bool_or_params, dict)
                and "fallback_group_size" in bool_or_params
            ):
                return False
        elif quant_predicate is not None:
            bool_or_params = quant_predicate(path, module)

        if isinstance(bool_or_params, dict) and "fallback_group_size" in bool_or_params:
            overrides = bool_or_params.copy()
            fallback_group_size = overrides.pop("fallback_group_size", None)
            bool_or_params = {**quant_params, **overrides}
            if (
                input_dims % bool_or_params.get("group_size", resolved_group_size)
                and fallback_group_size is not None
                and input_dims % fallback_group_size == 0
            ):
                bool_or_params["group_size"] = fallback_group_size
        module_group_size = (
            bool_or_params.get("group_size", resolved_group_size)
            if isinstance(bool_or_params, dict)
            else resolved_group_size
        )
        if input_dims % module_group_size != 0:
            return False
        if isinstance(bool_or_params, dict):
            quantization[path] = _quantization_json(bool_or_params)
        elif fine_grained_config and bool_or_params:
            quantization[path] = _quantization_json(quant_params)
        return bool_or_params

    nn.quantize(
        model,
        group_size,
        bits,
        mode=mode,
        class_predicate=wrapped_predicate,
    )
    # support hf model tree #957
    quantized_config["quantization_config"] = quantization

    bpw = compute_bits_per_weight(model)
    print(f"[INFO] Quantized model with {bpw:.3f} bits per weight.")

    return model, quantized_config


def dequantize_model(model: nn.Module) -> nn.Module:
    """
    Dequantize MLX quantized linear and embedding layers in the model.

    ``OneBitLinear`` and ``OneBitEmbedding`` are not converted by this helper.

    Args:
        model (nn.Module): The model with quantized layers.

    Returns:
        nn.Module: The model with dequantized layers.
    """
    dequantize_layers: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            continue
        weight = mx.dequantize(
            module.weight,
            module.scales,
            module.biases,
            module.group_size,
            module.bits,
            module.mode,
        )
        m: nn.Linear | nn.Embedding
        if isinstance(module, nn.QuantizedEmbedding):
            num_embeddings, dims = weight.shape
            m = nn.Embedding(num_embeddings, dims)
        else:
            output_dims, input_dims = weight.shape
            m = nn.Linear(input_dims, output_dims, bias="bias" in module)
            if "bias" in module:
                m.bias = module.bias
        m.weight = weight
        dequantize_layers.append((name, m))

    if dequantize_layers:
        modules = tree_unflatten(dequantize_layers)
        if not isinstance(modules, dict):
            raise TypeError("Dequantized model modules require a named root")
        model.update_modules(modules)
    return model
