"""Convert Hugging Face checkpoints into MLX safetensors and quantization recipes."""

import argparse
import glob
import shutil
from pathlib import Path
from typing import Callable, Optional, Protocol, Union, runtime_checkable

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path

from mlx_vlm.contracts import JSONObject, QuantizationOptions
from mlx_vlm.models.protocols import require_language_model
from mlx_vlm.utils import (
    MODEL_CONVERSION_DTYPES,
    create_model_card,
    fetch_from_hub,
    get_model_path,
    save_config,
    save_weights,
    skip_multimodal_module,
    upload_to_hub,
)


@runtime_checkable
class _QuantizationPredicate(Protocol):
    def quant_predicate(
        self, path: str, module: nn.Module
    ) -> bool | QuantizationOptions: ...


@runtime_checkable
class _CastPredicate(Protocol):
    def cast_predicate(self, path: str) -> bool: ...


QUANT_RECIPES = [
    "mixed_2_6",
    "mixed_3_4",
    "mixed_3_5",
    "mixed_3_6",
    "mixed_3_8",
    "mixed_4_6",
    "mixed_4_8",
]


def mixed_quant_predicate_builder(
    recipe: str, model: nn.Module
) -> Callable[[str, nn.Module], Union[bool, QuantizationOptions]]:
    """Build a mixed-bit quantization predicate for a named recipe.

    Recipes pair ``(low_bits, high_bits)`` at group size 64. Modules outside the
    indexed decoder layers stay unquantized, and the first and last eighth of
    the layers, every third layer after the first eighth, plus ``v_proj``,
    ``down_proj``, ``lm_head`` and ``embed_tokens`` receive the high bit width.
    """
    group_size = 64

    recipe_config = {
        "mixed_2_6": (2, 6),
        "mixed_3_4": (3, 4),
        "mixed_3_5": (3, 5),
        "mixed_3_6": (3, 6),
        "mixed_3_8": (3, 8),
        "mixed_4_6": (4, 6),
        "mixed_4_8": (4, 8),
    }

    if recipe not in recipe_config:
        raise ValueError(f"Invalid quant recipe {recipe}")

    low_bits, high_bits = recipe_config[recipe]

    down_keys = [k for k, _ in model.named_modules() if "down_proj" in k]
    if len(down_keys) == 0:
        raise ValueError("Model does not have expected keys for mixed quant.")

    # Look for the layer index location in the path:
    layer_location = next(
        (index for index, part in enumerate(down_keys[0].split(".")) if part.isdigit()),
        None,
    )
    if layer_location is None:
        raise ValueError("Mixed quantization requires indexed decoder layer paths")
    num_layers = len(require_language_model(model).layers)

    def mixed_quant_predicate(
        path: str,
        module: nn.Module,
    ) -> Union[bool, QuantizationOptions]:
        """Implements mixed quantization predicates with similar choices to, for example, llama.cpp's Q4_K_M.
        Ref: https://github.com/ggerganov/llama.cpp/blob/917786f43d0f29b7c77a0c56767c0fa4df68b1c5/src/llama.cpp#L5265
        By Alex Barron: https://gist.github.com/barronalex/84addb8078be21969f1690c1454855f3
        """

        if skip_multimodal_module(path):
            return False
        if not hasattr(module, "to_quantized"):
            return False
        weight = getattr(module, "weight", None)
        if not isinstance(weight, mx.array):
            raise TypeError("Quantizable module must expose an MLX weight array")
        if weight.shape[1] % group_size != 0:
            return False

        path_parts = path.split(".")
        index = 0

        if len(path_parts) > layer_location:
            element = path_parts[layer_location]
            if element.isdigit():
                index = int(element)

        use_more_bits = (
            index < num_layers // 8
            or index >= 7 * num_layers // 8
            or (index - num_layers // 8) % 3 == 2
        )

        if use_more_bits and ("v_proj" in path or "down_proj" in path):
            return {"group_size": group_size, "bits": high_bits}

        if "lm_head" in path or "embed_tokens" in path:
            return {"group_size": group_size, "bits": high_bits}

        return {"group_size": group_size, "bits": low_bits}

    return mixed_quant_predicate


@runtime_checkable
class _SavableProcessor(Protocol):
    def save_pretrained(self, save_directory: Path) -> list[str]: ...


def _cast_conversion_parameters(
    model: nn.Module, config: JSONObject, dtype: str | None
) -> None:
    """Cast floating parameters while respecting the model's native exclusions."""
    dtype_name = dtype if dtype is not None else config.get("torch_dtype")
    text_config = config.get("text_config")
    if dtype_name is None and isinstance(text_config, dict):
        dtype_name = text_config.get("dtype")
    if dtype_name is not None and not isinstance(dtype_name, str):
        raise TypeError("Model conversion dtype must be a string")
    conversion_dtypes = {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
    }
    if dtype_name in conversion_dtypes:
        selected_dtype = conversion_dtypes[dtype_name]
        print("[INFO] Using dtype:", dtype_name)

        def set_dtype(path: str, value: mx.array) -> mx.array:
            should_cast = (
                model.cast_predicate(path)
                if isinstance(model, _CastPredicate)
                else True
            )
            return (
                value.astype(selected_dtype)
                if should_cast and mx.issubdtype(value.dtype, mx.floating)
                else value
            )

        model.update(tree_map_with_path(set_dtype, model.parameters()))


def _save_conversion(
    model_path: Path,
    output_path: Path,
    model: nn.Module,
    processor: _SavableProcessor,
    config: JSONObject,
    hf_path: str,
    upload_repo: str | None,
) -> None:
    """Write converted parameters and processor sidecars to one output directory."""
    save_weights(output_path, model, donate_weights=True)

    # Copy Python and JSON files from the model path to the MLX path
    for pattern in ["*.py", "*.json"]:
        files = glob.glob(str(model_path / pattern))
        for file in files:
            # Skip the index file - save_weights() already generated the correct one
            if Path(file).name == "model.safetensors.index.json":
                continue
            shutil.copy(file, output_path)

    # Copy folders from the model path to the MLX path
    for item in model_path.iterdir():
        if item.is_dir():
            dest = output_path / item.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)

    processor.save_pretrained(output_path)

    save_config(config, config_path=output_path / "config.json")

    hf_repo = None if Path(hf_path).exists() else hf_path
    create_model_card(output_path, hf_repo)

    if upload_repo is not None:
        upload_to_hub(str(output_path), upload_repo)


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_group_size: int | None = 64,
    q_bits: int | None = 4,
    q_mode: str = "affine",
    dtype: Optional[str] = None,
    upload_repo: str | None = None,
    revision: Optional[str] = None,
    dequantize: bool = False,
    quant_predicate: Optional[str] = None,
):
    """Convert a supported checkpoint to an MLX directory, optionally quantizing.

    Loads ``hf_path`` lazily without running checkpoint code, optionally casts
    floating parameters, quantizes or dequantizes, and writes the weights,
    config, processor files and the source ``*.py``/``*.json`` files and
    subfolders to ``mlx_path``. Vision, projector and similar multimodal
    modules are never quantized.

    Args:
        hf_path: Local model directory or Hugging Face model ID.
        mlx_path: Output directory.
        quantize: Quantize the weights; mutually exclusive with ``dequantize``.
        q_group_size: Quantization group size; ``None`` selects the default for
            ``q_mode``.
        q_bits: Bits per weight; ``None`` selects the default for ``q_mode``.
        q_mode: MLX quantization mode (``affine``, ``mxfp4``, ``nvfp4`` or
            ``mxfp8`` on the command line).
        dtype: Cast floating parameters to this dtype; defaults to the
            checkpoint's declared dtype.
        upload_repo: Hugging Face repository to upload the result to.
        revision: Hub revision when ``hf_path`` is a Hub ID.
        dequantize: Dequantize supported MLX quantized layers. The saved
            configuration retains its original quantization metadata.
        quant_predicate: Name of a mixed-bit recipe such as ``mixed_4_6``.

    Raises:
        ValueError: If both ``quantize`` and ``dequantize`` are set, the model
            type is unsupported, or the recipe is invalid.
    """
    print("[INFO] Loading")
    model_path = get_model_path(hf_path, revision=revision)
    model, config, processor = fetch_from_hub(
        model_path, lazy=True, trust_remote_code=False
    )

    model_quant_predicate = model if isinstance(model, _QuantizationPredicate) else None

    def base_quant_predicate(
        path: str, module: nn.Module
    ) -> bool | QuantizationOptions:
        if skip_multimodal_module(path):
            return False
        if model_quant_predicate is not None:
            return model_quant_predicate.quant_predicate(path, module)
        return True

    target = model

    predicate = (
        mixed_quant_predicate_builder(quant_predicate, target)
        if quant_predicate is not None
        else base_quant_predicate
    )

    _cast_conversion_parameters(model, config, dtype)

    if quantize and dequantize:
        raise ValueError("Choose either quantize or dequantize, not both.")

    if quantize:
        from mlx_vlm.quant_utils import quantize_model

        print("[INFO] Quantizing")
        config.setdefault("vision_config", {})
        target, config = quantize_model(
            target,
            config,
            q_group_size,
            q_bits,
            mode=q_mode,
            quant_predicate=predicate,
        )

    if dequantize:
        from mlx_vlm.quant_utils import dequantize_model

        print("[INFO] Dequantizing")
        target = dequantize_model(target)

    _save_conversion(
        model_path, Path(mlx_path), target, processor, config, hf_path, upload_repo
    )


def configure_parser() -> argparse.ArgumentParser:
    """
    Configures and returns the argument parser for the script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face model to MLX format"
    )
    parser.add_argument(
        "--hf-path",
        "--model",
        type=str,
        help="Path to the model. This can be a local path or a Hugging Face Hub model identifier.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        help="Hugging Face revision (branch), when converting a model from the Hub.",
        default=None,
    )
    parser.add_argument(
        "--mlx-path", type=str, default="mlx_model", help="Path to save the MLX model."
    )
    parser.add_argument(
        "-q", "--quantize", help="Generate a quantized model.", action="store_true"
    )
    parser.add_argument(
        "--q-group-size",
        help="Group size for quantization.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--q-bits",
        help="Bits per weight for quantization.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--q-mode",
        help="The quantization mode.",
        type=str,
        choices=["affine", "mxfp4", "nvfp4", "mxfp8"],
        default="affine",
    )
    parser.add_argument(
        "--dtype",
        help="Type to save the parameter. Defaults to config.json's `torch_dtype` or the current model weights dtype",
        type=str,
        choices=MODEL_CONVERSION_DTYPES,
        default=None,
    )
    parser.add_argument(
        "--quant-predicate",
        help=f"Mixed-bit quantization recipe.",
        choices=QUANT_RECIPES,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--upload-repo",
        help="The Hugging Face repo to upload the model to.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-d",
        "--dequantize",
        help="Dequantize a quantized model.",
        action="store_true",
        default=False,
    )
    return parser


def main() -> None:
    """Run the conversion command line."""
    parser = configure_parser()
    args = parser.parse_args()
    convert(**vars(args))


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_vlm.convert ...` directly is deprecated."
        " Use `mlx_vlm.convert ...` or `python -m mlx_vlm convert ...` instead."
    )
    main()
