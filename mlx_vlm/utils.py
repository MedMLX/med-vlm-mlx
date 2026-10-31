"""Checkpoint loading, conversion and multimodal prompt preparation."""

from __future__ import annotations

import glob
import importlib
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Callable,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Union,
    cast,
)

import huggingface_hub
import mlx.core as mx
import mlx.nn as nn
import requests
from mlx.utils import tree_flatten, tree_map
from PIL import Image, ImageOps
from pydantic import TypeAdapter
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing_extensions import TypedDict

from mlx_vlm.contracts import (
    ImageSource,
    JSONObject,
    JSONValue,
    ProcessedInputs,
    QuantizationConfig,
    QuantizationOptions,
    evaluate,
    is_json_object,
    parse_int,
    require_processed_inputs,
    save_safetensors,
)
from mlx_vlm.models.base import BaseModelConfig, to_mlx
from mlx_vlm.models.gemma3 import Model as GemmaModel
from mlx_vlm.models.gemma3 import ModelConfig as GemmaConfig
from mlx_vlm.models.gemma3.processing_gemma3 import Gemma3Processor
from mlx_vlm.models.media_types import is_image_batch
from mlx_vlm.models.processing import tokenize_inputs
from mlx_vlm.tokenizer_utils import load_detokenizer_factory

if TYPE_CHECKING:
    from mlx_vlm.generate.types import TokenEncoder


type MedicalModel = GemmaModel


logger = logging.getLogger(__name__)

# Modes that support activation quantization
ACTIVATION_QUANTIZATION_MODES = {"nvfp4", "mxfp8"}

# Constants
SUPPORTED_MODEL_TYPES = ("gemma3",)

MAX_FILE_SIZE_GB = 5

MODEL_CONVERSION_DTYPES = ["float16", "bfloat16", "float32"]

GENERATION_CONFIG_DEFAULT_KEYS = (
    "eos_token_id",
    "temperature",
    "top_p",
    "top_k",
    "do_sample",
)


def apply_generation_config_defaults[T: BaseModelConfig](
    model_config: T, config: JSONObject
) -> T:
    """Copy generation keys from a checkpoint object onto the model config."""
    for key in GENERATION_CONFIG_DEFAULT_KEYS:
        if key in config:
            setattr(model_config, key, config[key])
    return model_config


def _merge_generation_config(config: JSONObject, generation_config: JSONObject) -> None:
    if not generation_config:
        return

    config["generation_config"] = generation_config
    for key in GENERATION_CONFIG_DEFAULT_KEYS:
        if key in generation_config:
            config[key] = generation_config[key]


def quantize_activations[T: nn.Module](model: T) -> T:
    """Convert eligible quantized linear layers to activation-quantized QQLinear.

    Layers outside the ``nvfp4``/``mxfp8`` modes or carrying a bias raise
    ``ValueError`` instead of being converted.
    """

    def _maybe_qq(m: nn.Module) -> nn.Module:
        """Convert a QuantizedLinear layer to QQLinear if compatible."""
        if isinstance(m, nn.QuantizedLinear):
            if m.mode not in ACTIVATION_QUANTIZATION_MODES:
                raise ValueError(
                    f"Mode ({m.mode}) does not support activation quantization. "
                    f"Supported modes: {', '.join(ACTIVATION_QUANTIZATION_MODES)}"
                )
            if "bias" in m:
                raise ValueError(
                    "Linear layer with bias does not support activation quantization"
                )
            # Get dimensions from quantized weight
            out_dims, in_dims = m.weight.shape
            in_dims *= 32 // m.bits
            qq = nn.QQLinear(in_dims, out_dims, m.group_size, m.bits, m.mode)
            qq.quantize()
            return qq
        return m

    leaves = tree_map(_maybe_qq, model.leaf_modules(), is_leaf=nn.Module.is_module)
    model.update_modules(leaves)

    return model


def skip_multimodal_module(path: str) -> bool:
    """Whether ``path`` lies in MedGemma's vision tower or projector."""
    return any(module in path for module in ("vision_tower", "multi_modal_projector"))


def get_model_and_args(
    config: JSONObject, model_path: Optional[Path] = None
) -> tuple[ModuleType, str]:
    """Resolve the maintained MedGemma model architecture.

    Checkpoint Python modules are rejected before import. MedGemma uses
    Gemma3; other architectures are outside this runtime, even when their
    weights share a serialization format.
    """
    if config.get("model_file"):
        raise ValueError(
            "Checkpoint model_file code is not supported by the medical runtime"
        )
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model type {model_type!r}; expected gemma3 (MedGemma)"
        )
    return importlib.import_module(f"mlx_vlm.models.{model_type}"), model_type


def _quantization_path_aliases(
    path: str, model: Optional[nn.Module] = None
) -> Tuple[str, ...]:
    """Return checkpoint quantization keys that may refer to a module path."""
    aliases = [path]
    if path.startswith("language_model."):
        aliases.append(path[len("language_model.") :])

    return tuple(dict.fromkeys(aliases))


def _quantization_for_module_path(
    quantization: QuantizationConfig, path: str, model: Optional[nn.Module] = None
) -> QuantizationOptions | None:
    for alias in _quantization_path_aliases(path, model):
        value = quantization.get(alias)
        if isinstance(value, dict):
            return value
        if value is False:
            return {}
    return None


def _drop_modules_without_weights(
    model: nn.Module,
    weights: dict[str, mx.array],
    declared_keys: Optional[set[str]] = None,
) -> None:
    """Drop weightless top-level VLM modules the checkpoint manifest also omits."""
    weighted_modules = {key.partition(".")[0] for key in weights}
    declared_modules = {key.partition(".")[0] for key in (declared_keys or weights)}
    dropped_modules: list[str] = []
    for name, child in list(model.items()):
        if name == "language_model" or not isinstance(child, nn.Module):
            continue
        if not tree_flatten(child.parameters()) or name in weighted_modules:
            continue
        if name in declared_modules:
            continue
        setattr(model, name, None)
        dropped_modules.append(name)

    if dropped_modules:
        logging.warning(
            "Checkpoint has no weights for module(s): %s. Disabling them.",
            ", ".join(dropped_modules),
        )


class _SnapshotDownload(Protocol):
    def __call__(
        self,
        *,
        repo_id: str,
        revision: str | None,
        allow_patterns: list[str],
        force_download: bool,
        cache_dir: str | None,
        local_files_only: bool,
        token: str | bool | None,
    ) -> str: ...


def get_model_path(
    path_or_hf_repo: str,
    revision: Optional[str] = None,
    force_download: bool = False,
    allow_patterns: Optional[List[str]] = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    token: str | bool | None = None,
) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.
        revision (str, optional): A revision id which can be a branch name, a tag, or a commit hash.
        force_download: Download again even if a snapshot is cached.
        allow_patterns: File patterns to download, or the default checkpoint patterns.
        cache_dir: Hugging Face cache directory, or the Hub default when omitted.
        local_files_only: Resolve only from the cache without a Hub request.
        token: Hugging Face authentication token, or Hub default when omitted.

    Returns:
        Path: The path to the model.
    """
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(
            cast(_SnapshotDownload, getattr(huggingface_hub, "snapshot_download"))(
                repo_id=path_or_hf_repo,
                revision=revision,
                allow_patterns=allow_patterns
                or [
                    "*.json",
                    "*.jsonl",
                    "*.safetensors",
                    "*.py",
                    "*.model",
                    "*.tiktoken",
                    "*.txt",
                    "*.jinja",
                ],
                force_download=force_download,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                token=token,
            )
        )
    return model_path


class CheckpointOptions(TypedDict, total=False, extra_items=JSONValue):
    """Loader options accepted by checkpoint discovery and loading."""

    cache_dir: str | None
    revision: str
    force_download: bool
    local_files_only: bool
    token: str | bool | None
    strict: bool
    quantize_activations: bool
    trust_remote_code: bool


def _native_checkpoint_options(kwargs: Mapping[str, object]) -> CheckpointOptions:
    """Validate raw loader options and reject checkpoint code before resolution."""
    if kwargs.get("trust_remote_code"):
        raise ValueError("Checkpoint Python code is outside the medical runtime")
    options = dict(kwargs)
    if isinstance(options.get("cache_dir"), Path):
        options["cache_dir"] = str(options["cache_dir"])
    options["trust_remote_code"] = False
    if options.get("revision") is None:
        options.pop("revision", None)
    return TypeAdapter(CheckpointOptions).validate_python(options, strict=True)


def _config_object(value: JSONValue, field: str) -> JSONObject:
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint {field} must be a JSON object")
    return value


def _native_quantization_options(
    value: Mapping[str, JSONValue] | QuantizationConfig | QuantizationOptions,
) -> QuantizationOptions:
    """Select native MLX options, keeping checkpoint metadata out of nn.quantize."""
    result: QuantizationOptions = {}
    for name in ("group_size", "bits"):
        entry = value.get(name)
        if entry is not None:
            if not isinstance(entry, int) or isinstance(entry, bool):
                raise ValueError(f"Quantization {name} must be an integer")
            if name == "group_size":
                result["group_size"] = entry
            else:
                result["bits"] = entry
    mode = value.get("mode")
    if mode is not None:
        if not isinstance(mode, str):
            raise ValueError("Quantization mode must be a string")
        result["mode"] = mode
    quantize_input = value.get("quantize_input")
    if quantize_input is not None:
        if not isinstance(quantize_input, bool):
            raise ValueError("quantize_input must be boolean")
        result["quantize_input"] = quantize_input
    return result


def load_model(
    model_path: Path,
    lazy: bool = False,
    **kwargs: object,
) -> MedicalModel:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
        revision (str, optional): A revision id which can be a branch name,
            a tag, or a commit hash. Default: ``None``.
        quantize_activations (bool, optional): If True, convert QuantizedLinear layers
            to QQLinear layers for activation quantization. Only supported for models
            quantized with 'nvfp4' or 'mxfp8' modes. Default: ``False``.

    Returns:
        nn.Module: The loaded and initialized model.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be instantiated.
    """
    options = _native_checkpoint_options(kwargs)
    strict = options.pop("strict", True)
    if (model_path / "offload_index.json").exists():
        raise ValueError("Expert-offload checkpoints are outside the medical runtime")
    config = load_config(model_path, **options)
    get_model_and_args(config=config, model_path=model_path)

    index_file = model_path / "model.safetensors.index.json"
    weight_files = []
    declared_keys: set[str] = set()
    if index_file.exists():
        try:
            with open(index_file) as f:
                index: object = json.load(f)
            if not is_json_object(index):
                raise ValueError("Safetensors index must contain a JSON object")
            weight_map = index.get("weight_map", {})
            if not isinstance(weight_map, dict) or not all(
                isinstance(value, str) for value in weight_map.values()
            ):
                raise ValueError(
                    "Safetensors weight_map must map tensor names to shard filenames"
                )
            declared_keys = set(weight_map)
            weight_files = [
                str(model_path / shard)
                for shard in sorted(
                    {value for value in weight_map.values() if isinstance(value, str)}
                )
                if (model_path / shard).exists()
            ]
        except (ValueError, OSError):
            weight_files = []
            declared_keys = set()
    if not weight_files:
        weight_files = [
            wf
            for wf in glob.glob(str(model_path / "*.safetensors"))
            if not wf.endswith("consolidated.safetensors")
        ]

    if not weight_files:
        logging.error(f"No safetensors found in {model_path}")
        raise FileNotFoundError(
            f"No safetensors found in {model_path}; provide a model directory "
            "containing safetensors weights."
        )

    weights: dict[str, mx.array] = {}
    for wf in weight_files:
        weights.update(_load_safetensors(wf))

    has_quantization = "quantization" in config
    gemma_config = apply_generation_config_defaults(
        GemmaConfig.from_dict(config), config
    )
    model = GemmaModel(gemma_config)

    # Only MLX-native ``quantization`` metadata (written by mlx_vlm.convert) is
    # loaded. Transformers-side quantized exports (FP8, ModelOpt, compressed
    # tensors, AWQ/GPTQ) store layouts MLX cannot load directly; MedGemma is
    # published unquantized, so convert that release instead.
    if not has_quantization:
        text_config = _config_object(config.get("text_config", {}), "text_config")
        if (
            "quantization_config" in config
            or "quantization_config" in text_config
            or (model_path / "hf_quant_config.json").exists()
        ):
            raise ValueError(
                "Pre-quantized Transformers checkpoints are not supported; load an "
                "MLX checkpoint or convert the unquantized release with "
                "mlx_vlm.convert --quantize"
            )

    weights = model.vision_tower.sanitize(weights)
    weights = model.language_model.sanitize(weights)

    if (quantization := config.get("quantization", None)) is not None:
        quantization = TypeAdapter(QuantizationConfig).validate_python(quantization)
        quantized_model = model

        def get_class_predicate(p: str, m: nn.Module) -> bool | QuantizationOptions:
            per_module_quantization = _quantization_for_module_path(
                quantization, p, model
            )
            # Handle custom per-layer quantization, including aliases supplied
            # by the model.
            if per_module_quantization is not None:
                return _native_quantization_options(per_module_quantization)
            if not hasattr(m, "to_quantized"):
                return False
            # Skip layers not divisible by 64
            weight: object = getattr(m, "weight", None)
            if isinstance(weight, mx.array) and weight.size % 64 != 0:
                return False
            # Handle legacy models which may not have everything quantized
            return f"{p}.scales" in weights

        native_options = _native_quantization_options(quantization)
        nn.quantize(
            quantized_model,
            group_size=native_options.get("group_size"),
            bits=native_options.get("bits"),
            mode=native_options.get("mode", "affine"),
            class_predicate=get_class_predicate,
        )

    if options.get("quantize_activations", False):
        if quantization is None:
            raise ValueError(
                "Activation quantization requires the model to be quantized first. "
                "Please use a quantized model with mode 'nvfp4' or 'mxfp8'."
            )
        model = quantize_activations(model)

    _drop_modules_without_weights(model, weights, declared_keys)

    model.load_weights(list(weights.items()), strict=strict)

    if not lazy:
        evaluate(model.parameters())

    setattr(model, "model_path", model_path)
    model.eval()
    return model


class _ReadSafetensors(Protocol):
    def __call__(
        self, file: str, /, *, format: Literal["safetensors"]
    ) -> dict[str, mx.array]: ...


def _load_safetensors(path: str) -> dict[str, mx.array]:
    return cast(_ReadSafetensors, getattr(mx, "load"))(path, format="safetensors")


def load(
    path_or_hf_repo: str,
    lazy: bool = False,
    revision: Optional[str] = None,
    strict: bool = True,
    **kwargs: object,
) -> Tuple[nn.Module, ProcessorMixin]:
    """
    Load the model and tokenizer from a given path or a huggingface repository.

    Args:
        path_or_hf_repo (str): The path or the huggingface repository to load the model from.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
        revision (str, optional): A revision id which can be a branch name,
            a tag, or a commit hash. Default: ``None``.
        strict (bool): Whether or not to raise an exception if weights don't
            match. Default: ``True``.
        **kwargs: Checkpoint options: ``cache_dir``, ``force_download``,
            ``local_files_only``, ``token``, and ``quantize_activations`` (convert
            QuantizedLinear layers to QQLinear layers for activation quantization;
            only supported for models quantized with 'nvfp4' or 'mxfp8' modes).
            ``trust_remote_code`` must be absent or false.

    Returns:
        Tuple[nn.Module, ProcessorMixin]: The loaded model and its native
        processor (Gemma3Processor).

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If the model type is not ``gemma3``, the
            checkpoint requests Python code (``model_file`` or
            ``trust_remote_code``), or an option is malformed.
    """
    options = _native_checkpoint_options(kwargs)
    force_download = options.get("force_download", False)
    model_path = get_model_path(
        path_or_hf_repo,
        force_download=force_download,
        revision=revision,
        cache_dir=options.get("cache_dir"),
        local_files_only=options.get("local_files_only", False),
        token=options.get("token"),
    )
    model = load_model(model_path, lazy, **{**options, "strict": strict})

    # Get the eos_token_id from the model config
    eos_token_id = model.config.eos_token_id

    processor = load_processor(model_path, True, eos_token_ids=eos_token_id, **options)

    return model, processor


def load_config(model_path: Union[str, Path], **kwargs: object) -> JSONObject:
    """Load model configuration from a path or Hugging Face repo.

    Args:
        model_path: Local path or Hugging Face repo ID to load config from
        **kwargs: Additional keyword arguments to pass to the config loader

    Returns:
        dict: Model configuration

    Raises:
        FileNotFoundError: If config.json is not found at the path
    """
    options = _native_checkpoint_options(kwargs)
    if isinstance(model_path, str):
        model_path = get_model_path(
            model_path,
            revision=options.get("revision"),
            force_download=options.get("force_download", False),
            cache_dir=options.get("cache_dir"),
            local_files_only=options.get("local_files_only", False),
            token=options.get("token"),
        )

    try:
        with open(model_path / "config.json", encoding="utf-8") as f:
            config: object = json.load(f)
        if not is_json_object(config):
            raise ValueError("config.json must contain a JSON object")

        generation_config_file = model_path / "generation_config.json"
        if generation_config_file.exists():
            with open(generation_config_file, encoding="utf-8") as f:
                generation_config: object = json.load(f)
            if not is_json_object(generation_config):
                raise ValueError("generation_config.json must contain a JSON object")
            _merge_generation_config(config, generation_config)

    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config not found at {model_path}") from exc

    return config


def load_processor(
    model_path: str | Path,
    add_detokenizer: bool = True,
    eos_token_ids: int | list[int] | None = None,
    **kwargs: object,
) -> ProcessorMixin:
    """Load the native medical processor without altering Transformers globally.

    Resolve the architecture before creating its tokenizer. Checkpoint Python
    code is disabled for the native processor and its tokenizer/image loaders.
    """
    options = _native_checkpoint_options(kwargs)
    if isinstance(model_path, str):
        model_path = get_model_path(
            model_path,
            revision=options.get("revision"),
            force_download=options.get("force_download", False),
            cache_dir=options.get("cache_dir"),
            local_files_only=options.get("local_files_only", False),
            token=options.get("token"),
        )
    get_model_and_args(load_config(model_path), model_path)
    processor = Gemma3Processor.from_pretrained(model_path, **options)
    if add_detokenizer:
        detokenizer_class = load_detokenizer_factory(model_path)

        # Get the tokenizer object
        tokenizer_obj = processor.tokenizer

        setattr(processor, "detokenizer", detokenizer_class(tokenizer_obj))

        # Create and assign the StoppingCriteria
        criteria = StoppingCriteria(
            eos_token_ids,
            tokenizer_obj,
            additional_eos_token_ids=getattr(processor, "additional_eos_token_ids", ()),
        )
        setattr(processor.tokenizer, "stopping_criteria", criteria)

    return processor


def fetch_from_hub(
    model_path: Path, lazy: bool = False, **kwargs: object
) -> Tuple[nn.Module, JSONObject, ProcessorMixin]:
    """Return the model, config and processor for a local checkpoint."""
    model = load_model(model_path, lazy, **kwargs)
    config = load_config(model_path, **kwargs)
    processor = load_processor(
        model_path,
        add_detokenizer=False,
        eos_token_ids=resolve_eos_token_ids(config.get("eos_token_id"), model),
        **kwargs,
    )
    return model, config, processor


def make_shards(
    weights: dict[str, mx.array], max_file_size_gb: int = MAX_FILE_SIZE_GB
) -> list[dict[str, mx.array]]:
    """
    Splits the weights into smaller shards.

    Args:
        weights (dict): Model weights.
        max_file_size_gb (int): Maximum size of each shard in gigabytes.

    Returns:
        list: List of weight shards.
    """
    max_file_size_bytes = max_file_size_gb << 30
    shards: list[dict[str, mx.array]] = []
    shard: dict[str, mx.array] = {}
    shard_size = 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)
    return shards


def create_model_card(
    path: Union[str, Path], hf_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Create model card for a converted MLX model.

    Args:
        path (Union[str, Path]): Local path to the converted model.
        hf_path (Optional[Union[str, Path]]): Original Hugging Face repo id or local path used for conversion.
    """
    from huggingface_hub import ModelCard, ModelCardData

    if hf_path is None:
        card = cast(
            Callable[[ModelCardData], ModelCard], getattr(ModelCard, "from_template")
        )(ModelCardData(language="en"))
    else:
        card = ModelCard.load(hf_path)
    if not isinstance(card.data, ModelCardData):
        raise TypeError("Model cards require ModelCardData metadata")
    card.data.library_name = "mlx"
    if card.data.pipeline_tag is None:
        card.data.pipeline_tag = "image-text-to-text"
    if card.data.tags is None:
        card.data.tags = ["mlx"]
    elif "mlx" not in card.data.tags:
        card.data.tags += ["mlx"]
    if hf_path is not None:
        card.data.base_model = str(hf_path)
    card.text = ""
    card.save(Path(path) / "README.md")


def upload_to_hub(path: str, upload_repo: str) -> None:
    """
    Uploads the model to Hugging Face hub.

    Args:
        path (str): Local path to the model.
        upload_repo (str): Name of the HF repo to upload to.
    """
    from huggingface_hub import HfApi, ModelCard, ModelCardData, logging

    from mlx_vlm import __version__

    logging.set_verbosity_info()
    card_path = Path(path) / "README.md"
    card = ModelCard.load(card_path)

    if not isinstance(card.data, ModelCardData):
        raise TypeError("Model cards require ModelCardData metadata")
    hf_path = card.data.base_model

    if hf_path is not None:
        provenance = f"""
        This model was converted to MLX format from [`{hf_path}`](https://huggingface.co/{hf_path})
        using med-vlm-mlx version **{__version__}**, MedMLX's fork of [MLX-VLM](https://github.com/Blaizzy/mlx-vlm).
        Refer to the [original model card](https://huggingface.co/{hf_path}) for more details on the model.
        """
    else:
        provenance = ""

    card.text = dedent(f"""
        # {upload_repo}
        {provenance}
        ## Use with Med-VLM-MLX

        On Apple Silicon with Python 3.13 or 3.14:

        ```bash
        python3.13 -m venv .venv
        source .venv/bin/activate
        python -m pip install "med-vlm-mlx @ git+https://github.com/MedMLX/med-vlm-mlx.git"
        ```

        This inference-only fork retains the `mlx_vlm` namespace from upstream
        [MLX-VLM](https://github.com/Blaizzy/mlx-vlm). Install upstream in a
        different environment because the distributions share modules.

        ```bash
        python -m mlx_vlm.generate --model {upload_repo} --max-tokens 100 --temperature 0.0 --prompt "Describe this image." --image <path_to_image>
        ```
        """)
    card.save(card_path)

    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_folder(
        folder_path=path,
        repo_id=upload_repo,
        repo_type="model",
    )
    print(f"Upload successful, go to https://huggingface.co/{upload_repo} for details.")


def save_weights(
    save_path: Union[str, Path],
    model: nn.Module,
    *,
    donate_weights: bool = False,
) -> None:
    """Save model weights into specified directory."""
    if isinstance(save_path, str):
        save_path = Path(save_path)

    weights = dict(tree_flatten(model.parameters()))

    save_path.mkdir(parents=True, exist_ok=True)

    shards: list[dict[str, mx.array] | None] = list(make_shards(weights))
    shards_count = len(shards)
    shard_file_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "model.safetensors"
    )

    total_size = sum(v.nbytes for v in weights.values())
    weight_map: JSONObject = {}
    index_data: JSONObject = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }

    # Write the weights and make sure no references are kept other than the
    # necessary ones
    if donate_weights:
        model.load_weights(
            [(name, mx.array([])) for name, _ in tree_flatten(model.parameters())],
            strict=False,
        )

    weights.clear()
    del weights

    for i in range(len(shards)):
        shard = shards[i]
        assert shard is not None
        shards[i] = None
        shard_name = shard_file_format.format(i + 1, shards_count)
        shard_path = save_path / shard_name

        save_safetensors(shard_path, shard, metadata={"format": "mlx"})

        for weight_name in shard.keys():
            weight_map[weight_name] = shard_name
        del shard

    index_data["weight_map"] = {k: weight_map[k] for k in sorted(weight_map)}

    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(
            index_data,
            f,
            indent=4,
        )


def save_config(
    config: JSONObject,
    config_path: Union[str, Path],
) -> None:
    """Save the model configuration to the ``config_path``.

    The final configuration will be sorted before saving for better readability.

    Args:
        config (dict): The model configuration.
        config_path (Union[str, Path]): Model configuration file path.
    """
    # Clean unused keys
    config.pop("_name_or_path", None)
    config.pop("torch_dtype", None)

    # sort the config for better readability
    config = dict(sorted(config.items()))

    # write the updated config to the config_path (if provided)
    with open(config_path, "w") as fid:
        json.dump(config, fid, indent=4)


def load_image(
    image_source: Union[str, Path, BytesIO, Image.Image], timeout: int = 10
) -> Image.Image:
    """
    Helper function to load an image from either a URL, file path, data URI,
    or BytesIO object.
    """
    import base64

    original_source = image_source
    try:
        if isinstance(image_source, Image.Image):
            image = image_source
        else:
            if isinstance(image_source, str) and image_source.startswith("data:image/"):
                if "," not in image_source:
                    raise ValueError(
                        "Invalid data URI format - missing comma separator"
                    )
                _, data = image_source.split(",", 1)
                image_source = BytesIO(base64.b64decode(data))
            if isinstance(image_source, str) and image_source.startswith(
                ("http://", "https://")
            ):
                with requests.get(
                    image_source, stream=True, timeout=timeout
                ) as response:
                    response.raise_for_status()
                    image_source = BytesIO(response.content)

            image = Image.open(image_source)
    except ValueError:
        raise
    except (OSError, requests.RequestException) as e:
        raise ValueError(f"Failed to load image from {original_source}: {e}") from e

    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def resize_image(img: Image.Image, max_size: tuple[int, int]) -> Image.Image:
    """Scale ``img`` to fit within ``max_size`` ``(width, height)``, keeping aspect.

    The scale factor is not capped at 1, so smaller images are enlarged. Sizes are
    truncated to integers and the default PIL resampling filter is used.
    """
    ratio = min(max_size[0] / img.width, max_size[1] / img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    return img.resize(new_size)


def process_image(
    img: str | Path | Image.Image,
    resize_shape: tuple[int, int] | None,
    image_processor: object,
) -> Image.Image:
    """Load an RGB image and optionally fit it to ``resize_shape`` ``(width, height)``.

    ``image_processor`` is accepted for call compatibility and not used.
    """
    image = load_image(img)
    return resize_image(image, resize_shape) if resize_shape is not None else image


type MedicalProcessor = Gemma3Processor


def _processor_kwargs(options: Mapping[str, object]) -> ProcessingKwargs:
    """Preserve flat native options and reject unsupported nested HF groups."""
    validated = require_processed_inputs(dict(options))
    for key in ("text_kwargs", "images_kwargs", "videos_kwargs", "audio_kwargs"):
        if key in validated:
            raise ValueError(
                f"{key} is unsupported; pass medical processor options as flat keywords"
            )
    # ProcessingKwargs is open: its four declared nested keys are absent, and
    # the native processors validate all remaining flat values at their ingress.
    return cast(ProcessingKwargs, validated)


def process_inputs(
    processor: MedicalProcessor,
    prompts: str | list[str] | None,
    images: object = None,
    add_special_tokens: bool = False,
    padding: bool = True,
    padding_side: str = "left",
    return_tensors: str = "mlx",
    **kwargs: object,
) -> ProcessedInputs:
    """Call a retained native processor without filtering its flat options."""
    if images is not None and not is_image_batch(images):
        raise TypeError("Expected image paths, PIL images or numeric tensors")
    options = _processor_kwargs(
        {
            **kwargs,
            "padding": padding,
            "padding_side": padding_side,
            "add_special_tokens": add_special_tokens,
            "return_tensors": return_tensors,
        }
    )
    result = processor(text=prompts, images=images, **options)
    return require_processed_inputs(getattr(result, "data", result))


def _uniform_images(
    images: list[Image.Image], processor: MedicalProcessor
) -> list[Image.Image]:
    """Apply explicitly requested batch resizing, or white padding without a size."""
    size: object = getattr(processor.image_processor, "size", None)
    target: tuple[int, int] | None = None
    if isinstance(size, int):
        target = (size, size)
    elif isinstance(size, tuple):
        dimensions = cast(tuple[object, ...], size)
        if (
            len(dimensions) == 2
            and isinstance(dimensions[0], int)
            and isinstance(dimensions[1], int)
        ):
            target = (dimensions[1], dimensions[0])
    elif is_json_object(size):
        target = (
            parse_int(
                size.get("width", 384),
            ),
            parse_int(
                size.get("height", 384),
            ),
        )
    if target is not None:
        return [image.resize(target, Image.Resampling.BICUBIC) for image in images]
    width, height = max(image.width for image in images), max(
        image.height for image in images
    )
    padded: list[Image.Image] = []
    for image in images:
        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
        padded.append(canvas)
    return padded


def prepare_inputs(
    processor: object,
    images: ImageSource | list[ImageSource] | list[str] | None = None,
    prompts: str | list[str] | None = None,
    image_token_index: int | None = None,
    resize_shape: tuple[int, int] | None = None,
    add_special_tokens: bool = False,
    padding: bool = True,
    padding_side: str = "left",
    pad_to_uniform_size: bool = False,
    return_tensors: str = "mlx",
    **kwargs: object,
) -> ProcessedInputs:
    """Prepare native medical text and images as MLX model inputs.

    Prompts are tokenized with left padding by default, so batched generation
    continues from the last column; the pad token falls back to the EOS token.
    Images given as paths, URLs or PIL images are loaded as RGB. ``resize_shape``
    fits each to ``(width, height)``; ``pad_to_uniform_size`` instead resizes
    PIL images to the processor's configured size, or white-pads them to a common
    canvas when it has none. Remaining keyword options are passed to the
    processor unchanged.

    Args:
        processor: A ``Gemma3Processor``; a bare tokenizer is accepted for text-only input.
        images: Image sources; tensors cannot be combined with resizing options.
        prompts: One prompt or a list of prompts.
        image_token_index: Unused; accepted for call compatibility.
        resize_shape: Optional ``(width, height)`` bound for each image.
        pad_to_uniform_size: Give all PIL images one shape (see above).

    Returns:
        Processor outputs as MLX arrays: ``input_ids`` and ``attention_mask`` and,
        with media, ``pixel_values``.

    Raises:
        ValueError: If media is passed with a bare tokenizer, or
            an option is invalid.
        TypeError: If the processor type or a media type is unsupported.
    """
    if not isinstance(processor, (Gemma3Processor, PreTrainedTokenizerBase)):
        raise TypeError(
            "Expected a native medical processor or a Transformers tokenizer"
        )
    has_images = images is not None and (not isinstance(images, list) or bool(images))
    tokenizer = (
        processor
        if isinstance(processor, PreTrainedTokenizerBase)
        else processor.tokenizer
    )
    if padding and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not has_images:
        return to_mlx(
            tokenize_inputs(
                tokenizer,
                prompts,
                require_processed_inputs(
                    {
                        **kwargs,
                        "padding": padding,
                        "padding_side": padding_side,
                        "add_special_tokens": add_special_tokens,
                        "return_tensors": return_tensors,
                    }
                ),
            )
        )
    if isinstance(processor, PreTrainedTokenizerBase):
        raise ValueError("Image inputs require a native medical processor")
    prepared_images: list[ImageSource] | None = None
    if has_images and images is not None:
        sources = images if isinstance(images, list) else [images]
        prepared_images = []
        for image in sources:
            if isinstance(image, (str, Path, Image.Image)):
                prepared_images.append(
                    process_image(image, resize_shape, processor.image_processor)
                )
            else:
                if resize_shape is not None or pad_to_uniform_size:
                    raise ValueError(
                        "Explicit image resizing/padding requires PIL images or paths"
                    )
                prepared_images.append(image)
        if pad_to_uniform_size and len(prepared_images) > 1:
            pil_images = [
                image for image in prepared_images if isinstance(image, Image.Image)
            ]
            prepared_images = list(_uniform_images(pil_images, processor))
    extra = dict(kwargs)
    result = process_inputs(
        processor,
        prompts,
        images=prepared_images,
        add_special_tokens=add_special_tokens,
        padding=padding,
        padding_side=padding_side,
        return_tensors=return_tensors,
        **extra,
    )
    if "images" in result:
        result["pixel_values"] = result.pop("images")
    return to_mlx(result)


type ImageShape = tuple[int, int] | tuple[int, int, int]


def group_images_by_shape(
    images: list[Image.Image], disable_grouping: bool = False
) -> tuple[dict[ImageShape, list[Image.Image]], dict[ImageShape, list[int]]]:
    """Group equal-size images, or include their index to disable grouping."""
    grouped_images: dict[ImageShape, list[Image.Image]] = {}
    grouped_indices: dict[ImageShape, list[int]] = {}
    for index, image in enumerate(images):
        shape: ImageShape = (
            (image.height, image.width, index)
            if disable_grouping
            else (image.height, image.width)
        )
        grouped_images.setdefault(shape, []).append(image)
        grouped_indices.setdefault(shape, []).append(index)
    return grouped_images, grouped_indices


def resolve_eos_token_ids(eos_token_ids: object, tokenizer: object) -> List[int]:
    """Union configured EOS token ids with the tokenizer's own EOS.

    A checkpoint's ``eos_token_id`` can disagree with the token its chat template
    ends turns on -- Chandra OCR 2 configures ``<|endoftext|>`` but emits
    ``<|im_end|>`` -- so neither source alone is enough to stop generation.
    """
    resolved: List[int] = []
    sources: tuple[object, ...] = (
        eos_token_ids,
        getattr(tokenizer, "eos_token_ids", None),
        getattr(tokenizer, "eos_token_id", None),
    )
    for source in sources:
        if isinstance(source, int):
            source = [source]
        if not isinstance(source, (list, tuple, set)):
            continue
        for token_id in cast(Iterable[object], source):
            if isinstance(token_id, int) and token_id not in resolved:
                resolved.append(token_id)
    return resolved


class StoppingCriteria:
    """Decide when generation reaches a configured end-of-sequence token."""

    def __init__(
        self,
        eos_token_ids: int | Sequence[int] | set[int] | None,
        tokenizer: TokenEncoder | None = None,
        additional_eos_token_ids: Sequence[int] | None = None,
    ) -> None:
        """Resolve initial EOS ids from the tokenizer and additional ids."""
        self.tokenizer = tokenizer
        self.additional_eos_token_ids = list(
            dict.fromkeys(additional_eos_token_ids or ())
        )
        self.reset(eos_token_ids)

    def add_eos_token_ids(
        self,
        new_eos_token_ids: (
            int | str | Sequence[int | str] | AbstractSet[int | str] | None
        ) = None,
    ) -> None:
        """
        Add new token IDs to the list of EOS token IDs.

        Args:
            new_eos_token_ids: Integer, string, or list of integers/strings representing token IDs to add.
                               If strings are provided, they will be converted to integers if possible.
        """
        if new_eos_token_ids is None:
            return

        if self.tokenizer is None:
            raise ValueError("Processor is not provided")

        tokens = (
            [new_eos_token_ids]
            if isinstance(new_eos_token_ids, (str, int))
            else new_eos_token_ids
        )
        resolved: list[int] = []
        for token in tokens:
            if isinstance(token, int):
                resolved.append(token)
            else:
                encoded = self.tokenizer.encode(" " + token, add_special_tokens=False)
                if not encoded:
                    raise ValueError(f"EOS marker {token!r} encoded to no tokens")
                resolved.append(encoded[-1])
        self.eos_token_ids.extend(resolved)

    def reset(
        self, eos_token_ids: int | Sequence[int] | set[int] | None = None
    ) -> None:
        """Rebuild the EOS id list from the tokenizer and additional ids."""
        resolved = resolve_eos_token_ids(eos_token_ids, self.tokenizer)
        resolved.extend(
            token_id
            for token_id in self.additional_eos_token_ids
            if token_id not in resolved
        )
        if getattr(self, "eos_token_ids", None) != resolved:
            self.eos_token_ids = resolved

    def __call__(self, input_ids: int | mx.array) -> bool:
        """Report whether a sampled token ends the sequence."""
        return input_ids in self.eos_token_ids


class ThinkingBudgetCriteria:
    """
    Enforces a budget on thinking tokens.

    Tracks tokens within thinking blocks (between start and end tokens) and
    forces a closing sequence (e.g. ``\\n</think>``) when budget is exceeded.
    """

    def __init__(
        self,
        tokenizer: TokenEncoder,
        thinking_budget: int,
        thinking_end_token: str = "</think>",
        thinking_start_token: Optional[str] = None,
        enable_thinking: bool = False,
        prompt_preopens_thinking: bool = False,
    ) -> None:
        """Encode the thinking markers and prepare the forced closing sequence.

        ``thinking_start_token`` is required; markers that encode to no tokens
        raise ``ValueError``.
        """
        self.tokenizer = tokenizer
        self.thinking_budget = thinking_budget
        self.enable_thinking = enable_thinking
        self.prompt_preopens_thinking = prompt_preopens_thinking

        if not thinking_start_token:
            raise ValueError(
                "Thinking budget requires an explicit thinking_start_token"
            )
        start_ids = tokenizer.encode(thinking_start_token, add_special_tokens=False)
        end_ids = tokenizer.encode(thinking_end_token, add_special_tokens=False)
        if not start_ids or not end_ids:
            raise ValueError("Thinking markers must each encode to at least one token")
        self.thinking_start_token_id = start_ids[-1]
        self.thinking_end_token_id = end_ids[-1]

        self._forced_sequence: List[int] = []
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if newline_ids:
            self._forced_sequence.append(newline_ids[-1])
        self._forced_sequence.append(self.thinking_end_token_id)
        self._forced_index = 0

        self.in_thinking = self.enable_thinking and self.prompt_preopens_thinking
        self.thinking_token_count = 0
        self.budget_exceeded = False
        self.forced_token_id: int | None = None

    def __call__(self, token_id: int) -> Optional[int]:
        """Process a token and return a forced token ID if budget exceeded, else None."""
        if self.enable_thinking and token_id == self.thinking_start_token_id:
            self.in_thinking = True
            return None

        if token_id == self.thinking_end_token_id:
            self.in_thinking = False
            self.budget_exceeded = False
            self._forced_index = 0
            return None

        if self.in_thinking:
            self.thinking_token_count += 1
            if self.thinking_token_count > self.thinking_budget:
                self.budget_exceeded = True

        if self.budget_exceeded and self._forced_index < len(self._forced_sequence):
            forced = self._forced_sequence[self._forced_index]
            self._forced_index += 1
            self.forced_token_id = forced
            return forced

        self.forced_token_id = None
        return None

    def pop_forced_token_id(self) -> Optional[int]:
        """Return and clear the pending forced token ID, if any."""
        if self.forced_token_id is None or not self.enable_thinking:
            return None

        forced_token_id = self.forced_token_id
        self.forced_token_id = None
        return forced_token_id


def should_add_special_tokens(model_type: str, processor: object) -> bool:
    """Return whether tokenization should add markers outside the chat template."""
    template_owns_markers = {
        "gemma3",
        "gemma3n",
        "gemma4",
        "gemma4_unified",
        "laguna",
    }
    if model_type not in template_owns_markers:
        return True
    return getattr(processor, "chat_template", None) is None
