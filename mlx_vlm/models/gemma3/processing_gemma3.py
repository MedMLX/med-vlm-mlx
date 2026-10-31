"""
Processor class for Gemma3.

Adapted from HuggingFace Transformers:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/processing_gemma3.py
"""

import re
from os import PathLike
from pathlib import Path
from typing import Protocol, Unpack, cast

import numpy as np
from transformers.audio_utils import AudioInput
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_base import ImageProcessingMixin
from transformers.image_utils import ImageInput, make_nested_list_of_images
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers.tokenization_utils_base import (
    PreTokenizedInput,
    PreTrainedTokenizerBase,
    TextInput,
)
from transformers.video_utils import VideoInput

from mlx_vlm.contracts import (
    ImageSource,
    InputTensor,
    JSONValue,
    ProcessedInputs,
    ProcessorValue,
    is_int_list,
    is_json_object,
    is_str_list,
    require_processed_inputs,
)
from mlx_vlm.models.base import load_chat_template, to_mlx
from mlx_vlm.models.media_types import ImageBatch, is_image_batch
from mlx_vlm.models.processing import (
    BatchDecodeTokens,
    DecodeIds,
    DecodeTokens,
    load_tokenizer,
    tokenize_inputs,
)


class _InitializeGemma(Protocol):
    def __call__(
        self,
        *,
        image_processor: ImageProcessingMixin,
        tokenizer: PreTrainedTokenizerBase,
        chat_template: str | dict[str, str] | None,
        **kwargs: ProcessorValue,
    ) -> None: ...


class _FetchImages(Protocol):
    def __call__(self, images: ImageBatch) -> object: ...


class _NestedImages(Protocol):
    def __call__(self, images: ImageBatch) -> list[list[ImageSource | InputTensor]]: ...


class _ProcessImages(Protocol):
    def __call__(self, images: ImageBatch, **kwargs: ProcessorValue) -> object: ...


class _LoadImageProcessor(Protocol):
    def __call__(
        self, path: str | Path, **kwargs: ProcessorValue
    ) -> ImageProcessingMixin: ...


class Gemma3Processor(ProcessorMixin):
    """Tokenize Gemma3 prompts and preprocess their images (text and images only).

    When images are supplied, each image placeholder (the tokenizer's
    ``boi_token``) is expanded to ``boi_token`` + ``image_seq_length`` copies
    of the image token + ``eoi_token``, surrounded by blank lines. Pan-and-scan
    crops add their own expanded image sequences. Without images, text is
    tokenized without expanding or validating image placeholders. Video and
    audio are rejected.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "image_seq_length"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    image_processor: ImageProcessingMixin
    tokenizer: PreTrainedTokenizerBase

    def __init__(
        self,
        image_processor: ImageProcessingMixin,
        tokenizer: PreTrainedTokenizerBase,
        chat_template: str | dict[str, str] | None = None,
        image_seq_length: int = 256,
        **kwargs: ProcessorValue,
    ) -> None:
        """Build the processor after validating the tokenizer's image tokens."""
        self.image_seq_length = image_seq_length
        image_id: object = getattr(tokenizer, "image_token_id", None)
        boi: object = getattr(tokenizer, "boi_token", None)
        image_token: object = getattr(tokenizer, "image_token", None)
        eoi: object = getattr(tokenizer, "eoi_token", None)
        if not isinstance(image_id, int) or not all(
            isinstance(value, str) for value in (boi, image_token, eoi)
        ):
            raise ValueError(
                "Gemma tokenizer requires image_token_id and image boundary tokens"
            )
        if (
            not isinstance(boi, str)
            or not isinstance(image_token, str)
            or not isinstance(eoi, str)
        ):
            raise TypeError("Gemma image boundary tokens must be strings")
        self.image_token_id = image_id
        self.boi_token = boi
        self.image_token = image_token
        image_tokens_expanded = "".join([image_token] * image_seq_length)
        self.full_image_sequence = f"\n\n{boi}{image_tokens_expanded}{eoi}\n\n"

        cast(_InitializeGemma, getattr(super(), "__init__"))(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            **kwargs,
        )

    def __call__(
        self,
        images: ImageInput | ImageBatch | None = None,
        text: (
            TextInput
            | PreTokenizedInput
            | list[TextInput]
            | list[PreTokenizedInput]
            | None
        ) = None,
        videos: VideoInput | None = None,
        audio: AudioInput | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ) -> BatchFeature:
        """Expand image placeholders, tokenize ``text`` and preprocess ``images``.

        Args:
            images: Image paths, URLs, PIL images or tensors, one inner list per
                prompt (or a flat list for a single prompt).
            text: Prompt string(s) containing one ``boi_token`` per image. When
                omitted, each prompt consists of one ``boi_token`` per image.
            videos: Must be ``None``.
            audio: Must be ``None``.
            **kwargs: Flat options. ``do_convert_rgb`` and the ``pan_and_scan_*``
                options go to the image processor; ``return_tensors`` is
                dropped; ``return_mm_token_type_ids`` adds ``token_type_ids``
                (1 at image tokens, 0 elsewhere); everything else goes to the
                tokenizer.

        Returns:
            A ``BatchFeature`` with tokenizer outputs (``input_ids``,
            ``attention_mask``, ...) and, with images, ``pixel_values`` from
            the Transformers image processor (one entry per image or crop).
            Convertible array-like values become MLX arrays; non-array values
            and ragged lists may remain in their original form.

        Image fetching and preprocessing happen before prompt batch and
        placeholder counts are checked. A mismatch can therefore fail after
        the image processor has already run.

        Raises:
            ValueError: If video or audio is given, neither ``text`` nor
                ``images`` is given, or supplied images and prompt placeholders
                differ in count.
            TypeError: If ``text`` or ``images`` have an unsupported type.
        """
        if videos is not None or audio is not None:
            raise ValueError(
                "The medical Gemma processor supports text and images only"
            )
        if images is not None and not is_image_batch(images):
            raise TypeError("Expected image paths, PIL images or image tensors")
        options = require_processed_inputs(kwargs)
        if text is None and images is None:
            raise ValueError("Provide at least one of `text` or `images`.")

        if isinstance(text, str):
            text = [text]

        if text is not None and not all(isinstance(item, str) for item in text):
            raise TypeError("Gemma processing requires strings with image placeholders")
        prompts = [item for item in (text or []) if isinstance(item, str)]
        image_inputs: ProcessedInputs = {}
        if images is not None:
            fetched = cast(_FetchImages, getattr(self.image_processor, "fetch_images"))(
                images
            )
            if not is_image_batch(fetched):
                raise TypeError("Image processor returned unsupported decoded images")
            images = fetched
            batched_images = cast(_NestedImages, make_nested_list_of_images)(images)

            # Image-processor options leave ``options``; the rest go to the tokenizer.
            images_kwargs: dict[str, ProcessorValue] = {}
            for k in list(options.keys()):
                if k in (
                    "do_convert_rgb",
                    "do_pan_and_scan",
                    "pan_and_scan_min_crop_size",
                    "pan_and_scan_max_num_crops",
                    "pan_and_scan_min_ratio_to_activate",
                ):
                    images_kwargs[k] = options.pop(k)

            image_inputs = require_processed_inputs(
                getattr(
                    cast(_ProcessImages, self.image_processor)(images, **images_kwargs),
                    "data",
                )
            )

            # Create empty text to be replaced with placeholders
            if not prompts:
                prompts = [
                    " ".join([self.boi_token] * len(imgs)) for imgs in batched_images
                ]

            if len(batched_images) != len(prompts):
                raise ValueError(
                    f"Received inconsistently sized batches of images ({len(batched_images)}) and text ({len(prompts)})."
                )

            # Replace image tokens by the full expanded sequence
            num_crops = image_inputs.pop("num_crops", None)
            if num_crops is not None:
                if hasattr(num_crops, "tolist"):
                    num_crops = getattr(num_crops, "tolist")()
                if not is_int_list(num_crops):
                    raise TypeError("Gemma num_crops must contain integer crop counts")
            if num_crops:
                crop_iter = iter(num_crops)
                batch_num_crops = [
                    [next(crop_iter) for _ in imgs] for imgs in batched_images
                ]
            else:
                batch_num_crops = [[0] * len(imgs) for imgs in batched_images]

            for batch_idx, (prompt, imgs, crops) in enumerate(
                zip(prompts, batched_images, batch_num_crops)
            ):
                image_indexes = [
                    m.start() for m in re.finditer(re.escape(self.boi_token), prompt)
                ]

                if len(imgs) != len(image_indexes):
                    raise ValueError(
                        f"Prompt contained {len(image_indexes)} image tokens but received {len(imgs)} images."
                    )

                # Insert additional image tokens for Pan-and-Scan crops
                for num, idx in reversed(list(zip(crops, image_indexes))):
                    if num:
                        formatted_image_text = (
                            f"Here is the original image {self.boi_token} and here are some crops to help you see better "
                            + " ".join([self.boi_token] * num)
                        )
                        prompt = (
                            prompt[:idx]
                            + formatted_image_text
                            + prompt[idx + len(self.boi_token) :]
                        )
                        prompts[batch_idx] = prompt

            # Expand placeholder image tokens to the full image token sequence
            prompts = [
                prompt.replace(self.boi_token, self.full_image_sequence)
                for prompt in prompts
            ]

        options.pop("return_tensors", None)
        return_mm_token_type_ids = options.pop("return_mm_token_type_ids", False)
        text_inputs = tokenize_inputs(self.tokenizer, prompts, options)

        # The tokenizer cannot tell which tokens are images, so build the types here.
        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(array_ids)
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data=to_mlx({**text_inputs, **image_inputs}))

    def batch_decode(
        self,
        sequences: list[int] | list[list[int]] | InputTensor,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs: JSONValue,
    ) -> list[str]:
        """Decode batched token sequences with the wrapped tokenizer."""
        return cast(BatchDecodeTokens, getattr(self.tokenizer, "batch_decode"))(
            sequences, skip_special_tokens, clean_up_tokenization_spaces, **kwargs
        )

    def decode(
        self,
        token_ids: DecodeIds,
        skip_special_tokens: bool = False,
        **kwargs: JSONValue,
    ) -> str | list[str]:
        """Decode token IDs with the wrapped tokenizer."""
        return cast(DecodeTokens, getattr(self.tokenizer, "decode"))(
            token_ids, skip_special_tokens, **kwargs
        )

    @property
    def model_input_names(self) -> list[str]:
        """Return tokenizer inputs plus image-processor inputs.

        ``token_type_ids`` is always included, while ``num_crops`` is dropped
        because it is instruction metadata rather than a model tensor.
        """
        tokenizer_input_names = self.tokenizer.model_input_names + ["token_type_ids"]
        image_processor_input_names: object = getattr(
            self.image_processor, "model_input_names", None
        )
        if not is_str_list(image_processor_input_names):
            raise TypeError("Image processor model_input_names must be strings")
        image_processor_input_names = [
            name for name in image_processor_input_names if name != "num_crops"
        ]
        return list(tokenizer_input_names + image_processor_input_names)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        cache_dir: str | PathLike[str] | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs: ProcessorValue,
    ) -> "Gemma3Processor":
        """Load the tokenizer, chat template and image processor of a checkpoint.

        ``image_seq_length`` and the image processor's ``patch_size`` and ``size``
        overrides are read from ``processor_config.json`` only when
        ``pretrained_model_name_or_path`` is a local directory containing it;
        otherwise ``image_seq_length`` is 256. The slow image processor
        (``use_fast=False``) is tried first, falling back to the default on
        ``ValueError``.

        Raises:
            ValueError: If ``processor_config.json`` is not a JSON object or has
                a malformed ``image_seq_length`` or ``image_processor`` entry.
        """
        import json
        from pathlib import Path

        from transformers import AutoImageProcessor

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        kwargs.update(
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
        )
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        kwargs.pop("use_fast", None)
        tokenizer = load_tokenizer(pretrained_model_name_or_path, kwargs)
        load_chat_template(tokenizer, pretrained_model_name_or_path)

        proc_cfg_path = Path(pretrained_model_name_or_path) / "processor_config.json"
        image_seq_length = 256
        ip_overrides: dict[str, ProcessorValue] = {}
        if proc_cfg_path.exists():
            with open(proc_cfg_path) as f:
                proc_cfg: object = json.load(f)
            if not is_json_object(proc_cfg):
                raise ValueError("processor_config.json must contain a JSON object")
            configured_length = proc_cfg.get("image_seq_length", 256)
            if not isinstance(configured_length, int):
                raise ValueError("Gemma image_seq_length must be an integer")
            image_seq_length = configured_length
            ip_cfg = proc_cfg.get("image_processor", {})
            if not isinstance(ip_cfg, dict):
                raise ValueError("Gemma image_processor must contain a JSON object")
            if "patch_size" in ip_cfg:
                ip_overrides["patch_size"] = ip_cfg["patch_size"]
            if "size" in ip_cfg:
                ip_overrides["size"] = ip_cfg["size"]

        try:
            image_processor = cast(
                _LoadImageProcessor, getattr(AutoImageProcessor, "from_pretrained")
            )(
                pretrained_model_name_or_path,
                use_fast=False,
                **ip_overrides,
                **kwargs,
            )
        except ValueError:
            image_processor = cast(
                _LoadImageProcessor, getattr(AutoImageProcessor, "from_pretrained")
            )(
                pretrained_model_name_or_path,
                **ip_overrides,
                **kwargs,
            )
        template: object = getattr(tokenizer, "chat_template", None)
        if template is not None and not isinstance(template, str):
            if not is_json_object(template) or not all(
                isinstance(value, str) for value in template.values()
            ):
                raise ValueError("Gemma chat_template must contain string templates")
            template = {
                key: value for key, value in template.items() if isinstance(value, str)
            }
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            # Generation uses this attribute to avoid adding a second BOS token
            # after the checkpoint's template has already supplied one.
            chat_template=template,
            image_seq_length=image_seq_length,
        )


__all__ = ["Gemma3Processor"]
