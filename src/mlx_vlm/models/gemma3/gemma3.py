"""Gemma3 multimodal model (MedGemma): SigLIP tower, projector and decoder."""

from collections.abc import Mapping, Sequence
from typing import Optional, Self, TypedDict, Unpack

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_vlm.contracts import (
    AttentionCache,
    ModelArgument,
    einsum,
    evaluate,
)
from mlx_vlm.models.base import InputEmbeddingsFeatures
from mlx_vlm.models.gemma3.attention import (
    image_attention_mask,
    image_attention_metadata,
)
from mlx_vlm.models.gemma3.config import ModelConfig
from mlx_vlm.models.gemma3.language import LanguageModel, RMSNorm
from mlx_vlm.models.gemma3.vision import VisionModel
from mlx_vlm.models.vision_features import merge_vision_features
from mlx_vlm.vision_cache import VisionFeatureCache

type _ParameterTree = (
    mx.array
    | list[_ParameterTree]
    | tuple[_ParameterTree, ...]
    | dict[str, _ParameterTree]
)


class EmbeddingOptions(TypedDict, total=False):
    """Optional keyword inputs of :meth:`Model.get_input_embeddings`."""

    # Projector output supplied by the caller (for example from
    # ``VisionFeatureCache``); skips the vision tower and projector when
    # ``pixel_values`` is also supplied. Shape
    # ``(images, mm_tokens_per_image, hidden)`` or ``(tokens, hidden)``.
    cached_image_features: mx.array | None
    vision_cache: VisionFeatureCache | None


class Gemma3MultiModalProjector(nn.Module):
    """Pool SigLIP patch tokens to ``mm_tokens_per_image`` and project them.

    Patches form a square ``patches_per_image x patches_per_image`` grid in
    row-major order. Average pooling reduces it to a ``tokens_per_side`` square
    grid (``mm_tokens_per_image`` tokens, 256 for MedGemma), followed by RMS
    normalization and a bias-free projection to the language hidden width.
    ``mm_tokens_per_image`` is validated as a positive square by ``TextConfig``.
    Matching the requested count also depends on the patch-grid size; that
    relationship is not validated here.
    """

    def __init__(self, config: ModelConfig):
        """Create the pooling kernel, RMS norm and projection weight."""
        super().__init__()
        self.mm_input_projection_weight = mx.ones(
            (config.vision_config.hidden_size, config.text_config.hidden_size)
        )

        self.mm_soft_emb_norm = RMSNorm(
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )
        self.patches_per_image = int(
            config.vision_config.image_size // config.vision_config.patch_size
        )
        self.tokens_per_side = int(config.text_config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(
            kernel_size=self.kernel_size, stride=self.kernel_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Map vision features to language-width image tokens.

        Args:
            x: Vision tower output, shape ``(batch, patches, vision_hidden)``
                with ``patches == patches_per_image ** 2``.

        Returns:
            Array of shape ``(batch, mm_tokens_per_image, text_hidden)`` in the
            dtype of ``x``. Features are not yet divided by the embedding
            scaler; see :meth:`Model.prepare_inputs_for_multimodal`.
        """
        b, _, l = x.shape

        reshaped_vision_outputs = mx.transpose(x, (0, 2, 1))
        reshaped_vision_outputs = mx.reshape(
            reshaped_vision_outputs,
            (b, l, self.patches_per_image, self.patches_per_image),
        )

        # Transpose to place h, w in indices 1, 2
        reshaped_vision_outputs = mx.transpose(reshaped_vision_outputs, (0, 2, 3, 1))
        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = mx.transpose(
            pooled_vision_outputs, (0, 3, 1, 2)
        ).flatten(2)
        pooled_vision_outputs = mx.transpose(pooled_vision_outputs, (0, 2, 1))

        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        projected_vision_outputs = einsum(
            "btm,md->btd", normed_vision_outputs, self.mm_input_projection_weight
        )
        return projected_vision_outputs.astype(x.dtype)


class Model(nn.Module):
    """MedGemma/Gemma3 image-text model.

    Image placeholder tokens in ``input_ids`` are replaced by projected SigLIP
    features. Text attends causally and each image block attends to itself
    bidirectionally. Linear-size validity and image-block metadata are carried
    through prefill and decoding; each layer constructs its current Q-by-K mask.
    """

    # Embedding outputs that ``generate_step`` keeps passing to the decoder at
    # every decode step: cached steps rebuild their mask slice from the prompt mask.
    generation_persistent_kwargs = (
        "attention_mask_4d",
        "prompt_key_validity",
        "prompt_image_blocks",
    )

    def __init__(self, config: ModelConfig):
        """Create the vision tower, multimodal projector and text decoder."""
        super().__init__()
        self.model_type = config.model_type
        self.config = config

        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.multi_modal_projector = Gemma3MultiModalProjector(config)
        self._vision_cache_identity = object()
        self._vision_cache_parameters: tuple[tuple[str, mx.array], ...] = ()

    def invalidate_vision_cache(self) -> None:
        """Invalidate projected features after an in-place model edit.

        Call after array-index assignments or writes through external buffer
        views. MLX exposes no public mutation counter for those edits. Standard
        ``update``/``load_weights``/``apply`` invalidate automatically; replacing
        vision/projector parameter objects is detected at the next cache lookup.
        No weight contents are read or synchronized to maintain this identity.
        """
        self._vision_cache_identity = object()
        self._vision_cache_parameters = ()

    def update(
        self, parameters: dict[str, _ParameterTree], strict: bool = True
    ) -> Self:
        """Replace parameters and invalidate even a partially applied update."""
        # Invalidate before the base update, which can change some parameters
        # before raising. Dropping observed references also releases old weights.
        self.invalidate_vision_cache()
        return super().update(parameters, strict=strict)

    def _vision_namespace(self, dtype: mx.Dtype) -> tuple[object, ...]:
        """Identify current parameter objects without relying on recyclable IDs."""
        parameters = tuple(
            tree_flatten(
                {
                    "vision": self.vision_tower.parameters(),
                    "projector": self.multi_modal_projector.parameters(),
                }
            )
        )
        previous = self._vision_cache_parameters
        if len(previous) != len(parameters) or any(
            old_name != name or old_weight is not weight
            for (old_name, old_weight), (name, weight) in zip(
                previous, parameters, strict=True
            )
        ):
            self.invalidate_vision_cache()
            # Only this model retains the current objects; cache keys retain the
            # generation token, never weights from obsolete model generations.
            self._vision_cache_parameters = parameters
        return (self._vision_cache_identity, repr(self.config), str(dtype))

    def encode_images(self, pixel_values: mx.array, *, dtype: mx.Dtype) -> mx.array:
        """Evaluate projected image tokens using the uncached embedding dtype."""
        hidden_state, _, _ = self.vision_tower(
            mx.transpose(pixel_values, (0, 2, 3, 1)).astype(dtype),
            output_hidden_states=True,
        )
        features = self.multi_modal_projector(hidden_state)
        evaluate(features)
        return features

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        **kwargs: Unpack[EmbeddingOptions],
    ) -> InputEmbeddingsFeatures:
        """Embed tokens, merge image features and build the prompt mask.

        Args:
            input_ids: Token IDs, shape ``(batch, sequence)``.
            pixel_values: Channel-first processor output, shape
                ``(images, 3, image_size, image_size)``, already rescaled and
                normalized by the image processor. ``None`` for text-only
                prompts.
            mask: Optional ``(batch, sequence)`` padding mask, 1 for real tokens.
            **kwargs: See :class:`EmbeddingOptions`. Cached features are used
                only when ``pixel_values`` is not ``None``; a cached feature
                array alone does not satisfy image placeholders at this boundary.

        Returns:
            ``inputs_embeds`` of shape ``(batch, sequence, hidden)``, not yet
            multiplied by the Gemma embedding scaler (the decoder applies it),
            plus boolean ``prompt_key_validity`` and int32 ``prompt_image_blocks``
            of shape ``(batch, sequence)``.

        Raises:
            ValueError: If ``input_ids`` is missing, image placeholders have no
                ``pixel_values``, or the placeholder count differs from the
                number of image feature rows.
        """
        if input_ids is None:
            raise ValueError("Gemma input_ids are required")
        if pixel_values is None:
            if bool(mx.any(input_ids == self.config.image_token_index).item()):
                raise ValueError("Gemma image placeholders require pixel_values")
            validity, blocks = image_attention_metadata(
                input_ids, mask, self.config.image_token_index
            )
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids),
                prompt_key_validity=validity,
                prompt_image_blocks=blocks,
            )

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        cached = kwargs.get("cached_image_features", None)
        vision_cache = kwargs.get("vision_cache")
        namespace = (
            self._vision_namespace(inputs_embeds.dtype)
            if vision_cache is not None
            else None
        )
        if cached is None and vision_cache is not None:
            cached = vision_cache.get(pixel_values, namespace=namespace)
            if cached is not None and not isinstance(cached, mx.array):
                raise TypeError("Gemma cached image features must be one MLX array")
        if cached is not None:
            image_features = cached
        else:
            image_features = self.encode_images(
                pixel_values,
                dtype=inputs_embeds.dtype,
            )
            if vision_cache is not None:
                vision_cache.put(pixel_values, image_features, namespace=namespace)

        final_inputs_embeds = self._merge_multimodal_embeddings(
            self.config.image_token_index,
            image_features,
            inputs_embeds,
            input_ids,
        )
        validity, blocks = image_attention_metadata(
            input_ids, mask, self.config.image_token_index
        )
        return InputEmbeddingsFeatures(
            inputs_embeds=final_inputs_embeds,
            prompt_key_validity=validity,
            prompt_image_blocks=blocks,
        )

    @staticmethod
    def prepare_inputs_for_multimodal(
        image_token_index: int,
        image_features: mx.array,
        inputs_embeds: mx.array,
        input_ids: mx.array,
        attention_mask: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        """Place image features at their placeholders and build the prompt mask.

        Args:
            image_token_index: Placeholder token ID.
            image_features: Projected features, ``(images, tokens, hidden)`` or
                ``(images * tokens, hidden)``, in image order.
            inputs_embeds: Unscaled token embeddings, ``(batch, sequence, hidden)``.
            input_ids: Token IDs, ``(batch, sequence)``.
            attention_mask: Optional ``(batch, sequence)`` padding mask.

        Returns:
            The merged embeddings and the ``(batch, 1, sequence, sequence)``
            boolean prompt mask.

        Raises:
            ValueError: If the placeholder count differs from the number of
                feature rows, or features are not finite.
        """
        embeddings = Model._merge_multimodal_embeddings(
            image_token_index,
            image_features,
            inputs_embeds,
            input_ids,
        )
        return embeddings, image_attention_mask(
            input_ids, attention_mask, image_token_index
        )

    @staticmethod
    def _merge_multimodal_embeddings(
        image_token_index: int,
        image_features: mx.array,
        inputs_embeds: mx.array,
        input_ids: mx.array,
    ) -> mx.array:
        """Merge image rows with the decoder's embedding-scale compensation."""
        # The decoder multiplies every input embedding by this scaler (rounded
        # through bfloat16), so divide the image features by it here and they
        # reach the decoder unscaled, as in Transformers. The width comes from
        # the tensor, not a config default: MedGemma's 2560 differs from a
        # default of 2048.
        scaler = mx.array(inputs_embeds.shape[-1] ** 0.5, mx.bfloat16).astype(
            inputs_embeds.dtype
        )
        embeddings = merge_vision_features(
            image_token_index, image_features / scaler, inputs_embeds, input_ids
        )
        return embeddings

    def chunked_prefill_policy(
        self,
        *,
        input_ids: mx.array | None = None,
        prefill_kwargs: Mapping[str, ModelArgument] | None = None,
        **kwargs: object,
    ) -> bool:
        """Enable planned image-aligned chunks while honoring explicit dense masks."""
        return self.language_model.chunked_prefill_policy(prefill_kwargs=prefill_kwargs)

    @property
    def layers(self):
        """Return the decoder blocks in execution order."""
        return self.language_model.model.layers

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: Optional[mx.array] = None,
        cache: Sequence[AttentionCache | None] | None = None,
        **kwargs: ModelArgument,
    ):
        """Run a full forward pass and return a ``LanguageModelOutput`` of logits.

        Computes embeddings and the prompt mask with :meth:`get_input_embeddings`
        (``pixel_values`` may be ``None``), then runs the decoder. ``logits`` has
        shape ``(batch, sequence, vocab)``. With a ``cache`` the prompt mask must
        cover the whole cached prompt, so image prompts cannot be fed in
        arbitrary pieces. Extra keyword arguments are accepted but not forwarded.
        """
        input_embeddings_features = self.get_input_embeddings(
            input_ids, pixel_values, mask
        )
        inputs_embeds = input_embeddings_features.inputs_embeds
        logits = self.language_model(
            inputs=input_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            attention_mask_4d=input_embeddings_features.attention_mask_4d,
            prompt_key_validity=input_embeddings_features.prompt_key_validity,
            prompt_image_blocks=input_embeddings_features.prompt_image_blocks,
        )
        return logits
