"""SigLIP vision tower for Gemma3/MedGemma image encoding."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_vlm.models.attention import VisionAttention as Attention
from mlx_vlm.models.base import check_array_shape
from mlx_vlm.models.gemma3.config import VisionConfig
from mlx_vlm.models.mlp import GELUMLP as MLP


class EncoderLayer(nn.Module):
    """SigLIP transformer layer with pre-norm attention and MLP."""

    def __init__(self, config: VisionConfig):
        """Build attention, MLP and the two layer norms."""
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Attention(
            config.hidden_size, config.num_attention_heads, bias=True
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """Apply pre-norm attention and MLP residual blocks to ``x``."""
        r = self.self_attn(self.layer_norm1(x), mask)
        h = x + r
        r = self.mlp(self.layer_norm2(h))
        return h + r


class Encoder(nn.Module):
    """Apply SigLIP encoder layers, optionally retaining the input and each layer output."""

    def __init__(self, config: VisionConfig):
        """Create ``num_hidden_layers`` SigLIP encoder layers."""
        super().__init__()
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(
        self,
        x: mx.array,
        output_hidden_states: Optional[bool] = None,
        mask: Optional[mx.array] = None,
    ) -> tuple[mx.array, tuple[mx.array, ...] | None]:
        """Run every layer and optionally return the input and layer states."""
        encoder_states = (x,) if output_hidden_states else None
        h = x
        for l in self.layers:
            x = l(x, mask=mask)
            if encoder_states is not None:
                encoder_states = encoder_states + (x,)

            h = x

        return (h, encoder_states)


class VisionEmbeddings(nn.Module):
    """Project channel-last image patches and add learned fixed-grid positions.

    The model expects the configured number of patches. Exact input height,
    width, and patch count are not checked explicitly; array broadcasting may
    accept a singleton patch grid.
    """

    def __init__(self, config: VisionConfig):
        """Create the patch convolution and fixed per-position embeddings."""
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """Patchify ``(batch, height, width, channels)`` pixels and add positions."""
        patch_embeddings = self.patch_embedding(x)
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
        position_ids = mx.array(np.arange(self.num_positions)[None, :])
        embeddings = patch_embeddings
        embeddings += self.position_embedding(position_ids)
        return embeddings


class SigLipVisionModel(nn.Module):
    """SigLIP tower: patch embeddings, encoder layers and post normalization."""

    def __init__(self, config: VisionConfig):
        """Create the embedding, encoder and post-normalization layers."""
        super().__init__()
        self.embeddings = VisionEmbeddings(config)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def __call__(
        self,
        x: mx.array,
        output_hidden_states: Optional[bool] = None,
    ) -> tuple[mx.array, mx.array, tuple[mx.array, ...] | None]:
        """Encode pixels and return normalized features, patches and states."""
        x = self.embeddings(x)

        encoder_outputs = self.encoder(
            x=x, output_hidden_states=output_hidden_states, mask=None
        )

        pooler_output = self.post_layernorm(encoder_outputs[0])

        return pooler_output, x, encoder_outputs[-1]


class VisionModel(nn.Module):
    """Gemma3 vision tower with checkpoint weight sanitization."""

    def __init__(self, config: VisionConfig):
        """Create the SigLIP tower for an accepted vision model type."""
        super().__init__()
        self.model_type = config.model_type
        if self.model_type not in ["siglip_vision_model", "gemma3", "gemma3_vision"]:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.vision_model = SigLipVisionModel(config)

    def __call__(
        self, x: mx.array, output_hidden_states: Optional[bool] = None
    ) -> tuple[mx.array, mx.array, tuple[mx.array, ...] | None]:
        """Encode images with the SigLIP tower.

        Args:
            x: Channel-last pixels, normally ``(batch, image_size, image_size,
                num_channels)`` as specified by the vision config.
            output_hidden_states: Also return the per-layer hidden states.

        Returns:
            The final hidden states after ``post_layernorm``, shape
            ``(batch, patches, hidden)`` (there is no pooling head); the patch
            embeddings before the encoder; and the hidden states of the
            embeddings and every layer, or ``None`` when not requested.
        """
        return self.vision_model(x, output_hidden_states)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Transpose PyTorch patch-embedding weights into MLX channels-last layout."""
        sanitized_weights: dict[str, mx.array] = {}
        for k, v in weights.items():
            if "patch_embedding.weight" in k:
                # PyTorch conv2d weight tensors have shape:
                #   [out_channels, in_channels, kH, KW]
                # MLX conv2d expects the weight be of shape:
                #   [out_channels, kH, KW, in_channels]
                if check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    sanitized_weights[k] = mx.transpose(v, (0, 2, 3, 1))
            else:
                sanitized_weights[k] = v

        return sanitized_weights
