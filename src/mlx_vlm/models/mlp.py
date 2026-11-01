"""Feed-forward blocks shared by the vision and language models."""

from typing import Protocol

import mlx.core as mx
import mlx.nn as nn


class MLPConfig(Protocol):
    """Widths needed to construct a two-layer feed-forward block."""

    hidden_size: int
    intermediate_size: int


class GELUMLP(nn.Module):
    """Gemma vision MLP using MLX's precise GELU and biased projections."""

    def __init__(self, config: MLPConfig):
        """Create the expansion and projection layers from ``config`` widths."""
        super().__init__()
        self.activation_fn = nn.GELU(approx="precise")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply the expansion, precise GELU, and projection."""
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x
