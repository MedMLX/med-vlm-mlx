"""Rotary position embedding for the Gemma3 decoder."""

from typing import Optional

import mlx.nn as nn

from mlx_vlm.contracts import RopeScaling


def initialize_rope(
    dims: int,
    base: float,
    traditional: bool,
    scaling_config: Optional[RopeScaling] = None,
) -> nn.RoPE:
    """Build Gemma3's default or linearly scaled RoPE.

    Linear scaling multiplies positions by ``1 / factor``, as Gemma3's global
    layers configure. Other RoPE types raise ``ValueError``.
    """
    rope_type = "default"
    if scaling_config is not None:
        rope_type = scaling_config.get("type") or scaling_config.get(
            "rope_type", "default"
        )
    if rope_type == "default":
        return nn.RoPE(dims, traditional=traditional, base=base, scale=1.0)
    if rope_type == "linear":
        if scaling_config is None or "factor" not in scaling_config:
            raise ValueError("Linear RoPE requires factor")
        return nn.RoPE(
            dims, traditional=traditional, base=base, scale=1 / scaling_config["factor"]
        )
    raise ValueError(f"Unsupported RoPE type {rope_type}")
