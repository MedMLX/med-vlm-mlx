"""Validate KV-cache quantization options before constructing MLX caches."""

from __future__ import annotations

from typing import Optional

from mlx_vlm.contracts import KVQuantConfig


def kv_quant_config(
    kv_bits: Optional[float], kv_group_size: int = 64
) -> Optional[KVQuantConfig]:
    """Validate uniform KV quantization settings for ``mx.quantize`` caches.

    ``kv_bits=None`` disables KV quantization. Fractional bit widths raise
    instead of being truncated, because the uniform backend packs whole bits.
    """
    if kv_bits is None:
        return None
    if not float(kv_bits).is_integer():
        raise ValueError(
            f"uniform KV quantization requires integer bits, got {kv_bits}"
        )
    return {"bits": int(kv_bits), "group_size": int(kv_group_size)}
