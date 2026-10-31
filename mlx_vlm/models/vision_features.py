"""Exact, ordered placement of vision features at their token placeholders."""

from typing import cast

import mlx.core as mx


def merge_vision_features(
    token_id: int,
    features: mx.array,
    inputs_embeds: mx.array,
    input_ids: mx.array,
) -> mx.array:
    """Overwrite placeholder-token embeddings with vision features, in order.

    Features are flattened to ``(num_features, hidden)`` and the i-th feature row
    replaces the i-th placeholder, counting in row-major order over ``input_ids``
    (batch row first, then sequence position). The placeholder count must equal
    the number of feature rows exactly: surplus or missing features are an error
    rather than being truncated or repeated.

    Args:
        token_id: Placeholder token ID to replace.
        features: ``(images, tokens, hidden)`` or ``(num_features, hidden)``.
        inputs_embeds: Token embeddings, shape ``(batch, sequence, hidden)``.
        input_ids: Token IDs, shape ``(batch, sequence)``.

    Returns:
        Array with the shape and dtype of ``inputs_embeds``; features are cast to
        that dtype. ``inputs_embeds`` itself when there are no placeholders. The
        input array is never modified.

    Raises:
        ValueError: If shapes or widths are inconsistent, the placeholder count
            differs from the feature count, or any feature is NaN or infinite.
    """
    if (
        input_ids.ndim != 2
        or inputs_embeds.ndim != 3
        or inputs_embeds.shape[:2] != input_ids.shape
    ):
        raise ValueError("Input embeddings must match [batch, sequence] token IDs")
    if features.ndim not in (2, 3) or features.shape[-1] != inputs_embeds.shape[-1]:
        raise ValueError("Vision feature width must match the language embedding width")
    features = mx.reshape(features, (-1, inputs_embeds.shape[-1]))
    positions = input_ids == token_id
    count = cast(int, mx.sum(positions).item())
    if count != features.shape[0]:
        raise ValueError(
            f"Vision placeholder count ({count}) does not match feature count ({features.shape[0]})"
        )
    if not bool(mx.all(mx.isfinite(features)).item()):
        raise ValueError("Vision features must be finite")
    if count == 0:
        return inputs_embeds
    indices = mx.maximum(
        mx.cumsum(mx.reshape(positions, (-1,)).astype(mx.int32)) - 1, 0
    )
    gathered = mx.reshape(features[indices], inputs_embeds.shape).astype(
        inputs_embeds.dtype
    )
    return mx.where(positions[..., None], gathered, inputs_embeds)
