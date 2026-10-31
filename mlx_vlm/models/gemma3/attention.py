"""Gemma image-block visibility and cache-aware attention mask construction."""

import mlx.core as mx

from mlx_vlm.contracts import AttentionCache, parse_int
from mlx_vlm.models.cache import BatchRotatingKVCache


def prompt_position(cache: AttentionCache | None) -> int:
    """Locate a query in the padded prompt, independently of per-row RoPE offsets.

    No cache starts at zero. A batched rotating cache uses its maximum row offset;
    other caches use an integer offset or a scalar ``position`` attribute.
    """
    if cache is None:
        return 0
    if isinstance(cache, BatchRotatingKVCache):
        # Warm rows align their prefix ends, even when the stored sliding
        # windows have discarded earlier columns. The ring cursor is local.
        return parse_int(mx.max(cache.offset).item())
    if isinstance(cache.offset, int):
        return cache.offset
    position: object = getattr(cache, "position", None)
    if not isinstance(position, int):
        raise TypeError("Gemma batched attention requires a scalar cache column cursor")
    return position


def image_attention_mask(
    input_ids: mx.array, padding_mask: mx.array | None, image_token_id: int
) -> mx.array:
    """Build the prompt visibility mask: causal text, bidirectional image blocks.

    A query may attend to a key at or before its own position, or to any key in
    its own image block. A block is a maximal run of consecutive
    ``image_token_id`` tokens; images that are not separated by another token
    (Gemma wraps each image in begin/end markers) would merge into one block.
    Padded keys are never visible. Padded query rows are not masked out, so
    callers must ignore their outputs.

    Args:
        input_ids: Token IDs with shape ``(batch, sequence)``; ``sequence > 0``.
        padding_mask: ``(batch, sequence)`` array of 0/1 or bool, where 0 marks
            a padded position. ``None`` means no padding.
        image_token_id: Placeholder ID that fills every image block.

    Returns:
        Boolean array of shape ``(batch, 1, sequence, sequence)`` indexed
        ``[batch, 0, query, key]``; ``True`` means the query may attend to the
        key. The singleton axis broadcasts over attention heads.

    Raises:
        ValueError: If ``input_ids`` is not 2-D with a nonempty sequence, the padding
            mask shape differs from ``input_ids``, or the mask is not binary.
    """
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("Gemma input_ids must be a nonempty [batch, sequence] array")
    if padding_mask is None:
        padding_mask = mx.ones(input_ids.shape, dtype=mx.bool_)
    if padding_mask.shape != input_ids.shape:
        raise ValueError("Gemma padding mask must match input_ids")
    if not bool(mx.all((padding_mask == 0) | (padding_mask == 1)).item()):
        raise ValueError("Gemma padding mask must contain only zero and one")
    is_image = input_ids == image_token_id
    previous = mx.pad(is_image[:, :-1], [(0, 0), (1, 0)])
    blocks = mx.cumsum((is_image & ~previous).astype(mx.int32), axis=1)
    same_image = (
        is_image[:, :, None]
        & is_image[:, None, :]
        & (blocks[:, :, None] == blocks[:, None, :])
    )
    positions = mx.arange(input_ids.shape[1])
    causal = positions[:, None] >= positions[None, :]
    return ((causal | same_image) & padding_mask[:, None, :].astype(mx.bool_))[:, None]


def cached_attention_mask(
    prompt_mask: mx.array,
    query_offset: int,
    query_length: int,
    key_length: int,
    window_size: int | None = None,
    rotation: int = 0,
) -> mx.array:
    """Slice the prompt mask for a cached forward step and extend it past the prompt.

    Query positions are absolute token positions ``[query_offset, query_offset +
    query_length)``. The keys are the ``key_length`` most recent positions, which
    is what a (possibly rotating) cache returns after ``update_and_fetch``.
    Pairs inside the prompt use ``prompt_mask`` (so image blocks stay
    bidirectional during chunked or cached prefill). A query past the prompt
    attends causally to every earlier key whose prompt position was not padded,
    read from the final prompt row, and to all generated keys.

    Args:
        prompt_mask: Boolean ``(batch, 1, prompt, prompt)`` mask from
            :func:`image_attention_mask`.
        query_offset: Absolute position of the first query token.
        query_length: Number of query tokens in this step.
        key_length: Number of keys returned by the cache; they end at the last
            query position, so a window cache that dropped older keys has
            ``key_length < query_offset + query_length``.
        window_size: Sliding-window width. A key at least ``window_size``
            positions behind its query is hidden; keys ahead of the query (the
            rest of an image block) are unaffected.
        rotation: Number of slots to roll the key axis by, so columns follow the
            ring-buffer slot order of a full rotating cache during single-token
            decoding. ``0`` for caches stored in temporal order.

    Returns:
        Boolean array of shape ``(batch, 1, query_length, key_length)``.

    The caller must supply consistent nonnegative positions and lengths, a positive
    window when used, and a key range that exists in the cache. These are not
    validated here; negative key indices follow MLX indexing semantics.

    Raises:
        ValueError: If ``prompt_mask`` is not a 4-D boolean array of shape
            ``(batch, 1, length, length)`` with ``length > 0``.
    """
    if prompt_mask.dtype != mx.bool_ or prompt_mask.ndim != 4:
        raise ValueError(
            "Gemma prompt attention must be a four-dimensional boolean mask"
        )
    prompt_length = prompt_mask.shape[-1]
    if prompt_length == 0 or prompt_mask.shape[1:3] != (1, prompt_length):
        raise ValueError(
            "Gemma prompt attention must have shape [batch, 1, length, length]"
        )
    end = query_offset + query_length
    queries = mx.arange(query_offset, end)
    keys = mx.arange(end - key_length, end)
    prompt_queries = mx.minimum(queries, prompt_length - 1)
    prompt_keys = mx.minimum(keys, prompt_length - 1)
    original = prompt_mask[:, :, prompt_queries[:, None], prompt_keys[None, :]]
    key_valid = (keys >= prompt_length) | prompt_mask[:, :, -1:, prompt_keys]
    causal = queries[:, None] >= keys[None, :]
    in_prompt = (queries[:, None] < prompt_length) & (keys[None, :] < prompt_length)
    mask = mx.where(in_prompt, original, causal & key_valid)
    if window_size is not None:
        mask &= keys[None, :] > queries[:, None] - window_size
    if rotation:
        mask = mx.roll(mask, rotation, -1)
    return mask
