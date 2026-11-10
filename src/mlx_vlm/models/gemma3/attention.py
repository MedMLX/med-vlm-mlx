"""Gemma image-block visibility and cache-aware attention mask construction."""

import mlx.core as mx

from mlx_vlm.contracts import AttentionCache, array_int_rows, parse_int
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


def image_attention_metadata(
    input_ids: mx.array, padding_mask: mx.array | None, image_token_id: int
) -> tuple[mx.array, mx.array]:
    """Return linear-size key validity and image-block IDs for a complete prompt.

    Both arrays have shape ``(batch, sequence)``. Validity is boolean; int32 block
    IDs are positive for each maximal placeholder run and zero for text. Two
    images need a separating token to form distinct blocks, as Gemma's image
    begin/end markers provide. Padding hides keys, not query rows.

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
    return padding_mask.astype(mx.bool_), mx.where(is_image, blocks, 0)


def image_attention_mask(
    input_ids: mx.array, padding_mask: mx.array | None, image_token_id: int
) -> mx.array:
    """Materialize the full causal-text/bidirectional-image mask for explicit callers.

    The model carries :func:`image_attention_metadata` instead. This dense helper
    retains the ``(batch, 1, sequence, sequence)`` mask interface.
    """
    validity, blocks = image_attention_metadata(input_ids, padding_mask, image_token_id)
    length = input_ids.shape[1]
    return cached_attention_mask(validity, 0, length, length, image_blocks=blocks)


def image_aligned_prefill_sizes(
    blocks: mx.array | None, offset: int, length: int, step_size: int
) -> list[int]:
    """Plan bounded chunks, extending a cut until every row's image block ends.

    The final chunk remains for first-token generation, including a whole image
    block when the prompt ends in one. Metadata crosses to host once per plan;
    no token-by-token synchronization or square mask is needed for scheduling.
    """
    boundaries = [True] * (offset + length + 1)
    for row in array_int_rows(blocks) if blocks is not None else []:
        for position in range(1, len(row)):
            if (
                position < len(boundaries)
                and row[position] > 0
                and row[position] == row[position - 1]
            ):
                boundaries[position] = False
    sizes: list[int] = []
    end = offset + length
    while offset < end - 1:
        stop = min(offset + step_size, end - 1)
        while stop < end and not boundaries[stop]:
            stop += 1
        if stop == end:
            break
        sizes.append(stop - offset)
        offset = stop
    return sizes


def cached_attention_mask(
    prompt_mask: mx.array,
    query_offset: int,
    query_length: int,
    key_length: int,
    window_size: int | None = None,
    rotation: int = 0,
    *,
    image_blocks: mx.array | None = None,
) -> mx.array:
    """Slice the prompt mask for a cached forward step and extend it past the prompt.

    Query positions are absolute token positions ``[query_offset, query_offset +
    query_length)``. The keys are the ``key_length`` most recent positions, which
    is what a (possibly rotating) cache returns after ``update_and_fetch``.
    Pairs inside the prompt use ``prompt_mask`` (so image blocks stay
    bidirectional during chunked or cached prefill). A query past the prompt
    attends causally to every earlier key whose prompt position was not padded,
    read from complete prompt metadata, and to all generated keys.

    Args:
        prompt_mask: Boolean ``(batch, prompt)`` key validity when ``image_blocks``
            is supplied, otherwise an explicit ``(batch, 1, prompt, prompt)`` mask.
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
        image_blocks: Int32 ``(batch, prompt)`` IDs from ``image_attention_metadata``.

    Returns:
        Boolean array of shape ``(batch, 1, query_length, key_length)``.

    The caller must supply consistent nonnegative positions and lengths, a positive
    window when used, and a key range that exists in the cache. These are not
    validated here; negative key indices follow MLX indexing semantics.

    Raises:
        ValueError: If dense or compact metadata has an incompatible shape/dtype.
    """
    if image_blocks is not None:
        if (
            prompt_mask.dtype != mx.bool_
            or prompt_mask.ndim != 2
            or image_blocks.shape != prompt_mask.shape
            or image_blocks.dtype != mx.int32
            or prompt_mask.shape[1] == 0
        ):
            raise ValueError(
                "Gemma compact attention requires matching boolean/int32 [batch, prompt] metadata"
            )
        prompt_length = prompt_mask.shape[1]
    elif prompt_mask.dtype != mx.bool_ or prompt_mask.ndim != 4:
        raise ValueError(
            "Gemma prompt attention must be a four-dimensional boolean mask"
        )
    else:
        prompt_length = prompt_mask.shape[-1]
    if image_blocks is None and (
        prompt_length == 0 or prompt_mask.shape[1:3] != (1, prompt_length)
    ):
        raise ValueError(
            "Gemma prompt attention must have shape [batch, 1, length, length]"
        )
    end = query_offset + query_length
    queries = mx.arange(query_offset, end)
    keys = mx.arange(end - key_length, end)
    prompt_queries = mx.minimum(queries, prompt_length - 1)
    prompt_keys = mx.minimum(keys, prompt_length - 1)
    causal = queries[:, None] >= keys[None, :]
    in_prompt = (queries[:, None] < prompt_length) & (keys[None, :] < prompt_length)
    if image_blocks is None:
        original = prompt_mask[:, :, prompt_queries[:, None], prompt_keys[None, :]]
        key_valid = (keys >= prompt_length) | prompt_mask[:, :, -1:, prompt_keys]
        mask = mx.where(in_prompt, original, causal & key_valid)
    else:
        query_blocks = image_blocks[:, prompt_queries, None]
        key_blocks = image_blocks[:, None, prompt_keys]
        same_image = (query_blocks > 0) & (query_blocks == key_blocks) & in_prompt
        key_valid = (keys >= prompt_length) | prompt_mask[:, prompt_keys]
        mask = ((causal | same_image) & key_valid[:, None, :])[:, None]
    if window_size is not None:
        mask &= keys[None, :] > queries[:, None] - window_size
    if rotation:
        mask = mx.roll(mask, rotation, -1)
    return mask
