import mlx.core as mx

from mlx_vlm.sample_utils import apply_top_n_sigma, apply_typical_p


def test_typical_sampling_renormalizes_after_sigma_masks_tokens() -> None:
    # Sigma retains 0.4 and 0.3. Their conditional probabilities are 4/7
    # and 3/7, and the first is more typical and alone exceeds the threshold.
    logprobs = mx.log(mx.array([[0.4, 0.3, 0.2, 0.1]]))
    filtered = apply_top_n_sigma(logprobs, 0.8)
    result = apply_typical_p(filtered, 0.5)

    assert mx.isfinite(filtered).tolist() == [[True, True, False, False]]
    assert mx.isfinite(result).tolist() == [[True, False, False, False]]


def test_typical_sampling_orders_by_surprisal_after_masking() -> None:
    # The surviving conditional probabilities are [0.5, 0.3, 0.2].
    # Entropy is about 1.03 nats, making the middle token most typical;
    # its probability exceeds 0.25, so it is the only retained token.
    logprobs = mx.array([[0.4, 0.24, 0.16, 0.0]])
    result = apply_typical_p(mx.log(logprobs), 0.25)

    assert mx.isfinite(result).tolist() == [[False, True, False, False]]
