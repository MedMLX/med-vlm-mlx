# med-vlm-mlx

Run MedGemma locally on Apple Silicon. Includes Python, CLI and HTTP interfaces.

[Docs](https://github.com/MedMLX/med-vlm-mlx/blob/main/docs/index.md) · [MedMLX](https://github.com/MedMLX)

| Input | Output |
| :--- | :--- |
| Image + text prompt | Generated text |

**Apple Silicon · Metal · Python 3.13** · [uv](https://docs.astral.sh/uv/)

## Quick start

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install "med-vlm-mlx>=0.3.1"
```

The three checkpoint integration checks below ran at source revision
[`d9336a2`](https://github.com/MedMLX/med-vlm-mlx/commit/d9336a242b001c1e27abb4bb2d5f99ada65959bd).

Download the pinned [MedGemma 1.5 4B](https://huggingface.co/mlx-community/medgemma-1.5-4b-it-4bit) weights, then run:

```bash
uvx --from huggingface_hub hf download mlx-community/medgemma-1.5-4b-it-4bit \
  --revision 762746b7a72b0b68dd8fa3f9aea5dc3980674d23 --local-dir medgemma

python -m mlx_vlm.generate \
  --model medgemma --image image.png \
  --prompt 'Describe the image.' --max-tokens 128
```

## Supported checkpoints

| Model | MLX weights | Revision |
| :--- | :--- | :--- |
| MedGemma 1.5 4B | [`mlx-community/medgemma-1.5-4b-it-4bit`](https://huggingface.co/mlx-community/medgemma-1.5-4b-it-4bit) | `762746b7a72b0b68dd8fa3f9aea5dc3980674d23` |
| MedGemma 4B (v1) | [`mlx-community/medgemma-4b-it-4bit`](https://huggingface.co/mlx-community/medgemma-4b-it-4bit) | `db10be5ae6bd569e2063c3c74b9b9dfbcbedcad3` |
| MedGemma 27B multimodal | [`mlx-community/medgemma-27b-it-4bit`](https://huggingface.co/mlx-community/medgemma-27b-it-4bit) | `f57d2d648d28950b1bc7abd6e1b7b06aff0e183a` |

All three are Gemma3 image-text checkpoints. Other architectures are rejected.

Uses the `mlx_vlm` namespace. Install in a separate environment from upstream `mlx-vlm`.

## Documentation

[Installation](docs/installation.md) · [Python, CLI and HTTP usage](docs/usage.md) · [CLI reference](docs/cli_reference.md) · [Development](CONTRIBUTING.md)

## License

MIT, with Apache-2.0 adaptations. See [LICENSE](https://github.com/MedMLX/med-vlm-mlx/blob/main/LICENSE) and [third-party notices](https://github.com/MedMLX/med-vlm-mlx/blob/main/THIRD_PARTY_NOTICES.md). MedGemma weights are subject to the [Health AI Developer Foundations terms](https://developers.google.com/health-ai-developer-foundations/terms).

For research and education. Outputs require human review.
