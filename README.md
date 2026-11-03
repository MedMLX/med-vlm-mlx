<div align="center">

# med-vlm-mlx

Run MedGemma locally on Apple Silicon. Includes Python, CLI and HTTP interfaces.

[![PyPI](https://img.shields.io/pypi/v/med-vlm-mlx)](https://pypi.org/project/med-vlm-mlx/)
[![Python](https://img.shields.io/pypi/pyversions/med-vlm-mlx)](https://pypi.org/project/med-vlm-mlx/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](https://github.com/MedMLX/med-vlm-mlx/blob/v0.3.2/LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MLX-black?logo=apple)](https://github.com/ml-explore/mlx)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-MedGemma-yellow)](https://huggingface.co/mlx-community/medgemma-1.5-4b-it-4bit)
[![MedMLX](https://img.shields.io/badge/MedMLX-runtime-A51C30)](https://github.com/MedMLX/medmlx)

[Model weights](https://huggingface.co/mlx-community/medgemma-1.5-4b-it-4bit) · [MedMLX](https://github.com/MedMLX/medmlx)

</div>

| Input | Output |
| :--- | :--- |
| Image + text prompt | Generated text |

## Use with MedMLX

The [MedMLX MCP server](https://github.com/MedMLX/medmlx/blob/main/docs/mcp.md) is the lowest-friction way to use MedGemma. MedMLX runs this package in its own isolated environment as the provisional `medgemma` model, so an AI agent can describe, check and run it for you. Pass a local MedGemma folder as `weights_path`; see the [bridge setup and limits](https://github.com/MedMLX/medmlx/blob/main/docs/medvlm-bridge.md).

```sh
uv tool install --python 3.13 "medmlx[mcp]"
MEDMLX="$(uv tool dir --bin --color never)/medmlx"
mkdir -p "$HOME/medmlx-work"
claude mcp add --scope user medmlx -- "$MEDMLX" serve --root "$HOME/medmlx-work"
```

For Codex, use `codex mcp add medmlx -- "$MEDMLX" serve --root "$HOME/medmlx-work"`. Other MCP clients take the same command and arguments. The same install provides the `medmlx` CLI, for example `medmlx describe medgemma`.

> **For AI agents:** prefer the MedMLX MCP tools (`list_models`, `describe_model`, `model_readiness`, `run_model`, `get_job`) over writing Python against this package.

## Install

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install "med-vlm-mlx>=0.3.2"
```

Needs macOS on Apple Silicon with Metal and Python 3.13. The package
imports as `mlx_vlm`, so install it in a separate environment from upstream
`mlx-vlm`: both install the same modules.

## Run

Download the pinned [MedGemma 1.5 4B](https://huggingface.co/mlx-community/medgemma-1.5-4b-it-4bit) weights, then run:

```bash
uvx --from huggingface_hub hf download mlx-community/medgemma-1.5-4b-it-4bit \
  --revision 762746b7a72b0b68dd8fa3f9aea5dc3980674d23 --local-dir medgemma

python -m mlx_vlm.generate \
  --model medgemma --image image.png \
  --prompt 'Describe the image.' --max-tokens 128
```

### Weights

| Model | MLX weights | Revision |
| :--- | :--- | :--- |
| MedGemma 1.5 4B | [`mlx-community/medgemma-1.5-4b-it-4bit`](https://huggingface.co/mlx-community/medgemma-1.5-4b-it-4bit) | `762746b7a72b0b68dd8fa3f9aea5dc3980674d23` |
| MedGemma 4B (v1) | [`mlx-community/medgemma-4b-it-4bit`](https://huggingface.co/mlx-community/medgemma-4b-it-4bit) | `db10be5ae6bd569e2063c3c74b9b9dfbcbedcad3` |
| MedGemma 27B multimodal | [`mlx-community/medgemma-27b-it-4bit`](https://huggingface.co/mlx-community/medgemma-27b-it-4bit) | `f57d2d648d28950b1bc7abd6e1b7b06aff0e183a` |

All three are 4-bit Gemma3 image-text checkpoints; other architectures are
rejected. The `mlx-community` copies are not gated, but the
[Health AI Developer Foundations terms](https://developers.google.com/health-ai-developer-foundations/terms)
still apply.

### Command line

| Command | Purpose |
| :--- | :--- |
| `python -m mlx_vlm generate` | Generate text from a prompt and optional images |
| `python -m mlx_vlm convert` | Convert or quantize a Gemma3 checkpoint to MLX |
| `python -m mlx_vlm server` | Serve an OpenAI-compatible HTTP API |

The `mlx_vlm.generate`, `mlx_vlm.convert` and `mlx_vlm.server` console scripts
call the same entry points; add `--help` for every option. `generate` takes
`--model` (default `mlx-community/medgemma-1.5-4b-it-4bit`, with `--revision`
for a pinned commit), `--image` (paths or URLs), a required `--prompt`,
`--system`, `--max-tokens` (2048), `--temperature` (0, greedy), `--top-p`,
`--top-k`, `--seed` (0) and `--prefill-step-size`. `--verbose` streams the text
with token timings. Prompts without images also work.

### Python API

```python
import mlx.core as mx
from mlx_vlm import apply_chat_template, generate, load

if not mx.metal.is_available():
    raise RuntimeError("This example requires Metal")
mx.set_default_device(mx.gpu)

model, processor = load("medgemma", strict=True)
prompt = apply_chat_template(processor, model.config, "Describe this image.", num_images=1)
response = generate(
    model, processor, prompt, image=["image.png"], max_tokens=128, temperature=0.0
)
print(response.text)
```

`load` takes a local folder, or a Hub ID with `revision`. With
`local_files_only=True` (and an optional `cache_dir`) it uses only the Hugging
Face cache and raises instead of downloading. `stream_generate` takes the same
arguments and yields `GenerationResult` text segments; iterate it to the end so
the prompt cache and cleanup run. The final result has `finish_reason` and
`token_ids`, which include the stop token. `batch_generate` shares the same
generation and cache code.

### HTTP server

```bash
python -m mlx_vlm.server --host 127.0.0.1 --port 8080 --model medgemma

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"medgemma","messages":[{"role":"user","content":"What is 2 + 2?"}],"max_tokens":16}'
```

The server binds 127.0.0.1:8080 by default and provides `/v1/chat/completions`,
`/v1/models`, `/health`, `/metrics` and `/v1/settings` for runtime settings;
schemas are at `/docs` while it runs. `--api-key` requires a bearer token for
inference and management requests and is required when binding outside loopback.
`--model-dir` adds folders to search for models. Images use a base64
`data:image/...` URI in an `image_url` content part next to the text. Requests
are batched when their sampling, seed and logprob settings match; others wait
their turn in queue order.

### Inputs and limitations

- Text and images only. Requests with audio, video, tool definitions or tool-call
  turns are rejected.
- Only MedGemma (Gemma3) checkpoints load. Checkpoint Python code (`model_file`)
  and external speculative drafters are refused.
- Inference only, on Metal: no training, adapters, or CPU, MPS or PyTorch
  fallback.
- Prompts with images are prefilled whole, because a chunk could split an
  image's bidirectional attention block; text-only prompts use chunked prefill.
  The full prompt mask grows with the square of the prompt length.
- The checks cover numerical agreement on small synthetic models and one public
  radiograph per checkpoint. Clinical accuracy has not been evaluated.

## License

MIT, with Apache-2.0 adaptations. See [LICENSE](https://github.com/MedMLX/med-vlm-mlx/blob/v0.3.2/LICENSE) and [third-party notices](https://github.com/MedMLX/med-vlm-mlx/blob/v0.3.2/THIRD_PARTY_NOTICES.md). MedGemma weights are subject to the [Health AI Developer Foundations terms](https://developers.google.com/health-ai-developer-foundations/terms).

For research and education. Outputs require human review.
