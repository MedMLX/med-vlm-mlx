# Medical inference

Run these examples in the active package environment from the
[installation guide](installation.md#use-the-package). Contributors using the
checkout's named conda environment can prefix commands with
`bash environment/setup.sh --run` from the repository root.
Only Gemma3 (MedGemma) checkpoints are accepted.
Pinned model IDs and revisions are in [the checkpoint manifest](../scripts/checkpoints.json).

## CLI

```sh
python -m mlx_vlm.generate \
  --model /path/to/medgemma --image image.png \
  --prompt 'Describe this image.' --max-tokens 128
```

Use `--verbose` to stream text with token timings. MedGemma accepts image input
for its visual path and also supports text-only prompts. [CLI options](cli_reference.md) describe the
maintained commands.

## Python

```python
import mlx.core as mx
from mlx_vlm import apply_chat_template, generate, load

if not mx.metal.is_available():
    raise RuntimeError("This example requires Metal")
mx.set_default_device(mx.gpu)

model, processor = load('/path/to/medgemma', strict=True)
prompt = apply_chat_template(
    processor, model.config, 'Describe this image.', num_images=1
)
if not isinstance(prompt, str):
    raise TypeError("The chat template must return text")
response = generate(
    model, processor, prompt, image=['image.png'], max_tokens=128,
    temperature=0.0,
)
print(response.text)
```

For a checkpoint already in a Hugging Face cache, `load` also accepts a repository
ID with `local_files_only=True` and an optional `cache_dir`. An absent snapshot
raises an error instead of fetching it. A local directory path also avoids Hub
snapshot resolution. MedGemma does not accept audio input.

`stream_generate` yields incremental `GenerationResult` text segments:

```python
from mlx_vlm import stream_generate

for chunk in stream_generate(
    model, processor, prompt, image=['image.png'], max_tokens=128,
    temperature=0.0,
):
    print(chunk.text, end='', flush=True)
print()
```

Iterate through the generator to exhaustion, including after its final result,
to run prompt-cache updates and generation cleanup. The final result includes
`finish_reason` and `token_ids`; those IDs include the stop token if one ended
generation.
`batch_generate` and the HTTP server share the native generation and cache
infrastructure. Medical image prompts preserve the model-specific prefill
contract; see [technical notes](technical.md).

## HTTP server

```sh
python -m mlx_vlm.server \
  --host 127.0.0.1 --port 8080 --model /path/to/medgemma
```

The service provides OpenAI-compatible `/v1/chat/completions`, `/v1/models`,
`/health`, `/metrics`, and runtime settings management. API schemas
are available at `/docs` while the server runs. `--api-key` requires a bearer
token for inference and management requests.

The batched server groups requests with matching sampling, seed and logprob
settings. Requests with different settings wait for a compatible batch, preserving
their queue order.

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/path/to/medgemma","messages":[{"role":"user","content":"What is 2 + 2?"}],"max_tokens":16}'
```

Image requests use an `image_url` content part alongside the text part.
Requests that define tools, carry tool-call turns or include video or audio parts
fail validation; MedGemma supports only text and images.
Speech, embeddings, reranking, decisions and image/video generation endpoints
were removed in version 0.2. Model discovery lists only the retained Gemma3 architecture. Clinical accuracy is outside these integration checks.
