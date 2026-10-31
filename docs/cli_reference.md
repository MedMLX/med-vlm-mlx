# CLI reference

Run commands in your active package environment. Contributors using the
checkout's named environment can prefix them with
`bash environment/setup.sh --run` from the repository root.
`python -m mlx_vlm --help` lists the three supported commands.

| Command | Purpose |
| --- | --- |
| `python -m mlx_vlm generate` | Generate text with MedGemma. |
| `python -m mlx_vlm convert` | Convert or quantize a supported checkpoint. |
| `python -m mlx_vlm server` | Serve medical text-generation HTTP protocols. |

The installed `mlx_vlm.generate`, `mlx_vlm.convert` and `mlx_vlm.server` console
scripts call the same entry points. Append `--help` for the current options.

Generation accepts `--model`, `--revision`, `--image`,
`--prompt`, `--system`, `--max-tokens`, `--temperature`, `--top-p`,
`--top-k`, `--seed`, `--prefill-step-size`, and `--verbose`. A prompt is required.
The default model is `mlx-community/medgemma-1.5-4b-it-4bit`; pass a pinned
`--revision` for a repeatable checkpoint selection.

The server accepts a model path, bind address and port, model search
directories, a bearer API key, token/prefill limits, KV precision and log level.
Use `--host 127.0.0.1` for a local service. Runtime settings retain the shared cache
controls exposed by the HTTP schema.

Version 0.2 removes the previous general catalog commands and output-modality
switches. Unsupported architectures, checkpoint Python modules, and external
speculative drafters fail explicitly.
