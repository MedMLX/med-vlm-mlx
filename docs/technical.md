# Development and validation

Use the environment described in
[development setup](installation.md#develop-from-source). Run commands from the
repository root.

## Local checks

| Command | What it checks |
| --- | --- |
| `make qa` | Formatting, Ruff policy, Pyright in its configured standard mode, checkpoint-manifest tests, and setup-process tests. |
| `make format-check` | Locked Black and isort checks over the maintained Python scope. |
| `make typecheck` | The maintained source, medical tests, scripts, and stubs listed in `pyproject.toml`. |
| `make test-medical-host` | Checkpoint-manifest parsing, round trips, and malformed inputs without loading a model. |
| `make test-setup` | Setup and command-runner behavior through subprocess tests. |
| `make test` | Medical regression tests with Metal. |

For an affected medical test, select its file explicitly:

```bash
make test TESTS=tests/medical/test_regressions.py
```

`make qa` uses the existing environment and does not load checkpoints or run
the numerical medical suite. Medical inference changes require the affected
Metal tests. Pyright's maintained scope is declared in `pyproject.toml`;
passing it does not establish numerical or clinical qualification.

Validation runs locally; the repository has no hosted CI workflows. Builds and
package publication are separate release tasks. There are no pre-commit hooks,
because they would duplicate the explicit formatter target.

## What the supporting folders contain

| Path | Purpose |
| --- | --- |
| [contracts/medical-fp32.json](../contracts/medical-fp32.json) | Numeric tolerances for the medical regressions, with recorded array types and shapes. |
| [scripts/checkpoints.json](../scripts/checkpoints.json) | Exact MedGemma checkpoint revisions. |
| [scripts/checkpoint_smoke.py](../scripts/checkpoint_smoke.py) | Strict checkpoint loading and a short image-to-text integration check on Metal. |
| [tests/medical/](../tests/medical/) | Numerical, generation, and manifest regressions. |

## Checkpoint integration

```bash
make smoke-medgemma                              # MedGemma 1.5 4B
make smoke-medgemma CHECKPOINT=medgemma-4b-v1    # MedGemma 4B v1
make smoke-medgemma CHECKPOINT=medgemma-27b      # MedGemma 27B multimodal
```

This command loads the pinned weights, downloads a public CC0 normal PA chest
radiograph from Wikimedia Commons into a temporary directory, asks the model to
describe it, and prints the generated text, token counts, checkpoint revision,
image source and device as JSON to the terminal. It downloads weights when absent
and selects Metal explicitly. Save results outside the checkout if a new record
is needed; medical images are never committed.

One public radiograph checks real image-to-text integration. It is not a
clinical benchmark.

## Medical implementation and evidence

The medical corrections cover Gemma image/text attention, feature scaling,
and preservation of caller-owned embeddings. Gemma prompts containing image tokens use whole-prompt prefill;
text-only prompts retain chunked prefill.

The [medical implementation record](medical-port.html) contains the review
measurements dated 8 October 2026. Treat its pass counts and model observations
as a dated snapshot; use the local checks above for the current checkout.

The FP32 regression tolerance is defined once in the numeric contract. Exact
mask and feature-placement checks accompany the floating-point comparisons.
Full quantized checkpoint parity, clinical accuracy, and a speed advantage
remain outside the recorded qualification.

## Maintenance

`environment/setup.sh` owns setup and command execution. Conda owns Python
3.13; `pyproject.toml` and `uv.lock` own Python dependencies. Keep the patched
host Node for Pyright instead of installing its bundled `nodejs` extra.

These Markdown guides are read directly in the repository; there is no separate
documentation build.

## Runtime scope in version 0.2

The supported runtime uses `models/gemma3` for MedGemma, with shared generation,
cache and quantization infrastructure. General-model examples, plugin
installers and media-generation
pipelines were removed. The maintained commands are `generate`, `convert` and
`server`; unsupported architectures and checkpoint `model_file` code fail
explicitly. The real-checkpoint and synthetic numerical checks remain separate
from clinical model evaluation.
