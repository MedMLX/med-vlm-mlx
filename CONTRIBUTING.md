# Contributing to med-vlm-mlx

This repository maintains Gemma3 inference for MedGemma 1.5 4B, 4B v1 and
27B multimodal on Apple Silicon. Keep changes within this architecture or the
shared loading, generation, cache, adapter and HTTP infrastructure. Scope decisions are in
[AGENTS.md](AGENTS.md); [technical notes](docs/technical.md) describe validation.

Use `make env` for setup, then `make qa` for formatting, Ruff, Pyright in its
configured standard mode, and host/setup checks. Inference changes also require
the affected Metal tests:

```sh
make test TESTS=tests/medical/test_regressions.py
```

Use `make smoke-medgemma` when checkpoint loading or
processor integration changes. Keep checkpoint revisions explicit and record
new reports outside the checkout. Retained engineering measurements establish
only the behavior and tolerances they actually exercise.

Checkpoint conversion must preserve existing native weights and be idempotent:
converting already converted tensors must leave their values unchanged. Cover
meaningful serialization round trips and malformed inputs with property tests;
exercise the resulting graph on Metal. Keep pretrained adapter inference
separate from training utilities.

Preserve upstream attribution and model-publisher terms. Contributions use the
repository's MIT license. Pull requests should explain the changed behavior,
the checks run and any remaining limitations.
