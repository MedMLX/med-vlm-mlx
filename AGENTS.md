# med-vlm-mlx

Focused MLX runtime for MedGemma 1.5 4B, MedGemma 4B v1 and MedGemma 27B
multimodal (`gemma3`), shipped in the `mlx_vlm` namespace. MIT, with Apache-2.0 adaptations.

## Rules

- Allow `Any` at untyped third-party library boundaries; keep package interfaces typed.

- Model execution runs in MLX on Metal (macOS arm64). There is no PyTorch,
  MPS, NumPy or CPU fallback; host I/O and preprocessing may use CPU
  libraries.
- Unsupported checkpoints, options and inputs raise a clear error. Do not add
  compatibility shims or silent fallbacks.
- Never commit model weights, medical images, patient data or generated
  outputs.
- Keep upstream notices in THIRD_PARTY_NOTICES.md; model weights retain their
  publisher terms.
- Keep the scope to MedGemma and the shared inference code it uses;
  do not reintroduce unrelated architectures, plugin installers or
  media-generation routes.
- Select the native processor directly by architecture and preserve every
  accepted processor option.
- Keep `uv.lock` on Pillow 13 and the patched CA bundle.
- Real-checkpoint validation records the weight revision, backend, precision
  and fixture provenance; it does not establish clinical accuracy.

## Checks

- `make env` creates the `med-vlm-mlx` conda environment from
  `environment/macos/environment.yml`; it needs mamba or conda, uv and Node.
- Run commands with `bash environment/setup.sh --run COMMAND`; this never
  installs dependencies.
- `make qa` runs formatting, Pyright and host tests; `make test
  TESTS=tests/test_name.py` runs selected tests.
- Medical inference changes need the affected local tests on Metal. There is
  no hosted CI.
