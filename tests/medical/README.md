The vision parity tests read the chest X-ray at
`~/.cache/medmlx/test-data/chest_xray.jpg` and compare tiny seeded models with
the installed Transformers implementation. Run
`MLX_ENABLE_TF32=0 python -m pytest tests/medical/test_regressions.py` in the
project environment.
