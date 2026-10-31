# Installation

Use macOS on Apple Silicon with Metal and Python 3.13.

## Use the package

Follow the [README quick start](../README.md#quick-start) to install the reviewed
source in an isolated uv environment. With that environment active, run package
commands from any working directory:

```bash
python -m mlx_vlm --help
```

The distribution is named `med-vlm-mlx` and imports as `mlx_vlm`. Keep upstream
`mlx-vlm` in a separate environment because both distributions install the same
modules. The [usage guide](usage.md) covers Python, CLI and HTTP inference.

## Develop from source

The checkout's development setup requires mamba or conda, uv, and maintained
host Node for Pyright. From a new checkout:

```bash
git clone https://github.com/MedMLX/med-vlm-mlx.git
cd med-vlm-mlx
make host-check
make env
```

Setup creates or updates the `med-vlm-mlx` conda environment with Python 3.13,
installs this checkout editably, and syncs the medical extra and development
tools from `uv.lock`. Mamba is preferred; conda is supported when mamba is absent.
The lock is checked before the environment is changed.

Run contributor commands from the repository root without activating the
environment:

```bash
bash environment/setup.sh --run python -m mlx_vlm --help
```

This runner requires the existing named environment and a current lock; it does
not install or update dependencies. Prefix package commands from the usage guide
with `bash environment/setup.sh --run` when using this contributor environment.
See [development and validation](technical.md) for local checks.

Version 0.2 retains the medical runtime and removes the general `eval` extra,
media generation and Gradio interface.
