.DEFAULT_GOAL := qa
TESTS ?= tests/medical
CHECKPOINT ?= medgemma
PYTEST_ARGS ?=

ENV_ARGS ?=
ENV_RUN := bash environment/setup.sh $(ENV_ARGS) --run
PRESET := base
SOURCE_DIRS ?= mlx_vlm scripts tests typings environment devtools
GENERATED_PATHS ?= .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
export PYTHONNOUSERSITE := 1
FORMATTED = $(SOURCE_DIRS)

.PHONY: env host-check check-env lint loc clean qa test test-medical-host test-setup typecheck format-check smoke-medgemma
qa: lint typecheck test-medical-host test-setup

env:
	bash environment/setup.sh $(ENV_ARGS)

host-check:
	bash environment/setup.sh $(ENV_ARGS) --host-check

check-env:
	$(ENV_RUN) python --version

lint: format-check
	$(ENV_RUN) python -m ruff check --ignore-noqa $(SOURCE_DIRS)

format-check:
	$(ENV_RUN) python -m black --check $(FORMATTED)
	$(ENV_RUN) python -m isort --check-only $(FORMATTED)

typecheck:
	@command -v node >/dev/null || { echo "Maintained host Node is required for Pyright" >&2; exit 1; }
	PYRIGHT_PYTHON_NODEJS_WHEEL=0 PYRIGHT_PYTHON_GLOBAL_NODE=1 $(ENV_RUN) sh -c 'python -m pyright --project . --warnings --pythonpath "$$(command -v python)"'

test-medical-host:
	$(ENV_RUN) python -m pytest -q tests/medical/test_manifest.py

test-setup:
	$(ENV_RUN) python -m pytest -q tests/setup

test:
	MLX_ENABLE_TF32=0 $(ENV_RUN) python -m pytest -q $(TESTS) $(PYTEST_ARGS)

smoke-medgemma:
	MLX_ENABLE_TF32=0 $(ENV_RUN) python scripts/checkpoint_smoke.py $(CHECKPOINT)

loc:
	@set -eu; . ./devtools/repo-paths.sh; \
	validate_repo_paths SOURCE_DIRS $(SOURCE_DIRS); \
	list=$$(mktemp); trap 'rm -f "$$list"' EXIT; \
	for dir in $(SOURCE_DIRS); do \
		if [ -e "$$dir" ]; then find "./$$dir" -type f \
			\( -name '*.py' -o -name '*.pyi' -o -name '*.sh' -o -name '*.ts' -o -name '*.tsx' -o -name '*.js' -o -name '*.jsx' -o -name '*.rs' \) -print0 >> "$$list"; fi; \
	done; test -s "$$list" || { echo "No source files found"; exit 1; }; xargs -0 wc -l < "$$list"

clean:
	@set -eu; . ./devtools/repo-paths.sh; \
	validate_repo_paths GENERATED_PATHS $(GENERATED_PATHS); \
	validate_repo_paths SOURCE_DIRS $(SOURCE_DIRS); \
	for path in $(GENERATED_PATHS); do \
		rm -rf -- "$$path"; \
	done; for dir in $(SOURCE_DIRS); do \
		if [ -d "$$dir" ]; then find "./$$dir" -type d -name __pycache__ -prune -exec rm -rf {} +; fi; \
	done
