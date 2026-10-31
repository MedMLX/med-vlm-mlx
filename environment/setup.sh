#!/usr/bin/env bash
# Conda owns Python; uv.lock owns the medical package and development tools.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TARGET="macos"
TARGET_SET=0
HOST_CHECK=0
PRINT_TARGET=0
RUN_COMMAND=()

usage() {
  cat <<'EOF'
Usage: bash environment/setup.sh [OPTIONS]

Options:
  --target macos          Select the Apple Silicon target.
  --host-check            Print host/target details without mutation.
  --print-target          Print the effective target without mutation.
  --run COMMAND [ARG ...] Run a command in the existing environment.
  -h, --help              Show this help.
EOF
}

die() {
  printf '[setup] ERROR: %s\n' "$1" >&2
  exit 1
}

parse_args() {
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --target)
        [[ "$#" -ge 2 && "$2" != -* ]] || die "--target requires a value."
        [[ "$TARGET_SET" -eq 0 ]] || die "--target may be provided only once."
        TARGET="$2"; TARGET_SET=1; shift 2
        ;;
      --target=*)
        [[ "$TARGET_SET" -eq 0 ]] || die "--target may be provided only once."
        TARGET="${1#--target=}"; TARGET_SET=1; shift
        ;;
      --host-check) HOST_CHECK=1; shift ;;
      --print-target) PRINT_TARGET=1; shift ;;
      --run)
        shift
        [[ "$#" -gt 0 ]] || die "--run requires a command."
        RUN_COMMAND=("$@")
        break
        ;;
      -h | --help) usage; exit 0 ;;
      *) usage >&2; die "Unknown argument: $1" ;;
    esac
  done
  if [[ "$HOST_CHECK" -eq 1 && "$PRINT_TARGET" -eq 1 ]] || \
    [[ "${#RUN_COMMAND[@]}" -gt 0 && "$((HOST_CHECK + PRINT_TARGET))" -gt 0 ]]; then
    die "Choose only one of --host-check, --print-target, or --run."
  fi
}

validate_host() {
  [[ "$TARGET" == macos ]] || die "Unsupported target '$TARGET'; only macos is supported."
  [[ "$(uname -s):$(uname -m)" == Darwin:arm64 ]] ||
    die "med-vlm-mlx requires macOS on Apple Silicon (Darwin arm64); found $(uname -s) $(uname -m)."
}

read_env_name() {
  local count line
  count="$(awk '/^[[:space:]]*name[[:space:]]*:/ {count++} END {print count+0}' "$ENV_FILE")"
  [[ "$count" -eq 1 ]] || die "Expected exactly one canonical name: entry in $ENV_FILE."
  line="$(grep -E '^[[:space:]]*name[[:space:]]*:' "$ENV_FILE")"
  [[ "$line" =~ ^name:[[:space:]]*([A-Za-z0-9][A-Za-z0-9._-]*)[[:space:]]*$ ]] ||
    die "Environment name must be an unquoted [A-Za-z0-9][A-Za-z0-9._-]* token."
  [[ "${BASH_REMATCH[1]}" != base ]] || die "Environment name 'base' is not allowed."
  printf '%s\n' "${BASH_REMATCH[1]}"
}

validate_repo_contract() {
  local file requirement
  for file in pyproject.toml uv.lock devtools/repo-paths.sh; do
    [[ -f "${REPO_ROOT}/$file" ]] || die "Missing $file; restore the repository setup files."
  done
  for requirement in "${SCRIPT_DIR}"/*/requirements.txt; do
    [[ ! -f "$requirement" ]] ||
      die "Remove duplicated $requirement; pyproject.toml and uv.lock own Python dependencies."
  done
  command -v uv >/dev/null 2>&1 || die "Install uv before running setup."
  command -v node >/dev/null 2>&1 || die "Install maintained host Node for Pyright."
  (cd -- "$REPO_ROOT"; uv lock --check)
}

select_manager() {
  if command -v mamba >/dev/null 2>&1; then
    printf 'mamba\n'
  elif command -v conda >/dev/null 2>&1; then
    printf 'conda\n'
  else
    die "Install mamba or conda before running setup."
  fi
}

environment_exists() {
  local environments
  environments="$("$MANAGER" env list)" ||
    die "Could not list $MANAGER environments; refusing to create or update '$ENV_NAME'."
  awk '{print $1}' <<<"$environments" | grep -Fxq "$ENV_NAME"
}

create_or_update_env() {
  if environment_exists; then
    CONDA_ALWAYS_YES=true MAMBA_ALWAYS_YES=true \
      "$MANAGER" env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
  else
    "$MANAGER" env create -n "$ENV_NAME" -f "$ENV_FILE" -y
  fi
}

sync_python_project() {
  local prefix
  prefix="$("${ENV_RUN[@]}" python -c 'import sys; print(sys.prefix)')"
  [[ -n "$prefix" && -d "$prefix" ]] || die "Could not resolve the prefix for '$ENV_NAME'."
  # --inexact retains conda-owned packages outside uv's dependency graph.
  (cd -- "$REPO_ROOT"; UV_PROJECT_ENVIRONMENT="$prefix" \
    uv sync --locked --inexact --extra medical --group dev)
}

parse_args "$@"
validate_host
if [[ "$PRINT_TARGET" -eq 1 ]]; then
  printf '%s\n' "$TARGET"
  exit 0
fi
if [[ "$HOST_CHECK" -eq 1 ]]; then
  printf 'host_os=%s\nhost_arch=%s\neffective_target=%s\n' "$(uname -s)" "$(uname -m)" "$TARGET"
  exit 0
fi

ENV_FILE="${SCRIPT_DIR}/${TARGET}/environment.yml"
[[ -f "$ENV_FILE" ]] || die "Missing environment spec: $ENV_FILE"
ENV_NAME="$(read_env_name)"
validate_repo_contract
MANAGER="$(select_manager)"
ENV_RUN=("$MANAGER" run -n "$ENV_NAME")
if [[ "$MANAGER" == conda ]]; then
  ENV_RUN+=(--no-capture-output --)
fi

if [[ "${#RUN_COMMAND[@]}" -gt 0 ]]; then
  environment_exists || die "Environment '$ENV_NAME' does not exist; run 'make env' first."
  exec "${ENV_RUN[@]}" "${RUN_COMMAND[@]}"
fi

printf '[setup] target=%s env=%s\n' "$TARGET" "$ENV_NAME"
create_or_update_env
sync_python_project
printf '[setup] ready: target=%s env=%s\n' "$TARGET" "$ENV_NAME"
