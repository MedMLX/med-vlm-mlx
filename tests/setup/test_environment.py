"""Setup process boundaries without creating or modifying real environments."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

ROOT = Path(__file__).resolve().parents[2]
FAKE_TOOL = """#!/bin/bash
set -eu
tool="$(basename "$0")"
if [[ "$tool" == uname ]]; then
  case "$1" in
    -s) printf '%s\\n' "${SETUP_TEST_OS:-Darwin}" ;;
    -m) printf '%s\\n' "${SETUP_TEST_ARCH:-arm64}" ;;
  esac
  exit 0
fi
printf '%s' "$tool" >> "$SETUP_TEST_LOG"
printf ' <%s>' "$@" >> "$SETUP_TEST_LOG"
printf ' | prefix=%s\\n' "${UV_PROJECT_ENVIRONMENT:-}" >> "$SETUP_TEST_LOG"
case "$tool:$1" in
  uv:lock) exit "${SETUP_TEST_LOCK_EXIT:-0}" ;;
  uv:sync) exit "${SETUP_TEST_SYNC_EXIT:-0}" ;;
  mamba:env | conda:env)
    if [[ "$2" == list ]]; then
      if [[ "${SETUP_TEST_EXISTS:-1}" == 1 ]]; then
        printf '%s /fake/prefix\\n' "${SETUP_TEST_NAME:-med-vlm-mlx}"
      fi
      exit "${SETUP_TEST_LIST_EXIT:-0}"
    fi
    if [[ "$2" == update ]]; then
      if [[ "$tool" == conda ]]; then
        [[ "${CONDA_ALWAYS_YES:-}" == true ]] || exit 90
        for arg in "$@"; do [[ "$arg" != -y ]] || exit 91; done
      else
        [[ "${MAMBA_ALWAYS_YES:-}" == true || " $* " == *" -y "* ]] || exit 92
      fi
    fi
    exit "${SETUP_TEST_CREATE_EXIT:-0}"
    ;;
  mamba:run | conda:run)
    shift 3
    if [[ "$tool" == conda ]]; then shift 2; fi
    if [[ "${1:-}:${2:-}" == python:-c ]]; then
      printf '%s\\n' "$SETUP_TEST_PREFIX"
    elif [[ "${1:-}:${2:-}" == python:-m || "${1:-}:${2:-}" == sh:-c ]]; then
      PATH="$SETUP_TEST_PREFIX/bin:$PATH" "$@"
    fi
    exit "${SETUP_TEST_RUN_EXIT:-0}"
    ;;
  python:-m) exit "${SETUP_TEST_RUN_EXIT:-0}" ;;
esac
"""


@dataclass(frozen=True)
class SetupCase:
    repo: Path
    env: dict[str, str]
    log: Path
    prefix: Path

    def run(
        self, *args: str, overrides: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Run the real entrypoint with isolated external executables."""
        return subprocess.run(
            ["/bin/bash", "environment/setup.sh", *args],
            cwd=self.repo,
            env=self.env | (overrides or {}),
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )


@pytest.fixture
def setup_case(tmp_path: Path) -> SetupCase:
    repo = tmp_path / "repo"
    for relative in (
        "environment/setup.sh",
        "environment/macos/environment.yml",
        "devtools/repo-paths.sh",
        "Makefile",
    ):
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    for name in ("pyproject.toml", "uv.lock"):
        (repo / name).touch()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("uname", "uv", "mamba", "node"):
        executable = bin_dir / name
        executable.write_text(FAKE_TOOL, encoding="utf-8")
        executable.chmod(0o755)
    prefix = tmp_path / "prefix with spaces"
    prefix.mkdir()
    (prefix / "bin").mkdir()
    python = prefix / "bin/python"
    python.write_text(FAKE_TOOL, encoding="utf-8")
    python.chmod(0o755)
    log = tmp_path / "commands.log"
    log.touch()
    env = os.environ | {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "SETUP_TEST_LOG": str(log),
        "SETUP_TEST_PREFIX": str(prefix),
    }
    return SetupCase(repo, env, log, prefix)


@pytest.mark.parametrize("exists,action", [("0", "create"), ("1", "update")])
@pytest.mark.parametrize("manager", ["mamba", "conda"])
def test_setup_syncs_locked_medical_dependencies_to_conda_prefix(
    setup_case: SetupCase, exists: str, action: str, manager: str
) -> None:
    if manager == "conda":
        bin_dir = Path(setup_case.env["PATH"].split(":")[0])
        (bin_dir / "mamba").rename(bin_dir / "conda")
    result = setup_case.run(overrides={"SETUP_TEST_EXISTS": exists})
    assert result.returncode == 0, result.stderr
    commands = setup_case.log.read_text()
    assert commands.startswith("uv <lock> <--check>")
    assert f"{manager} <env> <{action}> <-n> <med-vlm-mlx>" in commands
    assert (
        "uv <sync> <--locked> <--inexact> <--extra> <medical> <--group> <dev>"
        f" | prefix={setup_case.prefix}\n"
    ) in commands


@pytest.mark.parametrize(
    "overrides,expected,forbidden",
    [
        ({"SETUP_TEST_LOCK_EXIT": "17"}, 17, "mamba"),
        ({"SETUP_TEST_LIST_EXIT": "18"}, 1, "<create>"),
        ({"SETUP_TEST_CREATE_EXIT": "19"}, 19, "<sync>"),
        ({"SETUP_TEST_SYNC_EXIT": "20"}, 20, "[setup] ready"),
    ],
)
def test_setup_propagates_failures_without_followup_mutation(
    setup_case: SetupCase, overrides: dict[str, str], expected: int, forbidden: str
) -> None:
    result = setup_case.run(overrides=overrides)
    assert result.returncode == expected, result.stderr
    assert forbidden not in setup_case.log.read_text() + result.stdout
    assert "[setup] ready" not in result.stdout


@pytest.mark.parametrize("manager", ["mamba", "conda"])
def test_run_preserves_arguments_and_exit_status_without_sync(
    setup_case: SetupCase, manager: str
) -> None:
    if manager == "conda":
        bin_dir = Path(setup_case.env["PATH"].split(":")[0])
        (bin_dir / "mamba").rename(bin_dir / "conda")
    result = setup_case.run(
        "--run",
        "example",
        "argument with spaces",
        overrides={"SETUP_TEST_RUN_EXIT": "23"},
    )
    assert result.returncode == 23
    commands = setup_case.log.read_text()
    separator = " <--no-capture-output> <-->" if manager == "conda" else ""
    assert f"{manager} <run> <-n> <med-vlm-mlx>{separator}" in commands
    assert "<example> <argument with spaces>" in commands
    assert "<sync>" not in commands and "<create>" not in commands


def test_run_requires_existing_environment(setup_case: SetupCase) -> None:
    result = setup_case.run("--run", "python", overrides={"SETUP_TEST_EXISTS": "0"})
    assert result.returncode == 1
    assert "make env" in result.stderr
    commands = setup_case.log.read_text()
    assert "<create>" not in commands and "<sync>" not in commands


@pytest.mark.parametrize("mode", ["--host-check", "--print-target", "--run"])
@pytest.mark.parametrize("os_name,arch", [("Linux", "aarch64"), ("Darwin", "x86_64")])
def test_unsupported_host_fails_before_external_tools(
    setup_case: SetupCase, mode: str, os_name: str, arch: str
) -> None:
    args = (mode, "python") if mode == "--run" else (mode,)
    result = setup_case.run(
        *args, overrides={"SETUP_TEST_OS": os_name, "SETUP_TEST_ARCH": arch}
    )
    assert result.returncode == 1
    assert "Apple Silicon" in result.stderr
    assert setup_case.log.read_text() == ""


@pytest.mark.parametrize(
    "args",
    [
        ("--unknown",),
        ("--target",),
        ("--target", "linux"),
        ("--target=macos", "--target=macos"),
        ("--host-check", "--print-target"),
        ("--run",),
    ],
)
def test_invalid_arguments_fail_before_external_tools(
    setup_case: SetupCase, args: tuple[str, ...]
) -> None:
    assert setup_case.run(*args).returncode == 1
    assert setup_case.log.read_text() == ""


@pytest.mark.parametrize(
    "spec", ["", "name: base\n", "name: ../bad\n", "name: a\nname: b\n"]
)
def test_invalid_environment_names_fail_before_external_tools(
    setup_case: SetupCase, spec: str
) -> None:
    (setup_case.repo / "environment/macos/environment.yml").write_text(spec)
    assert setup_case.run().returncode == 1
    assert setup_case.log.read_text() == ""


def test_environment_name_round_trip(setup_case: SetupCase) -> None:
    # The subprocess timeout bounds hangs; scheduling is not a parser property.
    @settings(max_examples=20, derandomize=True, deadline=None)
    @given(
        st.from_regex(r"[a-z][a-z0-9._-]{0,20}", fullmatch=True).filter(
            lambda n: n != "base"
        )
    )
    def check(name: str) -> None:
        (setup_case.repo / "environment/macos/environment.yml").write_text(
            f"name: {name}\n"
        )
        result = setup_case.run("--run", "example", overrides={"SETUP_TEST_NAME": name})
        assert result.returncode == 0, result.stderr
        assert f"<run> <-n> <{name}> <example>" in setup_case.log.read_text()

    check()


def test_arbitrary_environment_names_do_not_crash(setup_case: SetupCase) -> None:
    @settings(max_examples=20, derandomize=True, deadline=None)
    @given(st.text(max_size=80))
    def check(name: str) -> None:
        (setup_case.repo / "environment/macos/environment.yml").write_text(
            f"name: {name}\n"
        )
        result = setup_case.run("--run", "example")
        assert result.returncode in (0, 1), result.stderr
        commands = setup_case.log.read_text()
        assert "<sync>" not in commands and "<create>" not in commands

    check()


@pytest.mark.parametrize("missing", ["pyproject.toml", "uv.lock"])
def test_partial_python_contract_fails_before_tools(
    setup_case: SetupCase, missing: str
) -> None:
    (setup_case.repo / missing).unlink()
    assert setup_case.run().returncode == 1
    assert setup_case.log.read_text() == ""


def test_target_requirements_fail_before_tools(setup_case: SetupCase) -> None:
    (setup_case.repo / "environment/macos/requirements.txt").touch()
    assert setup_case.run().returncode == 1
    assert setup_case.log.read_text() == ""


def test_make_qa_runs_only_through_existing_environment(setup_case: SetupCase) -> None:
    result = subprocess.run(
        ["/usr/bin/make", "qa"],
        cwd=setup_case.repo,
        env=setup_case.env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    commands = setup_case.log.read_text()
    for module in ("black", "isort", "ruff", "pytest"):
        assert f"mamba <run> <-n> <med-vlm-mlx> <python> <-m> <{module}>" in commands
    assert any(
        line.startswith("mamba <run> <-n> <med-vlm-mlx>") and "pyright" in line
        for line in commands.splitlines()
    )
    assert (
        f"python <-m> <pyright> <--project> <.> <--warnings>"
        f" <--pythonpath> <{setup_case.prefix / 'bin/python'}>"
    ) in commands
    assert "<sync>" not in commands and "<create>" not in commands
