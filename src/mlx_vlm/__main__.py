"""Command entry points for medical inference and checkpoint conversion."""

import importlib
import sys


def main() -> None:
    """Dispatch only maintained medical runtime commands."""
    commands = ("generate", "convert", "server")
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("Usage: python -m mlx_vlm {generate,convert,server} [options]")
        return
    command = sys.argv.pop(1)
    if command not in commands:
        raise ValueError(f"Unsupported command {command!r}; choose from {commands}")
    module = importlib.import_module(f"mlx_vlm.{command}")
    module.main()


if __name__ == "__main__":
    main()
