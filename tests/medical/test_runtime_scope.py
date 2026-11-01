"""Checkpoint Python is never executed."""

from pathlib import Path

import pytest

from mlx_vlm.utils import get_model_and_args


def test_checkpoint_python_is_never_executed(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    (tmp_path / "custom.py").write_text(
        "from pathlib import Path\n" + f"Path({str(marker)!r}).touch()\n"
    )
    with pytest.raises(ValueError):
        get_model_and_args(
            {"model_type": "gemma3", "model_file": "custom.py"}, tmp_path
        )
    assert not marker.exists()
