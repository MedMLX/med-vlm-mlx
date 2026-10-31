"""Checkpoint manifest parsing invariants."""

import json
from dataclasses import asdict

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scripts.checkpoint_smoke import Checkpoint, CheckpointName, parse_checkpoints


@settings(max_examples=30, derandomize=True)
@given(st.integers(min_value=0, max_value=2**160 - 1))
def test_checkpoint_manifest_round_trip(revision: int) -> None:
    checkpoint = Checkpoint(
        CheckpointName.MEDGEMMA, "publisher/model", f"{revision:040x}"
    )
    payload = json.dumps({"schema_version": 1, "checkpoints": [asdict(checkpoint)]})
    assert parse_checkpoints(payload) == (checkpoint,)


@settings(max_examples=30, derandomize=True)
@given(st.lists(st.integers(min_value=0, max_value=127), max_size=80))
def test_malformed_manifest_has_only_declared_parse_errors(
    codepoints: list[int],
) -> None:
    text = "".join(chr(codepoint) for codepoint in codepoints)
    try:
        parsed = parse_checkpoints(text)
    except (ValueError, TypeError):
        return
    assert all(isinstance(entry, Checkpoint) for entry in parsed)


def test_checkpoint_manifest_rejects_duplicate_names() -> None:
    row = {"name": "medgemma", "repo_id": "publisher/model", "revision": "a" * 40}
    with pytest.raises(ValueError, match="unique"):
        parse_checkpoints(json.dumps({"schema_version": 1, "checkpoints": [row, row]}))


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version":1,"schema_version":1,"checkpoints":[]}',
        '{"schema_version":true,"checkpoints":[]}',
        '{"schema_version":1,"checkpoints":NaN}',
    ],
)
def test_checkpoint_manifest_rejects_ambiguous_json(payload: str) -> None:
    with pytest.raises(ValueError):
        parse_checkpoints(payload)
