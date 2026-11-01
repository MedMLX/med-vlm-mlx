"""Pinned checkpoint manifest (``scripts/checkpoints.json``) parsing and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import NoReturn, cast


class CheckpointName(StrEnum):
    """Supported pinned checkpoints, selectable by their command-line name."""

    MEDGEMMA = "medgemma"
    MEDGEMMA_4B_V1 = "medgemma-4b-v1"
    MEDGEMMA_27B = "medgemma-27b"


@dataclass(frozen=True)
class Checkpoint:
    """A pinned Hugging Face checkpoint: ``owner/model`` ID and full commit hash."""

    name: CheckpointName
    repo_id: str
    revision: str

    def __post_init__(self) -> None:
        """Reject a name, ``owner/model`` id or full lowercase commit that is invalid."""
        name: object = getattr(self, "name")
        repo_id: object = getattr(self, "repo_id")
        revision: object = getattr(self, "revision")
        if not isinstance(name, CheckpointName):
            raise ValueError("Checkpoint name must be a CheckpointName")
        if (
            not isinstance(repo_id, str)
            or repo_id.count("/") != 1
            or not all(repo_id.split("/"))
        ):
            raise ValueError("Checkpoint repo_id must be an owner/model identifier")
        if not isinstance(revision, str) or len(revision) != 40:
            raise ValueError("Checkpoint revision must be a full 40-character commit")
        if set(revision) - set("0123456789abcdef"):
            raise ValueError(
                "Checkpoint revision must contain lowercase hexadecimal digits"
            )


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object from key/value pairs, rejecting duplicate keys."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def reject_nonfinite(value: str) -> NoReturn:
    """Reject the non-standard JSON constants ``NaN`` and ``Infinity``."""
    raise ValueError(f"Nonfinite JSON value: {value}")


def parse_checkpoints(text: str) -> tuple[Checkpoint, ...]:
    """Parse ``checkpoints.json`` into validated entries.

    The manifest must be a JSON object with exactly ``schema_version`` (the
    integer 1) and a nonempty ``checkpoints`` list. Each entry has exactly
    ``name``, ``repo_id`` and ``revision``, and names are unique.

    Raises:
        ValueError: On malformed JSON, duplicate or non-finite JSON values, or
            any violation of the manifest layout above.
    """
    decoded: object = json.loads(
        text, object_pairs_hook=unique_object, parse_constant=reject_nonfinite
    )
    if not isinstance(decoded, dict):
        raise ValueError("Checkpoint manifest requires schema_version=1")
    # The JSON decoder and unique_object construct string-keyed dictionaries.
    raw = cast(dict[str, object], decoded)
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ValueError("Checkpoint manifest requires schema_version=1")
    if set(raw) != {"schema_version", "checkpoints"}:
        raise ValueError(
            "Checkpoint manifest fields must be schema_version and checkpoints"
        )
    rows = raw.get("checkpoints")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Checkpoint manifest requires a nonempty checkpoints list")
    entries: list[Checkpoint] = []
    for entry in cast(list[object], rows):
        if not isinstance(entry, dict):
            raise ValueError("Checkpoint fields must be name, repo_id and revision")
        row = cast(dict[str, object], entry)
        if set(row) != {"name", "repo_id", "revision"}:
            raise ValueError("Checkpoint fields must be name, repo_id and revision")
        name, repo_id, revision = row["name"], row["repo_id"], row["revision"]
        if (
            not isinstance(name, str)
            or not isinstance(repo_id, str)
            or not isinstance(revision, str)
        ):
            raise ValueError("Checkpoint fields must be strings")
        entries.append(Checkpoint(CheckpointName(name), repo_id, revision))
    if len({entry.name for entry in entries}) != len(entries):
        raise ValueError("Checkpoint names must be unique")
    return tuple(entries)
