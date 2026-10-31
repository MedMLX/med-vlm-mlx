"""Run a pinned real-checkpoint integration check on explicit Metal hardware.

Usage: ``python scripts/checkpoint_smoke.py NAME`` with ``medgemma`` (MedGemma
1.5 4B), ``medgemma-4b-v1`` or ``medgemma-27b`` (multimodal). The script loads
the revision pinned in ``scripts/checkpoints.json`` (downloading it when absent)
with strict weight matching, downloads a public CC0 chest radiograph, asks for a
radiology description with greedy decoding, and prints a JSON record to standard
output. It checks that loading and image-to-text generation on a real radiograph
run and produce text; it does not score output quality or medical accuracy.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
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


# Normal PA chest radiograph by Mikael Häggström, CC0, Wikimedia Commons. It is
# downloaded per run into a temporary directory because medical images are never
# committed; see THIRD_PARTY_NOTICES.md.
RADIOGRAPH_PAGE = (
    "https://commons.wikimedia.org/wiki/"
    "File:Normal_posteroanterior_(PA)_chest_radiograph_(X-ray).jpg"
)
RADIOGRAPH_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/a/a1/"
    "Normal_posteroanterior_%28PA%29_chest_radiograph_%28X-ray%29.jpg"
)
RADIOGRAPH_PROMPT = "Describe this chest X-ray."
# Wikimedia rejects requests without a descriptive User-Agent.
USER_AGENT = "med-vlm-mlx checkpoint smoke (https://github.com/MedMLX/med-vlm-mlx)"


def download_radiograph(directory: Path) -> Path:
    """Download the public radiograph into ``directory`` and return its path."""
    import requests

    response = requests.get(
        RADIOGRAPH_URL, headers={"User-Agent": USER_AGENT}, timeout=30
    )
    response.raise_for_status()
    if not response.headers.get("content-type", "").startswith("image/"):
        raise RuntimeError("The radiograph download did not return an image")
    path = directory / "chest-pa.jpg"
    path.write_bytes(response.content)
    return path


def run_checkpoint(checkpoint: Checkpoint) -> None:
    """Describe the public chest radiograph and print a JSON record.

    Uses greedy decoding, at most 128 new tokens and a prefill step of 64. The
    record holds the model, revision, device, image source, prompt, token counts
    and text.

    Raises:
        RuntimeError: If Metal is unavailable, the response is not an image,
            or generation produces no text. Network errors propagate from requests.
        TypeError: If the loaded model or processor lacks a required interface.
    """
    import mlx.core as mx

    from mlx_vlm.contracts import is_json_object
    from mlx_vlm.generate import generate
    from mlx_vlm.generate.types import ProcessorLike
    from mlx_vlm.models.base import BaseModelConfig
    from mlx_vlm.models.protocols import require_generation_model
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load

    if not mx.metal.is_available():
        raise RuntimeError("Checkpoint smoke tests require MLX Metal")
    mx.set_default_device(mx.gpu)
    model, processor = load(
        checkpoint.repo_id, revision=checkpoint.revision, strict=True
    )
    model = require_generation_model(model)
    config = model.config
    if not isinstance(config, BaseModelConfig) and not is_json_object(config):
        raise TypeError("The checkpoint model must provide a model configuration")
    prompt = apply_chat_template(processor, config, RADIOGRAPH_PROMPT, num_images=1)
    if not isinstance(prompt, str):
        raise TypeError("The checkpoint chat template must produce a string")
    if not hasattr(processor, "tokenizer") or not hasattr(processor, "detokenizer"):
        raise TypeError(
            "The checkpoint processor must provide a tokenizer and detokenizer"
        )
    with tempfile.TemporaryDirectory() as directory:
        image_path = download_radiograph(Path(directory))
        result = generate(
            model,
            cast(ProcessorLike, processor),
            prompt,
            image=[str(image_path)],
            max_tokens=128,
            temperature=0.0,
            prefill_step_size=64,
        )
    if not result.text.strip() or result.generation_tokens < 1:
        raise RuntimeError("Checkpoint produced no text")
    evidence = {
        "schema_version": 1,
        "model": checkpoint.repo_id,
        "revision": checkpoint.revision,
        "backend": str(mx.default_device()),
        "device": mx.device_info(),
        "input": RADIOGRAPH_PAGE,
        "input_kind": "public CC0 patient chest radiograph; normal PA view",
        "prompt": RADIOGRAPH_PROMPT,
        "text": result.text,
        "generation_tokens": result.generation_tokens,
        "prompt_tokens": result.prompt_tokens,
        "claim": "Strict loading and public image-to-text inference; medical accuracy unqualified",
    }
    print(json.dumps(evidence, indent=2, allow_nan=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        names = "|".join(CheckpointName)
        raise SystemExit(f"Usage: python scripts/checkpoint_smoke.py {{{names}}}")
    selected = CheckpointName(sys.argv[1])
    manifest = Path(__file__).with_name("checkpoints.json")
    checkpoint = next(
        entry
        for entry in parse_checkpoints(manifest.read_text())
        if entry.name == selected
    )
    run_checkpoint(checkpoint)
