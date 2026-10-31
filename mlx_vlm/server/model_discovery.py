"""Inspect local model candidates without importing checkpoint code or weights."""

from pathlib import Path
from typing import Iterable, TypedDict

from huggingface_hub import HFCacheInfo
from pydantic import TypeAdapter

from mlx_vlm.contracts import JSONObject


class DiscoveredModel(TypedDict):
    """Local candidate identity and path, before conversion to a wire payload."""

    id: str
    path: Path
    created: int


MODEL_PATHS_ENV = "MLX_VLM_MODEL_PATHS"


def read_json_object(path: Path) -> JSONObject:
    """Read a JSON object, returning an empty mapping when unreadable or invalid."""
    try:
        value = TypeAdapter(JSONObject).validate_json(
            path.read_text(encoding="utf-8"), strict=True
        )
    except (OSError, ValueError):
        return {}
    return value


def _weight_status(directory: Path) -> bool | None:
    """Return weight validity, or None when no weight files or indexes exist."""

    def present(path: Path) -> bool:
        return path.is_file() and path.stat().st_size > 0

    shards: set[Path] = set()
    for index in directory.glob("*.safetensors.index.json"):
        weight_map = read_json_object(index).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        for filename in weight_map.values():
            if not isinstance(filename, str):
                return False
            shards.add(Path(filename))
    if shards:
        return all(
            not shard.is_absolute()
            and ".." not in shard.parts
            and shard.suffix == ".safetensors"
            and present(directory / shard)
            for shard in shards
        )
    weights = list(directory.glob("*.safetensors"))
    if not weights:
        return None
    return any(
        present(path)
        for path in weights
        if path.name not in {"adapter_model.safetensors", "consolidated.safetensors"}
    )


def is_model_directory(directory: Path) -> bool:
    """Check medical architecture metadata and nonempty weight availability.

    This is a discovery filter, not checkpoint validation: it does not parse
    weight files, verify tensor names or test whether the model can be loaded.
    For an index, every declared shard must be a nonempty relative safetensors
    path without parent traversal; without an index, any qualifying nonempty
    safetensors file is sufficient.
    """
    try:
        config = read_json_object(directory / "config.json")
        return (
            config.get("model_type") == "gemma3" and _weight_status(directory) is True
        )
    except (OSError, RuntimeError):
        return False


def discover_models(
    cache_info: HFCacheInfo | None, paths: Iterable[str] = ()
) -> list[DiscoveredModel]:
    """Discover HF snapshots and custom model folders or their immediate children.

    Prefer main, then the newest usable revision. Deduplicate by resolved path,
    also used as the ID for non-main snapshots and custom models.
    """
    models: dict[Path, DiscoveredModel] = {}

    def add(
        path: str | Path, *, model_id: str | None = None, created: float | None = None
    ) -> bool:
        try:
            path = Path(path).expanduser().resolve()
            if path in models or not is_model_directory(path):
                return False
            models[path] = {
                "id": str(path if model_id is None else model_id),
                "path": path,
                "created": int(path.stat().st_mtime if created is None else created),
            }
            return True
        except (OSError, RuntimeError):
            return False

    for repo in sorted(
        cache_info.repos if cache_info is not None else (), key=lambda r: r.repo_id
    ):
        if repo.repo_type != "model":
            continue
        main = repo.refs.get("main")
        for revision in sorted(
            repo.revisions,
            key=lambda r: (r != main, -r.last_modified, str(r.snapshot_path)),
        ):
            if add(
                revision.snapshot_path,
                model_id=repo.repo_id if revision == main else None,
                created=revision.last_modified,
            ):
                break

    for value in paths:
        try:
            root = Path(value).expanduser()
            # A model directory must not have its components listed separately.
            if (root / "config.json").exists() or (root / "model_index.json").exists():
                candidates = [root]
            else:
                candidates = sorted(p for p in root.iterdir() if p.is_dir())
        except (OSError, RuntimeError):
            continue
        for path in candidates:
            add(path)
    return sorted(models.values(), key=lambda model: model["id"].lower())
