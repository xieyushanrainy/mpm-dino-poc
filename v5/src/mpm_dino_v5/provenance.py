from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


INFERENCE_KEYS = frozenset({
    "x0", "x1", "input_mask", "reference", "dt", "gravity", "floor_z",
})
FORBIDDEN_QUERY_KEYS = frozenset({
    "target", "target_mask", "future_positions", "contact_target",
    "stage_labels", "event_time_target", "canonical_target", "family",
    "material_id", "material_labels", "point_material_ids", "solver_route",
})


def validate_inference_keys(keys: Iterable[str]) -> None:
    keys = set(keys)
    forbidden = keys & FORBIDDEN_QUERY_KEYS
    if forbidden:
        raise ValueError(f"target-derived or forbidden query inputs: {sorted(forbidden)}")
    unknown = keys - INFERENCE_KEYS
    if unknown:
        raise ValueError(f"undeclared V5 inference inputs: {sorted(unknown)}")


def canonical_json_sha256(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split_manifest(manifest: dict, expected_content_sha256: str) -> None:
    if manifest.get("manifest_content_sha256") != expected_content_sha256:
        raise ValueError("split manifest content hash mismatch")
    groups = [set(manifest["splits"][name]["uids"]) for name in ("train", "validation", "test")]
    if [len(group) for group in groups] != [40, 10, 10]:
        raise ValueError("V5 requires 40/10/10 UID splits")
    if any(left & right for i, left in enumerate(groups) for right in groups[i + 1:]):
        raise ValueError("UID leakage across splits")

