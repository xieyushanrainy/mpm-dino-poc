from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FAMILIES = ("rigid", "fluid", "soft_body")


def primary_category(metadata: dict) -> str:
    categories = [str(x.get("material_category", "unknown")) for x in metadata.get("vlm", {}).get("materials", [])]
    if not categories:
        return "unknown"
    counts = Counter(categories)
    return sorted(counts, key=lambda x: (-counts[x], x))[0]


def body_parameter(metadata: dict, key: str) -> float:
    bodies = metadata.get("simulation", {}).get("body_parameters", [])
    values = [float(body[key]) for body in bodies if body.get(key) is not None]
    if not values:
        raise ValueError(f"{metadata.get('uid', '<unknown>')} has no {key}")
    # The packaged soft set is homogeneous, but use an explicit mean if a file
    # contains more than one retained body entry.
    return float(np.mean(values))


@dataclass(frozen=True)
class ObjectRecord:
    uid: str
    family: str
    category: str
    sample_path: Path
    metadata_path: Path
    log10_e: float | None
    nu: float | None


def load_records(dataset: str | Path) -> dict[str, ObjectRecord]:
    dataset = Path(dataset)
    index = json.loads((dataset / "dataset.json").read_text())
    records: dict[str, ObjectRecord] = {}
    for item in index["objects"]:
        metadata_path = dataset / item["metadata"]
        metadata = json.loads(metadata_path.read_text())
        family = str(item.get("solver_route", metadata["simulation"]["solver_route"]))
        log10_e = nu = None
        if family == "soft_body":
            log10_e = float(np.log10(body_parameter(metadata, "youngs_modulus_pa")))
            nu = body_parameter(metadata, "poisson_ratio")
        uid = str(item["uid"])
        records[uid] = ObjectRecord(
            uid=uid,
            family=family,
            category=primary_category(metadata),
            sample_path=dataset / item["sample"],
            metadata_path=metadata_path,
            log10_e=log10_e,
            nu=nu,
        )
    return records


def load_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    split_sets = [set(manifest["splits"][name]) for name in ("train", "validation", "test")]
    if any(a & b for i, a in enumerate(split_sets) for b in split_sets[i + 1 :]):
        raise ValueError("manifest contains UID leakage between splits")
    return manifest


def pooled_dino(sample_path: Path, key: str = "dinov2_reprojected_features") -> tuple[np.ndarray, float]:
    valid_key = key.replace("features", "valid")
    with np.load(sample_path) as sample:
        if key not in sample:
            raise KeyError(f"{sample_path} has no {key}")
        features = sample[key].astype(np.float32)
        valid = sample[valid_key].astype(bool)
    valid &= np.isfinite(features).all(axis=1)
    if not valid.any():
        raise ValueError(f"{sample_path} has no valid DINO rows")
    observed = features[valid]
    return np.concatenate([observed.mean(0), observed.max(0)]).astype(np.float32), float(valid.mean())


def geometry_summary(sample_path: Path) -> np.ndarray:
    with np.load(sample_path) as sample:
        xyz = sample["trajectory_positions_m"][0].astype(np.float32)
        active = sample["point_active"][0].astype(bool)
    xyz = xyz[active]
    return np.concatenate([xyz.mean(0), xyz.std(0), xyz.min(0), xyz.max(0)]).astype(np.float32)


def make_feature(record: ObjectRecord, source: str, dino_key: str) -> np.ndarray:
    if source == "dino":
        return pooled_dino(record.sample_path, dino_key)[0]
    if source == "geometry":
        return geometry_summary(record.sample_path)
    if source == "valid_fraction":
        return np.asarray([pooled_dino(record.sample_path, dino_key)[1]], dtype=np.float32)
    raise ValueError(f"unknown feature source: {source}")


def audit_dataset(dataset: str | Path, manifest_path: str | Path, dino_key: str) -> dict:
    records, manifest = load_records(dataset), load_manifest(manifest_path)
    report: dict = {"objects": len(records), "splits": {}, "families": {}, "soft_targets": {}}
    valid_fractions = []
    for record in records.values():
        _, fraction = pooled_dino(record.sample_path, dino_key)
        valid_fractions.append(fraction)
    for split, uids in manifest["splits"].items():
        report["splits"][split] = dict(Counter(records[uid].family for uid in uids))
    for family in FAMILIES:
        report["families"][family] = sum(r.family == family for r in records.values())
    soft = [r for r in records.values() if r.family == "soft_body"]
    for name, values in {
        "log10_E": [r.log10_e for r in soft],
        "nu": [r.nu for r in soft],
    }.items():
        array = np.asarray(values, dtype=np.float64)
        report["soft_targets"][name] = {
            "count": len(array), "unique": len(np.unique(array)),
            "min": float(array.min()), "max": float(array.max()), "median": float(np.median(array)),
        }
    report["dino_valid_fraction"] = {
        "min": float(np.min(valid_fractions)), "median": float(np.median(valid_fractions)), "max": float(np.max(valid_fractions))
    }
    return report
