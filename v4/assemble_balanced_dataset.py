#!/usr/bin/env python3
"""Assemble a standalone, balanced rigid/fluid/soft V4 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROUTES = ("rigid", "fluid", "soft_body")
REQUIRED_ARRAYS = {
    "trajectory_positions_m", "dino_features", "dino_valid",
    "point_material_ids", "point_active",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def primary_category(metadata: dict) -> str:
    materials = metadata.get("vlm", {}).get("materials", [])
    ranked = sorted(
        materials,
        key=lambda item: (-float(item.get("confidence", 0.0)), int(item.get("material_id", 0))),
    )
    return str(ranked[0].get("material_category", "unknown")) if ranked else "unknown"


def candidates(package: Path, route: str) -> list[dict]:
    dataset = read_json(package / "dataset.json")
    result = []
    for record in dataset.get("objects", []):
        if record.get("solver_route") != route:
            continue
        metadata_path = package / record["metadata"]
        metadata = read_json(metadata_path)
        result.append({
            "record": record,
            "metadata": metadata,
            "package": package,
            "category": primary_category(metadata),
            "confidence": float(record.get("minimum_vlm_confidence", 0.0)),
        })
    return result


def diverse_selection(items: list[dict], count: int) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        buckets[item["category"]].append(item)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: (-item["confidence"], item["record"]["uid"]))
    selected = []
    while len(selected) < count and any(buckets.values()):
        order = sorted(
            (category for category, bucket in buckets.items() if bucket),
            key=lambda category: (sum(x["category"] == category for x in selected), category),
        )
        for category in order:
            if len(selected) >= count:
                break
            selected.append(buckets[category].pop(0))
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} eligible samples exist; need {count}")
    return selected


def validate_sample(object_dir: Path, route: str) -> None:
    sample = object_dir / "sample.npz"
    metadata = read_json(object_dir / "metadata.json")
    with np.load(sample, allow_pickle=False) as arrays:
        if set(arrays.files) != REQUIRED_ARRAYS:
            raise ValueError(f"{object_dir.name}: unexpected arrays {sorted(arrays.files)}")
        if arrays["trajectory_positions_m"].shape != (61, 2048, 3):
            raise ValueError(f"{object_dir.name}: unexpected trajectory shape")
        if arrays["dino_features"].shape != (2048, 384):
            raise ValueError(f"{object_dir.name}: unexpected DINO shape")
    if metadata["simulation"]["solver_route"] != route:
        raise ValueError(f"{object_dir.name}: route mismatch")
    if metadata.get("force", {}).get("force_fields"):
        raise ValueError(f"{object_dir.name}: expected zero control-force fields")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-package", required=True, help="Package containing rigid and fluid samples")
    parser.add_argument("--soft-package", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-route", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main_package = Path(args.main_package).resolve()
    soft_package = Path(args.soft_package).resolve()
    output = Path(args.output_dir).resolve()
    if args.per_route <= 0:
        raise SystemExit("--per-route must be positive")
    if output in (main_package, soft_package):
        raise SystemExit("output must be separate from source packages")
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists: {output}; pass --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    source_for_route = {"rigid": main_package, "fluid": main_package, "soft_body": soft_package}
    selected = {
        route: diverse_selection(candidates(source_for_route[route], route), args.per_route)
        for route in ROUTES
    }
    records = []
    selection_audit = []
    for route in ROUTES:
        for item in selected[route]:
            record = dict(item["record"])
            uid = record["uid"]
            source_object = item["package"] / "objects" / uid
            target_object = output / "objects" / uid
            if target_object.exists():
                raise ValueError(f"duplicate UID across routes: {uid}")
            shutil.copytree(source_object, target_object)
            metadata_path = target_object / "metadata.json"
            metadata = read_json(metadata_path)
            metadata.setdefault("packaging", {})["balanced_assembly"] = {
                "source_package": item["package"].name,
                "source_uid": uid,
                "selected_route": route,
                "selection_material_category": item["category"],
            }
            write_json(metadata_path, metadata)
            validate_sample(target_object, route)
            record["sample"] = f"objects/{uid}/sample.npz"
            record["metadata"] = f"objects/{uid}/metadata.json"
            records.append(record)
            selection_audit.append({
                "uid": uid, "route": route, "material_category": item["category"],
                "minimum_vlm_confidence": item["confidence"],
                "source_package": item["package"].name,
            })
            print(f"[{len(records)}/{args.per_route * len(ROUTES)}] {route}: {uid}", flush=True)

    now = datetime.now(timezone.utc).isoformat()
    dataset = {
        "schema_version": 1,
        "name": "MPM-DINO V4 balanced rigid-fluid-soft dataset",
        "created_at_utc": now,
        "object_count": len(records),
        "routes": dict(Counter(record["solver_route"] for record in records)),
        "per_route": args.per_route,
        "selection_policy": {
            "rigid": "round-robin primary material category; confidence-descending within category",
            "fluid": "all eligible samples; deterministic category/confidence ordering",
            "soft_body": "all eligible samples; deterministic category/confidence ordering",
        },
        "source_packages": [main_package.name, soft_package.name],
        "objects": records,
    }
    write_json(output / "dataset.json", dataset)
    write_json(output / "SELECTION_AUDIT.json", selection_audit)
    (output / "README.md").write_text(
        "# MPM-DINO V4 balanced dataset\n\n"
        f"Standalone compact package with {args.per_route} rigid, {args.per_route} fluid, "
        f"and {args.per_route} homogeneous soft-body samples. All samples have 61 frames, "
        "2,048 persistent points, 384-dimensional DINO features, and zero control-force fields.\n",
        encoding="utf-8",
    )
    (output / "PACKAGE_COMPLETE").write_text("complete\n", encoding="utf-8")
    write_json(output / "PACKAGE_SHA256.json", {
        "dataset.json": sha256(output / "dataset.json"),
        "selection_audit.json": sha256(output / "SELECTION_AUDIT.json"),
    })
    print(json.dumps({"output": str(output), "count": len(records), "routes": dataset["routes"]}, indent=2))


if __name__ == "__main__":
    main()
