#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mpm_dino_v4.v42_geometry_audit import geometry_learnability_audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit cross-object geometry/deformation learnability"
    )
    parser.add_argument("--dataset", default="v41/dataset")
    parser.add_argument("--manifest", default="v41/manifests/v41_uid_splits.json")
    parser.add_argument(
        "--output", default="v42/run/geometry_learnability_audit",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nearest-k", type=int, default=3)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    report = geometry_learnability_audit(
        args.dataset, manifest, args.output, args.device, args.nearest_k,
    )
    print(json.dumps(report["summary"], indent=2))
