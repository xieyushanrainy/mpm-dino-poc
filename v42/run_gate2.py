#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import socket
import time
from pathlib import Path

import torch

from mpm_dino_v4.v42_gate2 import file_sha256, train_v42_gate2


def main(args):
    manifest = json.loads(Path(args.manifest).read_text())
    root = Path(args.runs)
    root.mkdir(parents=True, exist_ok=True)
    sources = {
        str(seed): {
            "path": str(
                Path(args.gate1e_root) / f"seed{seed}" / "best_total.pt"
            )
        }
        for seed in args.seeds
    }
    for seed, source in sources.items():
        path = Path(source["path"])
        if not path.exists():
            raise FileNotFoundError(
                f"missing authoritative Gate-1E seed {seed} source: {path}"
            )
        source["sha256"] = file_sha256(path)
    provenance = {
        "experiment": "v42_gate2_geometry_only_canonical_local",
        "gate": 2,
        "gate3_or_dino_training": False,
        "started_unix": time.time(), "hostname": socket.gethostname(),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else None
        ),
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "gate1e_sources": sources, "arguments": vars(args),
    }
    matrix = root / "MATRIX_CONFIG.json"
    if matrix.exists():
        prior = json.loads(matrix.read_text())
        for key in (
            "arguments", "manifest_content_sha256", "gate1e_sources",
        ):
            if prior[key] != provenance[key]:
                raise ValueError(f"refusing changed Gate-2 matrix {key}")
    else:
        matrix.write_text(json.dumps(provenance, indent=2) + "\n")
    for seed in args.seeds:
        output = root / f"seed{seed}"
        if (output / "RUN_COMPLETE.json").exists():
            print(f"skip complete: {output}", flush=True)
            continue
        train_v42_gate2(
            args.dataset, manifest, sources[str(seed)]["path"], output, seed,
            device=args.device, epochs=args.epochs,
            draws_per_epoch=args.draws, lr=args.lr,
            accumulation=args.accumulation, patience=args.patience,
            plateau_patience=args.plateau_patience,
            max_batches=args.max_batches,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Gate 2 only: frozen Gate-1E zero-local baseline versus "
            "geometry-only zero-DINO canonical-local learning"
        )
    )
    parser.add_argument("--dataset", default="v41/dataset")
    parser.add_argument(
        "--manifest", default="v41/manifests/v41_uid_splits.json",
    )
    parser.add_argument(
        "--gate1e-root", required=True,
        help="root containing seed42/seed456/best_total.pt",
    )
    parser.add_argument("--runs", default="v42/runs/gate2_seed42_456")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 456])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--draws", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--max-batches", type=int)
    main(parser.parse_args())
