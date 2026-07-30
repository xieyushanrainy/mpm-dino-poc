#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import socket
import time
from pathlib import Path

import torch

from mpm_dino_v4.v42_gate1d_train import train_v42_gate1d


def main(args):
    manifest = json.loads(Path(args.manifest).read_text())
    root = Path(args.runs)
    root.mkdir(parents=True, exist_ok=True)
    provenance = {
        "experiment": "v42_gate1f_contact_rotation_matrix",
        "started_unix": time.time(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else None
        ),
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "arguments": vars(args),
    }
    matrix = root / "MATRIX_CONFIG.json"
    if matrix.exists():
        prior = json.loads(matrix.read_text())
        if prior["arguments"] != provenance["arguments"]:
            raise ValueError("refusing to mix changed Gate-1F arguments")
        if (
            prior["manifest_content_sha256"]
            != provenance["manifest_content_sha256"]
        ):
            raise ValueError("refusing to mix a changed manifest")
    else:
        matrix.write_text(json.dumps(provenance, indent=2) + "\n")

    for variant in args.variants:
        for seed in args.seeds:
            source = Path(args.gate1b_root) / f"seed{seed}" / "best.pt"
            if not source.exists():
                raise FileNotFoundError(f"missing Gate-1B source: {source}")
            output = root / variant / f"seed{seed}"
            if (output / "RUN_COMPLETE.json").exists():
                print(f"skip complete: {output}", flush=True)
                continue
            train_v42_gate1d(
                args.dataset, manifest, source, output, seed,
                device=args.device, epochs=args.epochs,
                draws_per_epoch=args.draws, lr=args.lr,
                accumulation=args.accumulation, patience=args.patience,
                plateau_patience=args.plateau_patience,
                min_eligible_epoch=args.min_eligible_epoch,
                max_batches=args.max_batches,
                contact_rotation_mode=variant,
                angular_damping=args.angular_damping,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="v41/dataset")
    parser.add_argument(
        "--manifest", default="v41/manifests/v41_uid_splits.json",
    )
    parser.add_argument(
        "--gate1b-root", default="v42/run/gate1b_seed42_456",
    )
    parser.add_argument("--runs", default="v42/runs/gate1f_seed42_456")
    parser.add_argument(
        "--variants", nargs="+", choices=("absolute", "impulse"),
        default=("absolute", "impulse"),
    )
    parser.add_argument("--angular-damping", type=float, default=0.95)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 456))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--draws", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--min-eligible-epoch", type=int, default=60)
    parser.add_argument("--max-batches", type=int)
    main(parser.parse_args())
