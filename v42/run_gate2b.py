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


REVIEWED_VARIANTS = {
    "balanced_x1": 1.0,
    "balanced_x5": 5.0,
    "balanced_x20": 20.0,
}


def loss_options(cap):
    return {
        "soft_deformation_amplification_cap": float(cap),
        "soft_deformation_quantile": 0.95,
        "soft_deformation_floor_fraction": 0.005,
        "family_balanced": True,
        "rigid_family_weight": 0.25,
        "rigid_zero_weight": 0.0,
    }


def main(args):
    unknown = set(args.variants) - REVIEWED_VARIANTS.keys()
    if unknown:
        raise ValueError(f"unknown Gate-2B variants: {sorted(unknown)}")
    if args.draws % 2:
        raise ValueError("Gate-2B requires an even draw count for exact family balance")
    manifest = json.loads(Path(args.manifest).read_text())
    root = Path(args.runs)
    root.mkdir(parents=True, exist_ok=True)
    sources = {
        str(seed): {
            "path": str(Path(args.gate1e_root) / f"seed{seed}" / "best_total.pt")
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
    variant_contracts = {
        variant: loss_options(REVIEWED_VARIANTS[variant])
        for variant in args.variants
    }
    provenance = {
        "experiment": "v42_gate2b_family_balanced_deformation_scaled",
        "gate": "2B",
        "parent_gate2_result": "failed_frozen_screen",
        "gate3_or_dino_training": False,
        "test_data_used": False,
        "started_unix": time.time(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "gate1e_sources": sources,
        "variant_contracts": variant_contracts,
        "arguments": vars(args),
        "frozen_evaluation_screen": "v42_gate2_gate2_screen",
    }
    matrix = root / "MATRIX_CONFIG.json"
    if matrix.exists():
        prior = json.loads(matrix.read_text())
        for key in (
            "arguments", "manifest_content_sha256", "gate1e_sources",
            "variant_contracts", "frozen_evaluation_screen",
        ):
            if prior[key] != provenance[key]:
                raise ValueError(f"refusing changed Gate-2B matrix {key}")
    else:
        matrix.write_text(json.dumps(provenance, indent=2) + "\n")
    for variant in args.variants:
        for seed in args.seeds:
            output = root / variant / f"seed{seed}"
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
                loss_options=variant_contracts[variant],
                experiment_name=(
                    "v42_gate2b_family_balanced_deformation_scaled_" + variant
                ),
                model_contract_version="gate2b_geometry_only_v1",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Gate 2B only: family-balanced geometry controls with detached "
            "per-episode soft-deformation scaling"
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
    parser.add_argument("--runs", default="v42/runs/gate2b_seed42_456")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 456])
    parser.add_argument(
        "--variants", nargs="+", default=list(REVIEWED_VARIANTS),
        choices=list(REVIEWED_VARIANTS),
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--draws", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--max-batches", type=int)
    main(parser.parse_args())
