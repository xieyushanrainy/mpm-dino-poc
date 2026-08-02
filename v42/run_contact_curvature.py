#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import socket
import time
from pathlib import Path

import torch

from mpm_dino_v4.v42_contact_curvature import (
    VARIANTS, train_contact_curvature_variant,
)
from mpm_dino_v4.v42_gate2 import file_sha256


def main(args):
    manifest = json.loads(Path(args.manifest).read_text())
    root = Path(args.runs)
    root.mkdir(parents=True, exist_ok=True)
    variants = tuple(args.variants)
    sources = {
        str(seed): {
            "path": str(Path(args.gate1e_root) / f"seed{seed}" / "best_total.pt")
        } for seed in args.seeds
    }
    for seed, source in sources.items():
        path = Path(source["path"])
        if not path.exists():
            raise FileNotFoundError(f"missing Gate-1E seed {seed}: {path}")
        source["sha256"] = file_sha256(path)
    config = {
        "experiment": "v42_contact_curvature_control",
        "analysis": "four_arm_mechanism_diagnostic_not_a_gate",
        "variants": list(variants),
        "condition_contract": {
            "shape": "[batch, frame, point, 7]",
            "contact_channels": [
                "oracle_signed_floor_gap", "oracle_floor_proximity",
                "oracle_vertical_velocity",
            ],
            "curvature_channels": [
                "reference_linearity", "reference_planarity",
                "reference_scattering", "reference_normal_variation",
            ],
            "injection": "additive_projection_before_region_adapter",
            "matched_architecture": True,
        },
        "oracle_warning": (
            "Contact is computed from future ground-truth positions and is "
            "not available at inference; this tests information sufficiency."
        ),
        "training_population": "soft_body_only",
        "training_frames": [
            "contact_onset", "compression", "peak_deformation",
        ],
        "objective": "per_episode_amplitude_normalized_canonical_mse",
        "selection": "validation_normalized_objective",
        "test_data_used": False,
        "real_dino_trained": False,
        "gate_decision": None,
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "gate1e_sources": sources,
        "arguments": vars(args),
        "started_unix": time.time(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    matrix = root / "MATRIX_CONFIG.json"
    if matrix.exists():
        prior = json.loads(matrix.read_text())
        for key in (
            "arguments", "manifest_content_sha256", "gate1e_sources",
            "variants", "condition_contract", "objective",
        ):
            if prior[key] != config[key]:
                raise ValueError(f"refusing changed matrix {key}")
    else:
        matrix.write_text(json.dumps(config, indent=2) + "\n")
    for variant in variants:
        for seed in args.seeds:
            output = root / variant / f"seed{seed}"
            if (output / "RUN_COMPLETE.json").exists():
                print(f"skip complete: {output}", flush=True)
                continue
            train_contact_curvature_variant(
                args.dataset, manifest, sources[str(seed)]["path"],
                output, seed, variant,
                device=args.device, epochs=args.epochs,
                draws_per_epoch=args.draws, lr=args.lr,
                accumulation=args.accumulation, patience=args.patience,
                plateau_patience=args.plateau_patience,
                max_batches=args.max_batches,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Matched oracle-contact and curvature control experiment"
    )
    parser.add_argument("--dataset", default="v41/dataset")
    parser.add_argument("--manifest", default="v41/manifests/v41_uid_splits.json")
    parser.add_argument("--gate1e-root", required=True)
    parser.add_argument(
        "--runs", default="v42/runs/contact_curvature_seed42_456",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 456])
    parser.add_argument(
        "--variants", nargs="+", choices=tuple(VARIANTS),
        default=list(VARIANTS),
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--draws", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--max-batches", type=int)
    main(parser.parse_args())
