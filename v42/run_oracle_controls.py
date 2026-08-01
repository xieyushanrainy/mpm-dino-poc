#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import socket
import time
from pathlib import Path

import torch

from mpm_dino_v4.v42_gate2 import file_sha256
from mpm_dino_v4.v42_oracle import (
    LOSS_CONTRACT, ORACLE_DIM, ORACLE_VARIANTS,
    summarize_controlled_matrix, train_oracle_variant,
)


def main(args):
    if args.draws % 2:
        raise ValueError("exact family balance requires an even draw count")
    unknown = set(args.variants) - set(ORACLE_VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    manifest = json.loads(Path(args.manifest).read_text())
    root = Path(args.runs)
    root.mkdir(parents=True, exist_ok=True)
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
        "experiment": "v42_oracle_temporal_material_controlled_group",
        "analysis": "exploratory_2x2_factorial_not_a_gate",
        "hypothesis": (
            "separate missing temporal state and material ambiguity from "
            "decoder/representation/optimization limitations"
        ),
        "variants": {key: list(value) for key, value in ORACLE_VARIANTS.items()},
        "selected_variants": args.variants,
        "oracle_condition_dim": ORACLE_DIM,
        "loss_contract": LOSS_CONTRACT,
        "stage_weighting": "total_mass",
        "test_data_used": False,
        "dino_trained": False,
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
            "selected_variants", "loss_contract", "oracle_condition_dim",
        ):
            if prior[key] != config[key]:
                raise ValueError(f"refusing changed oracle matrix {key}")
    else:
        matrix.write_text(json.dumps(config, indent=2) + "\n")
    for variant in args.variants:
        for seed in args.seeds:
            output = root / variant / f"seed{seed}"
            if (output / "RUN_COMPLETE.json").exists():
                print(f"skip complete: {output}", flush=True)
                continue
            train_oracle_variant(
                args.dataset, manifest, sources[str(seed)]["path"],
                output, seed, variant,
                device=args.device, epochs=args.epochs,
                draws_per_epoch=args.draws, lr=args.lr,
                accumulation=args.accumulation, patience=args.patience,
                plateau_patience=args.plateau_patience,
                max_batches=args.max_batches,
            )
    path = summarize_controlled_matrix(root, args.variants, args.seeds)
    if path:
        print(f"controlled effect report: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Matched 2x2 oracle temporal/material controlled experiment"
    )
    parser.add_argument("--dataset", default="v41/dataset")
    parser.add_argument("--manifest", default="v41/manifests/v41_uid_splits.json")
    parser.add_argument("--gate1e-root", required=True)
    parser.add_argument("--runs", default="v42/runs/oracle_controls_seed42_456")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 456])
    parser.add_argument(
        "--variants", nargs="+", choices=list(ORACLE_VARIANTS),
        default=list(ORACLE_VARIANTS),
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--draws", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--max-batches", type=int)
    main(parser.parse_args())
