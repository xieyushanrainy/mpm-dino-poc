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
from mpm_dino_v4.v43_causal_contact import (
    CAUSAL_VARIANTS, train_causal_contact_variant,
)


def main(args):
    manifest = json.loads(Path(args.manifest).read_text())
    root = Path(args.runs)
    root.mkdir(parents=True, exist_ok=True)
    variants = tuple(args.variants)
    sources, oracle = {}, {}
    for seed in args.seeds:
        gate1e = Path(args.gate1e_root) / f"seed{seed}" / "best_total.pt"
        champion = Path(args.oracle_root) / "adapter_full" / f"seed{seed}" / "best.pt"
        complete = Path(args.oracle_root) / "adapter_full" / f"seed{seed}" / "RUN_COMPLETE.json"
        if not gate1e.exists():
            raise FileNotFoundError(f"missing Gate-1E seed {seed}: {gate1e}")
        if not champion.exists() or not complete.exists():
            raise FileNotFoundError(f"missing oracle ceiling seed {seed}: {champion}")
        ceiling = json.loads(complete.read_text())
        champion_sha = file_sha256(champion)
        champion_state = torch.load(
            champion, map_location="cpu", weights_only=False,
        )
        champion_config = champion_state["config"]
        if (
            champion_config.get("condition_name") != "adapter_full"
            or champion_config.get("oracle_injection") != "adapter"
            or champion_config.get("model_contract_version")
            != "direct_point_decoder_probe_v1"
            or int(champion_config.get("seed", -1)) != seed
            or ceiling["best_checkpoint_sha256"] != champion_sha
        ):
            raise ValueError(f"invalid oracle ceiling contract for seed {seed}")
        sources[str(seed)] = {
            "path": str(gate1e), "sha256": file_sha256(gate1e),
        }
        oracle[str(seed)] = {
            "path": str(champion), "sha256": champion_sha,
            "best_epoch": ceiling["best_epoch"],
            "best_selection": ceiling["best_selection"],
        }
    config = {
        "experiment": "v43_causal_contact_oracle_replacement",
        "analysis": "matched_three_arm_diagnostic_with_frozen_oracle_ceiling",
        "variants": list(variants),
        "oracle_ceiling": oracle,
        "gate1e_sources": sources,
        "condition_contract": {
            "channels": 15,
            "identical_all_arms": [
                "four static curvature proxies",
                "seven zero discrete-stage channels for causal arms",
            ],
            "static_control": "curvature only; contact/time zero",
            "causal_timing_only": "curvature plus predicted relative event time",
            "causal_continuous": (
                "curvature plus rigid-proxy signed gap, smooth proximity, "
                "normal velocity, and predicted relative event time"
            ),
            "contact_threshold_radius_fraction": args.contact_threshold,
        },
        "causal_contract": {
            "allowed": [
                "x0/x1", "persistent input mask", "gravity/dt", "floor",
                "frozen Gate-1E COM", "frozen Gate-1E rotation",
            ],
            "forbidden": [
                "future target positions", "future target mask",
                "ground-truth stage", "ground-truth contact/time",
            ],
        },
        "training_population": "soft_body_only",
        "training_frames": ["contact_onset", "compression", "peak_deformation"],
        "objective": "per_episode_amplitude_normalized_canonical_mse",
        "selection": "validation_normalized_objective",
        "test_data_used": False,
        "manifest_content_sha256": manifest["manifest_content_sha256"],
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
            "oracle_ceiling", "variants", "condition_contract",
            "causal_contract", "objective",
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
            train_causal_contact_variant(
                args.dataset, manifest, sources[str(seed)]["path"],
                output, seed, variant,
                contact_threshold=args.contact_threshold,
                device=args.device, epochs=args.epochs,
                draws_per_epoch=args.draws, lr=args.lr,
                accumulation=args.accumulation, patience=args.patience,
                plateau_patience=args.plateau_patience,
                max_batches=args.max_batches,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replace V4.2 oracle contact/time with causal rigid-proxy features"
    )
    parser.add_argument("--dataset", default="v41/dataset")
    parser.add_argument("--manifest", default="v41/manifests/v41_uid_splits.json")
    parser.add_argument("--gate1e-root", required=True)
    parser.add_argument("--oracle-root", required=True)
    parser.add_argument(
        "--runs", default="v43/runs/causal_contact_seed42_456",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 456])
    parser.add_argument(
        "--variants", nargs="+", choices=CAUSAL_VARIANTS,
        default=list(CAUSAL_VARIANTS),
    )
    parser.add_argument("--contact-threshold", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--draws", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--max-batches", type=int)
    main(parser.parse_args())
