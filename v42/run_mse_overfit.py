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
from mpm_dino_v4.v42_overfit import (
    LOSS_OPTIONS, PASS_THRESHOLDS, train_overfit_mode,
)


def main(args):
    manifest = json.loads(Path(args.manifest).read_text())
    source = Path(args.gate1e_root) / f"seed{args.seed}" / "best_total.pt"
    if not source.exists():
        raise FileNotFoundError(f"missing authoritative Gate-1E source: {source}")
    root = Path(args.runs)
    root.mkdir(parents=True, exist_ok=True)
    contract = {
        "experiment": "v42_decoder_composite_canonical_mse_overfit",
        "diagnostic_order": "1C",
        "parent_diagnostics": [
            "v42_decoder_learnability_overfit_single_frame_failed",
            "v42_decoder_canonical_only_overfit_single_frame_failed",
        ],
        "hypothesis": (
            "restoring the composite objective while aligning its canonical "
            "term with canonical NRMSE improves one-frame spatial fitting"
        ),
        "changed_variable": (
            "canonical Smooth-L1/Huber is replaced by radius-normalized "
            "pointwise MSE inside the restored composite objective"
        ),
        "optimized_objective": {
            "canonical": "radius_normalized_pointwise_mse",
            "strain": 0.50,
            "edge_length": 0.25,
            "local_velocity": 0.25,
            "rigid_zero": 0.0,
        },
        "held_fixed": [
            "training-only strongest soft Panel-Z episode selection",
            "selected peak frame", "Gate-1E initialization",
            "geometry-only zero-DINO local pathway", "trainable prefixes",
            "AdamW learning rate and zero weight decay", "step budget",
            "95% strict canonical-error reduction threshold",
            "90-110% predicted/target magnitude threshold",
        ],
        "gate3_or_dino_training": False,
        "test_data_used": False,
        "validation_data_used": False,
        "started_unix": time.time(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "gate1e_source": {"path": str(source), "sha256": file_sha256(source)},
        "pass_thresholds": PASS_THRESHOLDS["single_frame"],
        "auxiliary_loss_contract": LOSS_OPTIONS,
        "arguments": vars(args),
    }
    matrix = root / "MATRIX_CONFIG.json"
    if matrix.exists():
        prior = json.loads(matrix.read_text())
        for key in (
            "arguments", "manifest_content_sha256", "gate1e_source",
            "pass_thresholds", "changed_variable", "optimized_objective",
            "held_fixed",
        ):
            if prior[key] != contract[key]:
                raise ValueError(f"refusing changed MSE-overfit matrix {key}")
    else:
        matrix.write_text(json.dumps(contract, indent=2) + "\n")

    output = root / "single_frame"
    if (output / "RUN_COMPLETE.json").exists():
        result = json.loads((output / "OVERFIT_RESULT.json").read_text())
        print(f"skip complete: {output}", flush=True)
    else:
        result = train_overfit_mode(
            args.dataset, manifest, source, output, args.seed, "single_frame",
            device=args.device, steps=args.steps, lr=args.lr,
            log_every=args.log_every, objective="composite_canonical_mse",
        )
    summary = {
        "status": "complete",
        "strict_near_exact_gate_passed": result["passed"],
        "strict_gate_note": (
            "95% reduction remains a near-exact diagnostic, not the sole "
            "criterion for useful non-collapse capability"
        ),
        "single_episode_authorized": False,
        "oracle_temporal_conditioning_authorized": False,
        "oracle_material_conditioning_authorized": False,
        "learned_dino_authorized": False,
    }
    (root / "DIAGNOSTIC_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Composite-loss single-frame audit with canonical normalized MSE"
    )
    parser.add_argument("--dataset", default="v41/dataset")
    parser.add_argument("--manifest", default="v41/manifests/v41_uid_splits.json")
    parser.add_argument("--gate1e-root", required=True)
    parser.add_argument("--runs", default="v42/runs/mse_overfit_seed42")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=25)
    main(parser.parse_args())
