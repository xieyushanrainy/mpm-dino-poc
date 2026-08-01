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
        "experiment": "v42_decoder_canonical_only_overfit",
        "diagnostic_order": "1B",
        "parent_diagnostic": "v42_decoder_learnability_overfit_single_frame_failed",
        "hypothesis": (
            "canonical-only optimization can fit the identical selected peak "
            "frame, isolating composite-loss conflict from decoder capacity"
        ),
        "changed_variable": "optimized scalar is canonical loss instead of composite local loss",
        "held_fixed": [
            "training-only strongest soft Panel-Z episode selection",
            "selected peak frame", "Gate-1E initialization",
            "geometry-only zero-DINO local pathway", "trainable prefixes",
            "AdamW learning rate and zero weight decay", "step budget",
            "95% canonical-error reduction threshold",
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
        "computed_but_not_optimized_loss_contract": LOSS_OPTIONS,
        "arguments": vars(args),
    }
    matrix = root / "MATRIX_CONFIG.json"
    if matrix.exists():
        prior = json.loads(matrix.read_text())
        for key in (
            "arguments", "manifest_content_sha256", "gate1e_source",
            "pass_thresholds", "changed_variable", "held_fixed",
        ):
            if prior[key] != contract[key]:
                raise ValueError(f"refusing changed canonical-overfit matrix {key}")
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
            log_every=args.log_every, objective="canonical_only",
        )
    summary = {
        "status": "complete",
        "canonical_only_single_frame_passed": result["passed"],
        "interpretation_if_passed": (
            "basic decoder capacity supported; composite-loss conflict remains "
            "the leading explanation and must be redesigned before temporal tests"
        ),
        "interpretation_if_failed": (
            "one-frame spatial representation/conditioning remains insufficient; "
            "audit decoder inputs and capacity before temporal tests"
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
        description="Canonical-only single-frame decoder capacity audit"
    )
    parser.add_argument("--dataset", default="v41/dataset")
    parser.add_argument("--manifest", default="v41/manifests/v41_uid_splits.json")
    parser.add_argument("--gate1e-root", required=True)
    parser.add_argument("--runs", default="v42/runs/canonical_overfit_seed42")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=25)
    main(parser.parse_args())
