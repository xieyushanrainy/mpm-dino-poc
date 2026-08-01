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
    LOSS_OPTIONS, OVERFIT_MODES, PASS_THRESHOLDS, train_overfit_mode,
)


def main(args):
    manifest = json.loads(Path(args.manifest).read_text())
    source = Path(args.gate1e_root) / f"seed{args.seed}" / "best_total.pt"
    if not source.exists():
        raise FileNotFoundError(f"missing authoritative Gate-1E source: {source}")
    root = Path(args.runs)
    root.mkdir(parents=True, exist_ok=True)
    provenance = {
        "experiment": "v42_decoder_learnability_overfit",
        "diagnostic_order": 1,
        "purpose": "single-example representational and optimization capacity",
        "next_gate_if_passed": "oracle_temporal_conditioning",
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
        "modes": args.modes,
        "pass_thresholds": {mode: PASS_THRESHOLDS[mode] for mode in args.modes},
        "loss_contract": LOSS_OPTIONS,
        "arguments": vars(args),
    }
    matrix = root / "MATRIX_CONFIG.json"
    if matrix.exists():
        prior = json.loads(matrix.read_text())
        for key in (
            "arguments", "manifest_content_sha256", "gate1e_source", "modes",
            "pass_thresholds", "loss_contract",
        ):
            if prior[key] != provenance[key]:
                raise ValueError(f"refusing changed overfit matrix {key}")
    else:
        matrix.write_text(json.dumps(provenance, indent=2) + "\n")

    results = {}
    for mode in args.modes:
        output = root / mode
        completion = output / "RUN_COMPLETE.json"
        if completion.exists():
            result = json.loads((output / "OVERFIT_RESULT.json").read_text())
            print(f"skip complete: {output}", flush=True)
        else:
            result = train_overfit_mode(
                args.dataset, manifest, source, output, args.seed, mode,
                device=args.device, steps=args.steps, lr=args.lr,
                log_every=args.log_every,
            )
        results[mode] = result["passed"]
        # The full-episode result is uninterpretable if one-frame capacity fails.
        if mode == "single_frame" and not result["passed"]:
            print("single-frame gate failed; stopping before single-episode", flush=True)
            break
    summary = {
        "status": "complete", "results": results,
        "decoder_learnability_passed": bool(
            results.get("single_frame") and results.get("single_episode")
        ),
        "oracle_temporal_conditioning_authorized": bool(
            results.get("single_frame") and results.get("single_episode")
        ),
        "oracle_material_conditioning_authorized": False,
        "learned_dino_authorized": False,
    }
    (root / "DIAGNOSTIC_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training-only single-frame and single-episode decoder overfit audit"
    )
    parser.add_argument("--dataset", default="v41/dataset")
    parser.add_argument("--manifest", default="v41/manifests/v41_uid_splits.json")
    parser.add_argument("--gate1e-root", required=True)
    parser.add_argument("--runs", default="v42/runs/decoder_overfit_seed42")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes", nargs="+", default=list(OVERFIT_MODES), choices=OVERFIT_MODES,
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=25)
    main(parser.parse_args())
