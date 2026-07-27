#!/usr/bin/env python3
"""Run the exact V4 Track B pooled-DINO architecture on the V4.1 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import time
from pathlib import Path

import torch

from mpm_dino_v4.v41_train import train_v41


MECHANISM = "track_b_pooled"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(args, manifest):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "experiment": "v41_exact_v4_track_b_pooled_dino_film",
        "started_unix": time.time(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else None
        ),
        "git_commit": commit,
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "resolved_arguments": vars(args),
        "architecture": {
            "source": "v4 Track B FullTrajectorySurrogate",
            "dino_path": "[N,384] -> [N,16] -> masked mean/max + valid fraction -> width-128 condition",
            "injection": "blockwise FiLM",
            "physical_backbone": "width 128, four graph-temporal blocks, four heads",
        },
        "loss_profile": args.loss_profile,
        "loss": (
            {
                "implementation": "compute_full_trajectory_loss",
                "normalization": "world_metres",
                "weights": [1.0, 1.0, 0.5, 0.25, 0.1, 0.25],
                "terms": [
                    "residual", "position", "com", "edge_vector",
                    "edge_length", "key_horizons",
                ],
                "key_horizons": [4, 8, 16, 59],
            }
            if args.loss_profile == "legacy"
            else {
                "implementation": "compute_shape_balanced_trajectory_loss",
                "normalization": "per_object_fixed_reference_radius",
                "weights": [1.0, 0.5, 1.0, 0.5, 0.25],
                "terms": [
                    "world", "com", "shape", "strain", "key_horizons",
                ],
                "key_horizons": [16, 30, 40],
            }
        ),
    }


def train_if_needed(path, root, manifest, mode, seed, args, zero_reference=None):
    if (path / "RUN_COMPLETE.json").exists():
        print(f"skip complete: {path}", flush=True)
        return path / "best.pt"
    path.mkdir(parents=True, exist_ok=True)
    (path / "RUNNING.json").write_text(json.dumps({
        "status": "running",
        "started_unix": time.time(),
        "mechanism": MECHANISM,
        "dino_mode": mode,
        "seed": seed,
    }, indent=2) + "\n")
    try:
        result = train_v41(
            root=root,
            manifest=manifest,
            output=path,
            mechanism=MECHANISM,
            dino_mode=mode,
            seed=seed,
            device=args.device,
            epochs=args.epochs,
            draws_per_epoch=args.draws,
            patience=args.patience,
            plateau_patience=args.plateau_patience,
            amp=args.amp,
            resume=True,
            zero_reference=zero_reference,
            loss_profile=args.loss_profile,
        )
    except BaseException as exc:
        (path / "RUN_FAILED.json").write_text(json.dumps({
            "status": "failed_or_interrupted",
            "time_unix": time.time(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "resume_command": "rerun the identical matrix command",
        }, indent=2) + "\n")
        raise
    finally:
        (path / "RUNNING.json").unlink(missing_ok=True)
    (path / "RUN_FAILED.json").unlink(missing_ok=True)
    return result


def run(args):
    manifest = json.loads(Path(args.manifest).read_text())
    root, runs = Path(args.dataset), Path(args.runs)
    runs.mkdir(parents=True, exist_ok=True)
    current = provenance(args, manifest)
    matrix_config = runs / "MATRIX_CONFIG.json"
    if matrix_config.exists():
        previous = json.loads(matrix_config.read_text())
        keys = (
            "epochs", "draws", "patience", "plateau_patience", "seeds",
            "device", "amp", "loss_profile",
        )
        changed = [
            key for key in keys
            if previous["resolved_arguments"].get(key)
            != current["resolved_arguments"].get(key)
        ]
        if changed:
            raise ValueError(
                f"refusing to mix changed matrix settings: {changed}"
            )
        if previous["manifest_content_sha256"] != manifest["manifest_content_sha256"]:
            raise ValueError("refusing to mix a changed V4.1 manifest")
    else:
        matrix_config.write_text(json.dumps(current, indent=2) + "\n")

    for seed in args.seeds:
        zero = runs / "zero" / f"seed{seed}"
        real = runs / "real" / f"seed{seed}"
        zero_best = train_if_needed(
            zero, root, manifest, "zero", seed, args,
        )
        if not zero_best.exists():
            raise RuntimeError(
                f"matched zero seed {seed} has no guarded best checkpoint; "
                "real-DINO H1 guard cannot be defined"
            )
        train_if_needed(
            real, root, manifest, "real", seed, args,
            zero_reference=zero_best,
        )

    records = []
    for marker in sorted(runs.glob("**/RUN_COMPLETE.json")):
        records.append({
            "run": str(marker.parent.relative_to(runs)),
            "completion": json.loads(marker.read_text()),
            "config_sha256": sha256(marker.parent / "config.json"),
        })
    (runs / "MATRIX_COMPLETE.json").write_text(json.dumps({
        "status": "complete" if len(records) == 6 else "incomplete",
        "completed_unix": time.time(),
        "scientific_runs": len(records),
        "runs": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="v41/dataset")
    parser.add_argument(
        "--manifest", default="v41/manifests/v41_uid_splits.json",
    )
    parser.add_argument(
        "--runs",
        default="v41/runs/track_b_pooled_cap150_p30_fp32",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456],
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--draws", type=int, default=40)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=False,
    )
    parser.add_argument(
        "--loss-profile",
        choices=("legacy", "shape_balanced_v1"),
        default="legacy",
    )
    run(parser.parse_args())
