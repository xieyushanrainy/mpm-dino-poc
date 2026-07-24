#!/usr/bin/env python3
"""Run the reviewed V4.1 matched matrix, resuming completed run directories."""

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


def completed(path: Path) -> bool:
    return (path / "RUN_COMPLETE.json").exists()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(args, manifest):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "started_unix": time.time(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": commit,
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "resolved_arguments": vars(args),
    }


def train_if_needed(path, root, manifest, mechanism, mode, seed, args, **extra):
    if completed(path):
        print(f"skip complete: {path}", flush=True)
        return path / "best.pt"
    path.mkdir(parents=True, exist_ok=True)
    (path / "RUNNING.json").write_text(json.dumps({
        "status": "running", "started_unix": time.time(),
        "mechanism": mechanism, "dino_mode": mode, "seed": seed,
    }, indent=2) + "\n")
    try:
        result = train_v41(
            root, manifest, path, mechanism, mode, seed, args.device,
            args.epochs, args.draws, patience=args.patience,
            plateau_patience=args.plateau_patience, amp=args.amp, resume=True,
            **extra,
        )
    except BaseException as exc:
        (path / "RUN_FAILED.json").write_text(json.dumps({
            "status": "failed_or_interrupted", "time_unix": time.time(),
            "error_type": type(exc).__name__, "error": str(exc),
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
    matrix_config = provenance(args, manifest)
    config_path = runs / "MATRIX_CONFIG.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        keys = ("epochs", "draws", "patience", "plateau_patience", "seeds")
        if any(previous["resolved_arguments"].get(k) != matrix_config["resolved_arguments"].get(k) for k in keys):
            raise ValueError("refusing to mix changed training budgets in an existing matrix directory")
    else:
        config_path.write_text(json.dumps(matrix_config, indent=2, default=str) + "\n")
    for seed in args.seeds:
        for mechanism in ("m1", "m2"):
            zero = runs / mechanism / "zero" / f"seed{seed}"
            real = runs / mechanism / "real" / f"seed{seed}"
            train_if_needed(zero, root, manifest, mechanism, "zero", seed, args)
            train_if_needed(real, root, manifest, mechanism, "real", seed, args,
                            zero_reference=zero / "best.pt")
        stage1 = runs / "m6" / "stage1" / f"seed{seed}"
        train_if_needed(stage1, root, manifest, "none", "zero", seed, args)
        zero = runs / "m6" / "zero" / f"seed{seed}"
        real = runs / "m6" / "real" / f"seed{seed}"
        train_if_needed(zero, root, manifest, "m6", "zero", seed, args,
                        stage1_checkpoint=stage1 / "best.pt")
        train_if_needed(real, root, manifest, "m6", "real", seed, args,
                        stage1_checkpoint=stage1 / "best.pt",
                        zero_reference=zero / "best.pt")
    run_records = []
    for marker in sorted(runs.glob("**/RUN_COMPLETE.json")):
        run_records.append({
            "run": str(marker.parent.relative_to(runs)),
            "completion": json.loads(marker.read_text()),
            "config_sha256": sha256(marker.parent / "config.json"),
        })
    (runs / "MATRIX_COMPLETE.json").write_text(json.dumps({
        "status": "complete", "completed_unix": time.time(),
        "scientific_runs": len(run_records), "runs": run_records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="v41/dataset")
    parser.add_argument("--manifest", default="v41/manifests/v41_uid_splits.json")
    parser.add_argument("--runs", default="v41/runs/lab_plateau30")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument(
        "--epochs", type=int, default=1000,
        help="Safety ceiling; normal termination is plateau patience.",
    )
    parser.add_argument("--draws", type=int, default=40)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    run(parser.parse_args())
