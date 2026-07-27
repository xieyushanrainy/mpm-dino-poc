#!/usr/bin/env python3
"""Run authorized shuffled controls for the promoted split-region mechanism."""

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


MECHANISM = "split_region"
LOSS_PROFILE = "legacy_shape_aux_v1"
MODES = ("point_shuffled", "scene_shuffled")


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
        "experiment": "v41_split_region_authorized_shuffled_controls",
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
        "mechanism": MECHANISM,
        "loss_profile": LOSS_PROFILE,
        "dino_modes": list(MODES),
        "control_interpretation": {
            "point_shuffled": "tests point-to-DINO alignment",
            "scene_shuffled": "tests object/scene-level DINO identity",
        },
    }


def train_if_needed(path, root, manifest, mode, seed, args, zero_reference):
    if (path / "RUN_COMPLETE.json").exists():
        print(f"skip complete: {path}", flush=True)
        return
    path.mkdir(parents=True, exist_ok=True)
    (path / "RUNNING.json").write_text(json.dumps({
        "status": "running",
        "started_unix": time.time(),
        "mechanism": MECHANISM,
        "dino_mode": mode,
        "seed": seed,
    }, indent=2) + "\n")
    try:
        train_v41(
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
            loss_profile=LOSS_PROFILE,
        )
    except BaseException as exc:
        (path / "RUN_FAILED.json").write_text(json.dumps({
            "status": "failed_or_interrupted",
            "time_unix": time.time(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "resume_command": "rerun the identical shuffled-control command",
        }, indent=2) + "\n")
        raise
    finally:
        (path / "RUNNING.json").unlink(missing_ok=True)
    (path / "RUN_FAILED.json").unlink(missing_ok=True)


def run(args):
    manifest = json.loads(Path(args.manifest).read_text())
    promoted = Path(args.promoted_runs)
    output = Path(args.runs)
    output.mkdir(parents=True, exist_ok=True)
    current = provenance(args, manifest)
    matrix_config = output / "MATRIX_CONFIG.json"
    if matrix_config.exists():
        previous = json.loads(matrix_config.read_text())
        keys = (
            "epochs", "draws", "patience", "plateau_patience", "seeds",
            "device", "amp", "promoted_runs",
        )
        changed = [
            key for key in keys
            if previous["resolved_arguments"].get(key)
            != current["resolved_arguments"].get(key)
        ]
        if changed:
            raise ValueError(
                f"refusing to mix changed control settings: {changed}"
            )
        if (
            previous["manifest_content_sha256"]
            != manifest["manifest_content_sha256"]
        ):
            raise ValueError("refusing to mix a changed V4.1 manifest")
    else:
        matrix_config.write_text(json.dumps(current, indent=2) + "\n")

    for seed in args.seeds:
        zero_run = promoted / "zero" / f"seed{seed}"
        zero_best = zero_run / "best.pt"
        zero_config_path = zero_run / "config.json"
        if not zero_best.exists() or not zero_config_path.exists():
            raise FileNotFoundError(
                f"missing promoted-matrix zero reference for seed {seed}"
            )
        expected_initial_hash = json.loads(
            zero_config_path.read_text()
        )["starting_model_sha256"]
        for mode in MODES:
            run_path = output / mode / f"seed{seed}"
            train_if_needed(
                run_path, Path(args.dataset), manifest, mode, seed, args,
                zero_best,
            )
            control_config = json.loads(
                (run_path / "config.json").read_text()
            )
            if (
                control_config["starting_model_sha256"]
                != expected_initial_hash
            ):
                raise RuntimeError(
                    f"{mode} seed {seed} initialization differs from "
                    "the promoted real/zero pair"
                )

    records = []
    for marker in sorted(output.glob("**/RUN_COMPLETE.json")):
        records.append({
            "run": str(marker.parent.relative_to(output)),
            "completion": json.loads(marker.read_text()),
            "config_sha256": sha256(marker.parent / "config.json"),
        })
    expected = len(args.seeds) * len(MODES)
    (output / "MATRIX_COMPLETE.json").write_text(json.dumps({
        "status": "complete" if len(records) == expected else "incomplete",
        "completed_unix": time.time(),
        "expected_scientific_runs": expected,
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
        "--promoted-runs",
        default="v41/runs/track_b_split_region_cap100_p15_fp32",
    )
    parser.add_argument(
        "--runs",
        default="v41/runs/track_b_split_region_shuffled_cap100_p15_fp32",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456],
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--draws", type=int, default=40)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=False,
    )
    run(parser.parse_args())
