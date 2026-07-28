#!/usr/bin/env python3
"""Run the V4.1 COM-normalized local-shape Phase-2 experiment."""

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

from mpm_dino_v4.v41_shape_train import train_v41_local_shape


CONDITIONS = (
    "physical_only", "geometry_tokens", "real_dino", "point_shuffled",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(args, manifest):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "experiment": "v41_com_normalized_local_shape_phase2",
        "started_unix": time.time(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else None
        ),
        "git_commit": commit,
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "resolved_arguments": vars(args),
        "conditions": list(CONDITIONS),
        "scientific_contract": {
            "future_com_input": False,
            "world_and_com_loss": False,
            "primary_family": "soft_body",
            "sampling": "30 soft + 10 rigid draws per default epoch",
            "rigid_role": "negative control with zero-local penalty",
            "selection": "soft validation shape NRMSE H16/H30/H40",
            "evaluation": (
                "soft shape/strain and rigid false-deformation separately"
            ),
        },
    }


def train_if_needed(
    path, root, manifest, condition, seed, args,
    geometry_reference=None,
):
    if (path / "RUN_COMPLETE.json").exists():
        print(f"skip complete: {path}", flush=True)
        return path / "best.pt"
    path.mkdir(parents=True, exist_ok=True)
    (path / "RUNNING.json").write_text(json.dumps({
        "status": "running",
        "started_unix": time.time(),
        "condition": condition,
        "seed": seed,
    }, indent=2) + "\n")
    try:
        result = train_v41_local_shape(
            root=root,
            manifest=manifest,
            output=path,
            condition=condition,
            seed=seed,
            device=args.device,
            epochs=args.epochs,
            draws_per_epoch=args.draws,
            patience=args.patience,
            plateau_patience=args.plateau_patience,
            amp=args.amp,
            resume=True,
            geometry_reference=geometry_reference,
        )
    except BaseException as exc:
        (path / "RUN_FAILED.json").write_text(json.dumps({
            "status": "failed_or_interrupted",
            "time_unix": time.time(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "resume_command": "rerun the identical Phase-2 command",
        }, indent=2) + "\n")
        raise
    finally:
        (path / "RUNNING.json").unlink(missing_ok=True)
    (path / "RUN_FAILED.json").unlink(missing_ok=True)
    return result


def run(args):
    manifest = json.loads(Path(args.manifest).read_text())
    output = Path(args.runs)
    output.mkdir(parents=True, exist_ok=True)
    current = provenance(args, manifest)
    matrix_config = output / "MATRIX_CONFIG.json"
    if matrix_config.exists():
        previous = json.loads(matrix_config.read_text())
        keys = (
            "epochs", "draws", "patience", "plateau_patience",
            "seeds", "device", "amp",
        )
        changed = [
            key for key in keys
            if previous["resolved_arguments"].get(key)
            != current["resolved_arguments"].get(key)
        ]
        if changed:
            raise ValueError(
                f"refusing to mix changed Phase-2 settings: {changed}"
            )
        if (
            previous["manifest_content_sha256"]
            != manifest["manifest_content_sha256"]
        ):
            raise ValueError("refusing to mix a changed manifest")
    else:
        matrix_config.write_text(json.dumps(current, indent=2) + "\n")

    for seed in args.seeds:
        physical = output / "physical_only" / f"seed{seed}"
        geometry = output / "geometry_tokens" / f"seed{seed}"
        train_if_needed(
            physical, Path(args.dataset), manifest,
            "physical_only", seed, args,
        )
        geometry_best = train_if_needed(
            geometry, Path(args.dataset), manifest,
            "geometry_tokens", seed, args,
        )
        if not geometry_best.exists():
            raise RuntimeError(
                f"seed {seed} geometry-token reference has no guarded best"
            )
        for condition in ("real_dino", "point_shuffled"):
            train_if_needed(
                output / condition / f"seed{seed}",
                Path(args.dataset), manifest, condition, seed, args,
                geometry_reference=geometry_best,
            )

        configs = {
            condition: json.loads(
                (output / condition / f"seed{seed}" / "config.json")
                .read_text()
            )
            for condition in CONDITIONS
        }
        token_hashes = {
            configs[condition]["starting_model_sha256"]
            for condition in (
                "geometry_tokens", "real_dino", "point_shuffled",
            )
        }
        if len(token_hashes) != 1:
            raise RuntimeError(
                f"seed {seed} token conditions lack identical initialization"
            )
        trunk_hashes = {
            config["starting_physical_trunk_sha256"]
            for config in configs.values()
        }
        if len(trunk_hashes) != 1:
            raise RuntimeError(
                f"seed {seed} conditions lack identical physical trunk"
            )

    records = []
    for marker in sorted(output.glob("**/RUN_COMPLETE.json")):
        records.append({
            "run": str(marker.parent.relative_to(output)),
            "completion": json.loads(marker.read_text()),
            "config_sha256": sha256(marker.parent / "config.json"),
        })
    expected = len(args.seeds) * len(CONDITIONS)
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
        "--runs",
        default="v41/runs/local_shape_phase2_seed42_456",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 456],
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--draws", type=int, default=40)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=False,
    )
    run(parser.parse_args())
