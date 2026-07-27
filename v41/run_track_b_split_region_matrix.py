#!/usr/bin/env python3
"""Run the strict-local-DINO Track B region-token experiment on V4.1."""

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
        "experiment": "v41_track_b_strict_local_dino_region_tokens",
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
            "mechanism": MECHANISM,
            "physical_backbone": (
                "DINO-free width-128, four-block, four-head graph-temporal trunk"
            ),
            "com_path": "physical hidden -> pooled COM head; no DINO path",
            "local_path": (
                "physical hidden -> four geometry-aware DINO region tokens "
                "-> zero-init local adapter -> zero-mean local head"
            ),
            "region_tokens": 4,
            "dino_isolation": "strict_local_only",
        },
        "loss": {
            "profile": LOSS_PROFILE,
            "formula": "legacy_track_b + 0.2 * shape_only_auxiliary",
            "legacy_key_horizons": [4, 8, 16, 59],
            "shape_key_horizons": [16, 30, 40, 59],
            "shape_terms": {
                "shape": 1.0,
                "normalized_edge_strain": 0.5,
                "shape_key_horizons": 0.25,
            },
            "h59_role": "training stability anchor; diagnostic-only for promotion",
        },
        "scientific_contract": {
            "primary_comparison": "real versus matched zero DINO",
            "matched_initialization": "full model byte-identical per seed",
            "evaluation_horizons": [1, 8, 16, 30, 40, 59],
            "promotion_rule": (
                "three-seed mean win at Panel Z H30 or H40; at least "
                "two paired-seed wins; H1 no more than 10% worse"
            ),
            "shuffled_controls": "only after promotion",
        },
    }


def train_if_needed(
    path, root, manifest, mode, seed, args, zero_reference=None,
):
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
            loss_profile=LOSS_PROFILE,
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


def assert_matched_initialization(zero: Path, real: Path, seed: int):
    zero_config = json.loads((zero / "config.json").read_text())
    real_config = json.loads((real / "config.json").read_text())
    zero_hash = zero_config["starting_model_sha256"]
    real_hash = real_config["starting_model_sha256"]
    if zero_hash != real_hash:
        raise RuntimeError(
            f"seed {seed} real/zero initial models are not byte-identical: "
            f"{zero_hash} != {real_hash}"
        )


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
            "device", "amp",
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
        if (
            previous["manifest_content_sha256"]
            != manifest["manifest_content_sha256"]
        ):
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
        assert_matched_initialization(zero, real, seed)

    records = []
    for marker in sorted(runs.glob("**/RUN_COMPLETE.json")):
        records.append({
            "run": str(marker.parent.relative_to(runs)),
            "completion": json.loads(marker.read_text()),
            "config_sha256": sha256(marker.parent / "config.json"),
        })
    expected = len(args.seeds) * 2
    (runs / "MATRIX_COMPLETE.json").write_text(json.dumps({
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
        default="v41/runs/track_b_split_region_cap100_p15_fp32",
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
