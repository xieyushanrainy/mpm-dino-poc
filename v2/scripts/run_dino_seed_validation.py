#!/usr/bin/env python3
"""Repeat final-layer and zero-DINO ablations with additional seeds."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def split(cache_dir: Path, manifest: Path) -> list[str]:
    return [str(cache_dir / f"{Path(line).stem}.pt") for line in manifest.read_text().splitlines() if line]


def run(command: list[str], project: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("running:", " ".join(command), flush=True)
    with log.open("a") as handle:
        process = subprocess.Popen(command, cwd=project, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        if process.wait():
            raise SystemExit(f"failed; see {log}")


def main() -> None:
    project = Path(__file__).resolve().parents[2]
    manifests = project / "data" / "shared" / "splits"
    variants = {
        "block11": project / "data" / "v2" / "cache",
        "zero": project / "data" / "v2" / "cache_dino_zero",
    }
    root = project / "v2" / "runs" / "dino_seed_validation"
    for seed in (123, 456):
        for name, cache_dir in variants.items():
            train = split(cache_dir, manifests / "poc_train.txt")
            val = split(cache_dir, manifests / "poc_val.txt")
            output = root / f"seed{seed}" / name / "one_step"
            run([
                sys.executable, "v2/scripts/train.py", *train, "--val-caches", *val,
                "--device", "mps", "--output", str(output), "--epochs", "60",
                "--lr-patience", "3", "--early-stop-patience", "8",
                "--min-relative-improvement", "0.005", "--seed", str(seed),
            ], project, output / "train.log")
            evaluation = root / f"seed{seed}" / name / "eval_h4"
            run([
                sys.executable, "v2/scripts/train_rollout.py", *train, "--val-caches", *val,
                "--checkpoint", str(output / "best.pt"), "--steps", "4", "--epochs", "0",
                "--device", "mps", "--output", str(evaluation),
            ], project, evaluation / "evaluate.log")


if __name__ == "__main__":
    main()
