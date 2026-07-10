#!/usr/bin/env python3
"""Run the V2 one-step, rollout-2, and rollout-4 training lineage sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def split_paths(project: Path, split: str) -> list[str]:
    names = (project / "data" / "shared" / "splits" / f"poc_{split}.txt").read_text().splitlines()
    return [str(project / "data" / "v2" / "cache" / f"{Path(name.strip()).stem}.pt")
            for name in names if name.strip()]


def run(command: list[str], project: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("running:", " ".join(command), flush=True)
    with log.open("a") as handle:
        process = subprocess.Popen(command, cwd=project, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        if process.wait() != 0:
            raise SystemExit(f"stage failed with exit code {process.returncode}; see {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--run-root", type=Path, default=Path("v2/runs/v2b_mps"))
    parser.add_argument("--one-step-epochs", type=int, default=60)
    parser.add_argument("--rollout-epochs", type=int, default=20)
    parser.add_argument("--start-stage", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--rollout2-name", default="rollout_s2")
    parser.add_argument("--rollout2-lr", type=float, default=2.5e-5)
    parser.add_argument("--rollout2-teacher-weight", type=float, default=0.5)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    run_root = args.run_root if args.run_root.is_absolute() else project / args.run_root
    train, val = split_paths(project, "train"), split_paths(project, "val")
    common = ["--val-caches", *val, "--device", args.device]

    one = run_root / "one_step"
    if args.start_stage == 1:
        run([
            sys.executable, "v2/scripts/train.py", *train, *common, "--output", str(one),
            "--epochs", str(args.one_step_epochs), "--lr-patience", "3", "--early-stop-patience", "8",
            "--min-relative-improvement", "0.005",
        ], project, one / "train.log")

    step2 = run_root / args.rollout2_name
    if args.start_stage in (1, 2):
        run([
            sys.executable, "v2/scripts/train_rollout.py", *train, *common,
            "--checkpoint", str(one / "best.pt"), "--steps", "2", "--output", str(step2),
            "--epochs", str(args.rollout_epochs), "--lr", str(args.rollout2_lr), "--lr-patience", "2",
            "--teacher-weight", str(args.rollout2_teacher_weight), "--early-stop-patience", "6",
            "--min-relative-improvement", "0.005",
        ], project, step2 / "train.log")

    step4 = run_root / "rollout_s4"
    run([
        sys.executable, "v2/scripts/train_rollout.py", *train, *common,
        "--checkpoint", str(step2 / "best.pt"), "--steps", "4", "--output", str(step4),
        "--epochs", str(args.rollout_epochs), "--lr", "1e-5", "--lr-patience", "2",
        "--early-stop-patience", "6", "--min-relative-improvement", "0.005",
    ], project, step4 / "train.log")
    print(f"completed V2 training lineage at {run_root}", flush=True)


if __name__ == "__main__":
    main()
