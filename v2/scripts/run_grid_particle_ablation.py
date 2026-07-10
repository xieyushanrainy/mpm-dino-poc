#!/usr/bin/env python3
"""Run the staged grid-only, particle-only, and fused architecture ablation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def cache_paths(cache_dir: Path, manifest: Path):
    return [str(cache_dir / f"{Path(line).stem}.pt") for line in manifest.read_text().splitlines() if line]


def run(command, project, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    print("running:", " ".join(map(str, command)), flush=True)
    with log.open("a") as handle:
        process = subprocess.Popen(command, cwd=project, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True); handle.write(line); handle.flush()
        if process.wait():
            raise SystemExit(f"failed; see {log}")


def train(project, root, variant, seed, train_caches, val_caches):
    output = root / f"seed{seed}" / variant / "one_step"
    if not (output / "best.pt").exists():
        run([sys.executable, "v2/scripts/train.py", *train_caches, "--val-caches", *val_caches,
             "--variant", variant, "--seed", str(seed), "--device", "mps", "--output", str(output),
             "--epochs", "60", "--lr-patience", "3", "--early-stop-patience", "8",
             "--min-relative-improvement", "0.005"], project, output / "train.log")
    return output / "best.pt"


def evaluate(project, checkpoint, caches, output):
    if not output.exists():
        run([sys.executable, "v2/scripts/evaluate_horizons.py", str(checkpoint), *caches,
             "--horizons", "1", "4", "8", "16", "--device", "mps", "--output", str(output)],
            project, output.with_suffix(".log"))


def main():
    project = Path(__file__).resolve().parents[2]
    root = project / "v2" / "runs" / "grid_particle_ablation"
    manifests = project / "data" / "shared" / "splits"
    cache_dir = project / "data" / "v2" / "cache"
    train_caches = cache_paths(cache_dir, manifests / "poc_train.txt")
    val_caches = cache_paths(cache_dir, manifests / "poc_val.txt")
    test_caches = cache_paths(cache_dir, manifests / "poc_test.txt")
    fused = {
        42: project / "v2/runs/v2b_mps/one_step/best.pt",
        123: project / "v2/runs/dino_seed_validation/seed123/block11/one_step/best.pt",
        456: project / "v2/runs/dino_seed_validation/seed456/block11/one_step/best.pt",
    }
    checkpoints = {"fused": fused[42]}
    for variant in ("grid_only", "particle_only"):
        checkpoints[variant] = train(project, root, variant, 42, train_caches, val_caches)
    for variant, checkpoint in checkpoints.items():
        evaluate(project, checkpoint, val_caches, root / "seed42" / variant / "validation_horizons.json")
    alternatives = {
        variant: json.loads((root / "seed42" / variant / "validation_horizons.json").read_text())
        ["horizons"]["4"]["particle_mean"] for variant in ("grid_only", "particle_only")}
    selected = min(alternatives, key=alternatives.get)
    (root / "selected_alternative.txt").write_text(selected + "\n")
    selected_checkpoints = {42: checkpoints[selected]}
    for seed in (123, 456):
        selected_checkpoints[seed] = train(project, root, selected, seed, train_caches, val_caches)
        evaluate(project, fused[seed], val_caches, root / f"seed{seed}" / "fused" / "validation_horizons.json")
        evaluate(project, selected_checkpoints[seed], val_caches,
                 root / f"seed{seed}" / selected / "validation_horizons.json")
    for variant, checkpoint in checkpoints.items():
        evaluate(project, checkpoint, test_caches, root / "seed42" / variant / "test_horizons.json")
    for seed in (123, 456):
        evaluate(project, fused[seed], test_caches, root / f"seed{seed}" / "fused" / "test_horizons.json")
        evaluate(project, selected_checkpoints[seed], test_caches,
                 root / f"seed{seed}" / selected / "test_horizons.json")
    print(f"selected alternative: {selected}", flush=True)


if __name__ == "__main__":
    main()
