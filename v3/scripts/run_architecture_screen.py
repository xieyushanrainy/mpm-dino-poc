#!/usr/bin/env python3
"""Run the V3 DINO-centric architecture screen and winner-focused controls."""

from __future__ import annotations

import json
import argparse
import subprocess
import sys
from pathlib import Path


SEEDS = (42, 123, 456)
HORIZONS = ("1", "4", "8", "16")
VARIANTS = ("graph_direct", "latent_graph", "action_token_graph")


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


def train_v2_particle(project, root, seed, train_caches, val_caches, device):
    output = root / "baseline_particle_only" / f"seed{seed}" / "one_step"
    if not (output / "best.pt").exists():
        run([sys.executable, "v2/scripts/train.py", *train_caches, "--val-caches", *val_caches,
             "--variant", "particle_only", "--seed", str(seed), "--device", device, "--output", str(output),
             "--epochs", "60", "--lr-patience", "3", "--early-stop-patience", "8",
             "--min-relative-improvement", "0.005"], project, output / "train.log")
    return output / "best.pt"


def train_v3(project, root, variant, seed, dino_mode, train_caches, val_caches, device):
    output = root / variant / dino_mode / f"seed{seed}" / "one_step"
    if not (output / "best.pt").exists():
        run([sys.executable, "v3/scripts/train.py", *train_caches, "--val-caches", *val_caches,
             "--variant", variant, "--dino-mode", dino_mode, "--seed", str(seed), "--device", device,
             "--output", str(output), "--epochs", "60", "--lr-patience", "3",
             "--early-stop-patience", "8", "--min-relative-improvement", "0.005"],
            project, output / "train.log")
    return output / "best.pt"


def evaluate(project, checkpoint, caches, output, v3: bool, device):
    if not output.exists():
        script = "v3/scripts/evaluate_horizons.py" if v3 else "v2/scripts/evaluate_horizons.py"
        run([sys.executable, script, str(checkpoint), *caches,
             "--horizons", *HORIZONS, "--device", device, "--output", str(output)],
            project, output.with_suffix(".log"))


def h4_h8_score(path: Path) -> float:
    result = json.loads(path.read_text())
    return 0.5 * (result["horizons"]["4"]["particle_mean"] + result["horizons"]["8"]["particle_mean"])


def summarize(root: Path, variants: list[tuple[str, str]], seeds=SEEDS):
    summary = {}
    for variant, mode in variants:
        values = []
        for seed in seeds:
            path = root / variant / mode / f"seed{seed}" / "validation_horizons.json"
            values.append(h4_h8_score(path))
        summary[f"{variant}:{mode}"] = {"h4_h8_mean": sum(values) / len(values), "seed_scores": values}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    root = args.output_root or project / "v3" / "runs" / "architecture_screen"
    manifests = project / "data" / "shared" / "splits"
    cache_dir = project / "data" / "v2" / "cache"
    train_caches = cache_paths(cache_dir, manifests / "poc_train.txt")
    val_caches = cache_paths(cache_dir, manifests / "poc_val.txt")
    test_caches = cache_paths(cache_dir, manifests / "poc_test.txt")

    for seed in SEEDS:
        baseline = train_v2_particle(project, root, seed, train_caches, val_caches, args.device)
        evaluate(project, baseline, val_caches, root / "baseline_particle_only" / f"seed{seed}" / "validation_horizons.json", False, args.device)

    for variant in VARIANTS:
        for seed in SEEDS:
            checkpoint = train_v3(project, root, variant, seed, "final", train_caches, val_caches, args.device)
            evaluate(project, checkpoint, val_caches, root / variant / "final" / f"seed{seed}" / "validation_horizons.json", True, args.device)

    screen = summarize(root, [(variant, "final") for variant in VARIANTS])
    winner = min(screen, key=lambda key: screen[key]["h4_h8_mean"])
    winner_variant, _ = winner.split(":")
    (root / "screen_summary.json").write_text(json.dumps({"screen": screen, "winner": winner}, indent=2) + "\n")
    (root / "selected_v3_variant.txt").write_text(winner_variant + "\n")

    control_modes = ["zero", "shuffled_particles"]
    if winner_variant == "latent_graph":
        control_modes.append("geometry_only")
    for mode in control_modes:
        for seed in SEEDS:
            checkpoint = train_v3(project, root, winner_variant, seed, mode, train_caches, val_caches, args.device)
            evaluate(project, checkpoint, val_caches, root / winner_variant / mode / f"seed{seed}" / "validation_horizons.json", True, args.device)

    for seed in SEEDS:
        baseline = root / "baseline_particle_only" / f"seed{seed}" / "one_step" / "best.pt"
        evaluate(project, baseline, test_caches, root / "baseline_particle_only" / f"seed{seed}" / "test_horizons.json", False, args.device)
        winner_checkpoint = root / winner_variant / "final" / f"seed{seed}" / "one_step" / "best.pt"
        evaluate(project, winner_checkpoint, test_caches, root / winner_variant / "final" / f"seed{seed}" / "test_horizons.json", True, args.device)

    controls = summarize(root, [(winner_variant, mode) for mode in ["final", *control_modes]])
    (root / "dino_control_summary.json").write_text(json.dumps({"winner": winner_variant, "controls": controls}, indent=2) + "\n")
    print(f"selected V3 variant: {winner_variant}", flush=True)


if __name__ == "__main__":
    main()
