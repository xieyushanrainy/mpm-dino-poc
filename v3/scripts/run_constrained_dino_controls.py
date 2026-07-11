#!/usr/bin/env python3
"""Run bottlenecked latent-graph DINO control checks across seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_SEEDS = (42, 123, 456)
DEFAULT_MODES = ("final", "shuffled_particles", "scene_shuffled")
HORIZONS = ("1", "4", "8", "16")


def cache_paths(cache_dir: Path, manifest: Path) -> list[str]:
    return [str(cache_dir / f"{Path(line).stem}.pt") for line in manifest.read_text().splitlines() if line.strip()]


def run(command: list[str], project: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("running:", " ".join(map(str, command)), flush=True)
    with log.open("a") as handle:
        process = subprocess.Popen(
            command, cwd=project, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        if process.wait():
            raise SystemExit(f"failed; see {log}")


def train(
    project: Path,
    root: Path,
    seed: int,
    mode: str,
    train_caches: list[str],
    val_caches: list[str],
    args: argparse.Namespace,
) -> Path:
    output = root / mode / f"seed{seed}" / "one_step"
    checkpoint = output / "best.pt"
    if not checkpoint.exists():
        run([
            sys.executable, "v3/scripts/train.py", *train_caches,
            "--val-caches", *val_caches,
            "--variant", "latent_graph",
            "--dino-mode", mode,
            "--seed", str(seed),
            "--device", args.device,
            "--output", str(output),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--lr-patience", str(args.lr_patience),
            "--early-stop-patience", str(args.early_stop_patience),
            "--min-relative-improvement", str(args.min_relative_improvement),
            "--latent-geometry-mode", args.latent_geometry_mode,
            "--latent-geometry-dim", str(args.latent_geometry_dim),
        ], project, output / "train.log")
    return checkpoint


def evaluate(project: Path, checkpoint: Path, val_caches: list[str], output: Path, device: str) -> None:
    if not output.exists():
        run([
            sys.executable, "v3/scripts/evaluate_horizons.py", str(checkpoint), *val_caches,
            "--horizons", *HORIZONS,
            "--device", device,
            "--output", str(output),
        ], project, output.with_suffix(".log"))


def h4_h8(path: Path) -> float:
    result = json.loads(path.read_text())
    return 0.5 * (result["horizons"]["4"]["particle_mean"] + result["horizons"]["8"]["particle_mean"])


def summarize(root: Path, modes: tuple[str, ...], seeds: tuple[int, ...]) -> dict:
    summary = {}
    for mode in modes:
        seed_scores = []
        horizons = {horizon: [] for horizon in HORIZONS}
        for seed in seeds:
            result = json.loads((root / mode / f"seed{seed}" / "validation_horizons.json").read_text())
            seed_scores.append(h4_h8(root / mode / f"seed{seed}" / "validation_horizons.json"))
            for horizon in HORIZONS:
                horizons[horizon].append(result["horizons"][horizon]["particle_mean"])
        summary[mode] = {
            "h4_h8_mean": sum(seed_scores) / len(seed_scores),
            "seed_h4_h8": dict(zip(map(str, seeds), seed_scores)),
            "horizon_means": {
                horizon: sum(values) / len(values) for horizon, values in horizons.items()
            },
        }
    final = summary.get("final")
    if final is not None:
        for mode, values in summary.items():
            values["delta_vs_final_h4_h8_percent"] = 100 * (values["h4_h8_mean"] / final["h4_h8_mean"] - 1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--modes", nargs="+", choices=DEFAULT_MODES, default=list(DEFAULT_MODES))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--min-relative-improvement", type=float, default=0.005)
    parser.add_argument("--latent-geometry-mode", choices=("bottleneck", "none"), default="bottleneck")
    parser.add_argument("--latent-geometry-dim", type=int, default=1)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[2]
    root = args.output_root or project / "v3" / "runs" / "constrained_dino_controls" / f"{args.latent_geometry_mode}{args.latent_geometry_dim}"
    manifests = project / "data" / "shared" / "splits"
    cache_dir = project / "data" / "v2" / "cache"
    train_caches = cache_paths(cache_dir, manifests / "poc_train.txt")
    val_caches = cache_paths(cache_dir, manifests / "poc_val.txt")
    seeds = tuple(args.seeds)
    modes = tuple(args.modes)

    for seed in seeds:
        for mode in modes:
            checkpoint = train(project, root, seed, mode, train_caches, val_caches, args)
            evaluate(project, checkpoint, val_caches, root / mode / f"seed{seed}" / "validation_horizons.json", args.device)

    summary = {
        "variant": "latent_graph",
        "latent_geometry_mode": args.latent_geometry_mode,
        "latent_geometry_dim": args.latent_geometry_dim,
        "seeds": list(seeds),
        "modes": list(modes),
        "summary": summarize(root, modes, seeds),
    }
    output = root / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
