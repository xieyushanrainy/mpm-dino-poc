#!/usr/bin/env python3
"""Run no-bottleneck latent-graph final-DINO vs zero-DINO checks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_SEEDS = (45, 123, 456)
MODES = ("final", "zero")
HORIZONS = ("1", "4", "8", "16")


def cache_paths(cache_dir: Path, manifest: Path) -> list[str]:
    return [str(cache_dir / f"{Path(line).stem}.pt") for line in manifest.read_text().splitlines() if line.strip()]


def run(command: list[str], project: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("running:", " ".join(map(str, command)), flush=True)
    with log.open("a") as handle:
        process = subprocess.Popen(
            command,
            cwd=project,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
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
    if checkpoint.exists():
        print(f"skipping existing checkpoint: {checkpoint}", flush=True)
        return checkpoint
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
        "--latent-geometry-mode", "full",
    ], project, output / "train.log")
    return checkpoint


def evaluate(project: Path, checkpoint: Path, val_caches: list[str], output: Path, device: str) -> None:
    if output.exists():
        print(f"skipping existing evaluation: {output}", flush=True)
        return
    run([
        sys.executable, "v3/scripts/evaluate_horizons.py", str(checkpoint), *val_caches,
        "--horizons", *HORIZONS,
        "--device", device,
        "--output", str(output),
    ], project, output.with_suffix(".log"))


def load_result(root: Path, mode: str, seed: int) -> dict:
    return json.loads((root / mode / f"seed{seed}" / "validation_horizons.json").read_text())


def particle(result: dict, horizon: str) -> float:
    return float(result["horizons"][horizon]["particle_mean"])


def edge(result: dict, horizon: str, key: str) -> float | None:
    value = result["horizons"][horizon].get(key)
    return None if value is None else float(value)


def h4_h8(result: dict) -> float:
    return 0.5 * (particle(result, "4") + particle(result, "8"))


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.8f}"


def pct(delta: float) -> str:
    return f"{100.0 * delta:.2f}%"


def build_summary(
    root: Path,
    seeds: tuple[int, ...],
    train_manifest: Path,
    val_manifest: Path,
    experiment_name: str,
) -> dict:
    results = {mode: {str(seed): load_result(root, mode, seed) for seed in seeds} for mode in MODES}
    per_mode = {}
    per_seed = {}
    for mode in MODES:
        scores = []
        horizon_values = {horizon: [] for horizon in HORIZONS}
        for seed in seeds:
            result = results[mode][str(seed)]
            scores.append(h4_h8(result))
            for horizon in HORIZONS:
                horizon_values[horizon].append(particle(result, horizon))
        per_mode[mode] = {
            "h4_h8_mean": sum(scores) / len(scores),
            "horizon_means": {
                horizon: sum(values) / len(values) for horizon, values in horizon_values.items()
            },
        }
    for seed in seeds:
        final_score = h4_h8(results["final"][str(seed)])
        zero_score = h4_h8(results["zero"][str(seed)])
        per_seed[str(seed)] = {
            "final_h4_h8": final_score,
            "zero_h4_h8": zero_score,
            "final_delta_vs_zero": final_score - zero_score,
            "final_delta_vs_zero_percent": final_score / zero_score - 1.0,
        }
    aggregate_delta = per_mode["final"]["h4_h8_mean"] - per_mode["zero"]["h4_h8_mean"]
    return {
        "variant": "latent_graph",
        "latent_geometry_mode": "full",
        "experiment_name": experiment_name,
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "seeds": list(seeds),
        "modes": list(MODES),
        "results": results,
        "per_mode": per_mode,
        "per_seed": per_seed,
        "aggregate": {
            "final_delta_vs_zero": aggregate_delta,
            "final_delta_vs_zero_percent": per_mode["final"]["h4_h8_mean"] / per_mode["zero"]["h4_h8_mean"] - 1.0,
        },
    }


def conclusion(summary: dict) -> str:
    seed_deltas = [values["final_delta_vs_zero_percent"] for values in summary["per_seed"].values()]
    aggregate_delta = summary["aggregate"]["final_delta_vs_zero_percent"]
    if all(delta < -0.005 for delta in seed_deltas):
        return "PASS: final-DINO beats zero-DINO on every seed by more than a tiny margin."
    if all(delta > 0 for delta in seed_deltas) or aggregate_delta > 0:
        return "FAIL: final-DINO ties or loses to zero-DINO, so DINO should not be required in the main V3 architecture."
    return "INCONCLUSIVE: results are mixed or margins are tiny, so DINO remains weak and scene-shuffle can wait."


def write_markdown(summary: dict, output: Path) -> None:
    seeds = [int(seed) for seed in summary["seeds"]]
    lines = [
        f"# {summary['experiment_name']}",
        "",
        "## Setup",
        "",
        "- Architecture: `latent_graph`",
        "- Geometry: default full geometry, no bottleneck",
        "- DINO modes: `final`, `zero`",
        f"- Seeds: `{', '.join(map(str, seeds))}`",
        f"- Train manifest: `{summary['train_manifest']}`",
        f"- Validation manifest: `{summary['val_manifest']}`",
        "- Evaluation: recurrent validation horizons H1/H4/H8/H16",
        "- Primary metric: mean of H4 and H8 particle error",
        "",
        "## Per-Seed Validation Particle Error",
        "",
        "| Seed | Mode | H1 | H4 | H8 | H16 | H4/H8 mean | H16 edge vector | H16 edge length |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in seeds:
        for mode in MODES:
            result = summary["results"][mode][str(seed)]
            lines.append(
                f"| {seed} | {mode} | {fmt(particle(result, '1'))} | {fmt(particle(result, '4'))} | "
                f"{fmt(particle(result, '8'))} | {fmt(particle(result, '16'))} | {fmt(h4_h8(result))} | "
                f"{fmt(edge(result, '16', 'edge_vector'))} | {fmt(edge(result, '16', 'edge_length'))} |"
            )
    lines.extend([
        "",
        "## H4/H8 Final vs Zero",
        "",
        "| Seed | final-DINO | zero-DINO | Final delta | Final delta % |",
        "|---:|---:|---:|---:|---:|",
    ])
    for seed in seeds:
        item = summary["per_seed"][str(seed)]
        lines.append(
            f"| {seed} | {fmt(item['final_h4_h8'])} | {fmt(item['zero_h4_h8'])} | "
            f"{fmt(item['final_delta_vs_zero'])} | {pct(item['final_delta_vs_zero_percent'])} |"
        )
    lines.extend([
        "",
        "## Aggregate",
        "",
        "| Mode | H1 mean | H4 mean | H8 mean | H16 mean | H4/H8 mean |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for mode in MODES:
        values = summary["per_mode"][mode]
        horizons = values["horizon_means"]
        lines.append(
            f"| {mode} | {fmt(horizons['1'])} | {fmt(horizons['4'])} | {fmt(horizons['8'])} | "
            f"{fmt(horizons['16'])} | {fmt(values['h4_h8_mean'])} |"
        )
    lines.extend([
        "",
        f"Aggregate final delta vs zero: {fmt(summary['aggregate']['final_delta_vs_zero'])} "
        f"({pct(summary['aggregate']['final_delta_vs_zero_percent'])}).",
        "",
        "## Interpretation",
        "",
        conclusion(summary),
        "",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--experiment-name", default="No-Bottleneck Final-DINO vs Zero-DINO")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--min-relative-improvement", type=float, default=0.005)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[2]
    root = args.output_root or project / "v3" / "runs" / "no_bottleneck_final_vs_zero"
    artifact = args.artifact or project / "v3" / "artifacts" / "NO_BOTTLENECK_FINAL_VS_ZERO.md"
    manifests = project / "data" / "shared" / "splits"
    cache_dir = project / "data" / "v2" / "cache"
    train_manifest = args.train_manifest or manifests / "poc_train.txt"
    val_manifest = args.val_manifest or manifests / "poc_val.txt"
    train_caches = cache_paths(cache_dir, train_manifest)
    val_caches = cache_paths(cache_dir, val_manifest)
    seeds = tuple(args.seeds)

    for seed in seeds:
        for mode in MODES:
            checkpoint = train(project, root, seed, mode, train_caches, val_caches, args)
            evaluate(project, checkpoint, val_caches, root / mode / f"seed{seed}" / "validation_horizons.json", args.device)

    summary = build_summary(root, seeds, train_manifest, val_manifest, args.experiment_name)
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(summary, artifact)
    print(json.dumps({
        "summary": str(summary_path),
        "artifact": str(artifact),
        "conclusion": conclusion(summary),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
