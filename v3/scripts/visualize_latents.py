#!/usr/bin/env python3
"""Visualize V3 object latents or raw pooled DINO scene embeddings.

The primary mode expects a trained ``latent_graph`` checkpoint and plots the
learned ``z_object`` vectors.  The fallback ``raw_dino`` mode plots pooled
frame-0 DINO scene descriptors without needing a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import torch

from mpm_dino_v2.cache import load_v2_cache
from mpm_dino_v3.model import V3ParticleSurrogate


FAMILY_COLORS = {
    "cloth": "#3b82f6",
    "rope": "#16a34a",
    "sloth": "#9333ea",
    "zebra": "#111827",
    "dinosor": "#f97316",
    "package": "#dc2626",
    "other": "#64748b",
}
SPLIT_MARKERS = {"train": "circle", "val": "triangle", "test": "square"}


def manifest_cache_paths(cache_dir: Path, manifest: Path) -> list[Path]:
    paths = []
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if line:
            paths.append(cache_dir / f"{Path(line).stem}.pt")
    return paths


def infer_family(name: str) -> str:
    for family in ("cloth", "rope", "sloth", "zebra", "dinosor", "package"):
        if family in name:
            return family
    return "other"


def infer_action(name: str) -> str:
    if "double_lift" in name:
        return "double_lift"
    if "double_stretch" in name:
        return "double_stretch"
    if "single_clift" in name:
        return "single_clift"
    if "single_lift" in name:
        return "single_lift"
    if "single_push" in name:
        return "single_push"
    if "rope_double_hand" in name:
        return "double_hand"
    return "other"


def load_checkpoint_model(checkpoint_path: Path, dino_dim: int, device: torch.device) -> tuple[V3ParticleSurrogate, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    if config["variant"] != "latent_graph":
        raise ValueError(f"checkpoint variant must be latent_graph, got {config['variant']}")
    model = V3ParticleSurrogate(
        dino_dim=dino_dim,
        dino_embed_dim=config["dino_embed_dim"],
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        layers=config["layers"],
        variant=config["variant"],
        attention_heads=config["attention_heads"],
        resolution=config["resolution"],
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), config


def apply_dino_mode(dino: torch.Tensor, dino_imputed: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    if mode in {"zero", "geometry_only"}:
        return torch.zeros_like(dino), dino_imputed
    if mode == "shuffled_particles":
        return torch.roll(dino, shifts=1, dims=1), torch.roll(dino_imputed, shifts=1, dims=1)
    if mode == "final":
        return dino, dino_imputed
    raise ValueError(f"unsupported DINO mode: {mode}")


def pooled_raw_dino(scene: dict) -> torch.Tensor:
    mask = scene["particle_mask"]
    dino = scene["dino"][mask]
    imputed = scene["dino_imputed"][mask].to(dino.dtype)[:, None]
    mean = dino.mean(dim=0)
    std = dino.std(dim=0, unbiased=False)
    imputed_fraction = imputed.mean(dim=0)
    return torch.cat((mean, std, imputed_fraction), dim=0)


def pca2(vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered = vectors - vectors.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    coords = centered @ vh[:2].T
    denom = singular_values.square().sum().clamp_min(1e-12)
    explained = singular_values[:2].square() / denom
    return coords, explained


def nearest_family_accuracy(coords: torch.Tensor, families: list[str]) -> float:
    if len(families) < 2:
        return 0.0
    distances = torch.cdist(coords, coords)
    distances.fill_diagonal_(torch.inf)
    nearest = distances.argmin(dim=1).tolist()
    correct = sum(families[i] == families[j] for i, j in enumerate(nearest))
    return correct / len(families)


def marker_svg(kind: str, x: float, y: float, color: str, label: str) -> str:
    safe = html.escape(label)
    if kind == "triangle":
        points = f"{x},{y - 7} {x - 7},{y + 6} {x + 7},{y + 6}"
        return f'<polygon points="{points}" fill="{color}"><title>{safe}</title></polygon>'
    if kind == "square":
        return f'<rect x="{x - 6}" y="{y - 6}" width="12" height="12" fill="{color}"><title>{safe}</title></rect>'
    return f'<circle cx="{x}" cy="{y}" r="6" fill="{color}"><title>{safe}</title></circle>'


def write_svg(rows: list[dict], explained: torch.Tensor, output: Path, title: str) -> None:
    width, height, pad = 920, 680, 70
    xs = [row["pc1"] for row in rows]
    ys = [row["pc2"] for row in rows]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax += 1
    if ymax == ymin:
        ymax += 1

    def scale_x(x: float) -> float:
        return pad + (x - xmin) / (xmax - xmin) * (width - 2 * pad)

    def scale_y(y: float) -> float:
        return height - pad - (y - ymin) / (ymax - ymin) * (height - 2 * pad)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{pad}" y="34" font-family="Arial" font-size="22" font-weight="700">{html.escape(title)}</text>',
        f'<text x="{pad}" y="58" font-family="Arial" font-size="13" fill="#475569">PC1 {float(explained[0]) * 100:.1f}% / PC2 {float(explained[1]) * 100:.1f}% variance</text>',
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#cbd5e1"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#cbd5e1"/>',
    ]
    for row in rows:
        label = f"{row['scene']} | {row['split']} | {row['family']} | {row['action']}"
        elements.append(marker_svg(
            SPLIT_MARKERS.get(row["split"], "circle"),
            scale_x(row["pc1"]),
            scale_y(row["pc2"]),
            FAMILY_COLORS.get(row["family"], FAMILY_COLORS["other"]),
            label,
        ))
        elements.append(
            f'<text x="{scale_x(row["pc1"]) + 8}" y="{scale_y(row["pc2"]) + 4}" '
            f'font-family="Arial" font-size="10" fill="#334155">{html.escape(row["scene"])}</text>'
        )
    legend_x, legend_y = width - 220, 88
    elements.append(f'<text x="{legend_x}" y="{legend_y - 18}" font-family="Arial" font-size="13" font-weight="700">family color</text>')
    for i, (family, color) in enumerate(FAMILY_COLORS.items()):
        y = legend_y + i * 20
        elements.append(f'<circle cx="{legend_x}" cy="{y}" r="6" fill="{color}"/>')
        elements.append(f'<text x="{legend_x + 14}" y="{y + 4}" font-family="Arial" font-size="12">{family}</text>')
    elements.append("</svg>")
    output.write_text("\n".join(elements) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Trained V3 latent_graph checkpoint")
    parser.add_argument("--mode", choices=("latent", "raw_dino"), default="latent")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/v2/cache"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/shared/splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("v3/artifacts/latent_viz"))
    parser.add_argument("--tag", help="Output filename prefix. Defaults to mode/checkpoint DINO mode.")
    parser.add_argument("--dino-mode", choices=("checkpoint", "final", "zero", "shuffled_particles", "geometry_only"),
                        default="checkpoint")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.mode == "latent" and args.checkpoint is None:
        raise SystemExit("--checkpoint is required for --mode latent")

    split_manifests = {
        "train": args.splits_dir / "poc_train.txt",
        "val": args.splits_dir / "poc_val.txt",
        "test": args.splits_dir / "poc_test.txt",
    }
    scenes = []
    for split, manifest in split_manifests.items():
        for path in manifest_cache_paths(args.cache_dir, manifest):
            scene = load_v2_cache(path)
            scenes.append((split, path.stem, scene))
    if not scenes:
        raise SystemExit("no scenes found")

    device = torch.device(args.device)
    model = None
    checkpoint_config = {}
    if args.mode == "latent":
        model, checkpoint_config = load_checkpoint_model(args.checkpoint, scenes[0][2]["dino"].shape[-1], device)
    dino_mode = checkpoint_config.get("dino_mode", "final") if args.dino_mode == "checkpoint" else args.dino_mode

    vectors = []
    metadata = []
    with torch.no_grad():
        for split, name, scene in scenes:
            if args.mode == "latent":
                assert model is not None
                dino = scene["dino"].to(device)[None]
                dino_imputed = scene["dino_imputed"].to(device)[None]
                dino, dino_imputed = apply_dino_mode(dino, dino_imputed, dino_mode)
                z = model.latent_encoder(
                    scene["x0"].to(device)[None],
                    dino,
                    dino_imputed,
                    scene["particle_mask"].to(device)[None],
                )[0].cpu()
            else:
                z = pooled_raw_dino(scene)
            vectors.append(z)
            metadata.append({
                "split": split,
                "scene": name,
                "family": infer_family(name),
                "action": infer_action(name),
            })

    matrix = torch.stack(vectors).float()
    coords, explained = pca2(matrix)
    families = [row["family"] for row in metadata]
    accuracy = nearest_family_accuracy(coords, families)

    rows = []
    for row, coord, vector in zip(metadata, coords, matrix):
        rows.append({
            **row,
            "pc1": float(coord[0]),
            "pc2": float(coord[1]),
            "latent_norm": float(vector.norm()),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.tag or (f"latent_{dino_mode}" if args.mode == "latent" else "raw_dino")
    csv_path = args.output_dir / f"{prefix}_pca.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    svg_path = args.output_dir / f"{prefix}_pca.svg"
    write_svg(rows, explained, svg_path, "V3 learned object latents" if args.mode == "latent" else "Raw pooled DINO scene descriptors")
    summary = {
        "mode": args.mode,
        "dino_mode": dino_mode if args.mode == "latent" else None,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "scene_count": len(rows),
        "pca_explained_variance": [float(explained[0]), float(explained[1])],
        "nearest_family_accuracy_2d": accuracy,
        "csv": str(csv_path),
        "svg": str(svg_path),
    }
    summary_path = args.output_dir / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
