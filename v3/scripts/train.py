#!/usr/bin/env python3
"""Train V3 particle-native one-step candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mpm_dino_v2.data import ScenePairDataset
from mpm_dino_v2.losses import one_step_loss
from mpm_dino_v3.model import V3ParticleSurrogate


MODEL_KEYS = [
    "positions", "velocities", "dino", "particle_mask", "dino_imputed",
    "controller_positions", "controller_velocity", "controller_mask", "scale", "dt",
    "x0", "neighbour_indices", "neighbour_mask", "rest_edge_vectors", "rest_edge_lengths",
]
SELECTION_METRIC = "val_particle_mean"


DINO_MODES = ("final", "zero", "shuffled_particles", "scene_shuffled", "geometry_only")


def scene_donor_order(scene_count: int, seed: int) -> list[int]:
    if scene_count <= 1:
        return list(range(scene_count))
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(scene_count, generator=generator).tolist()
    return order[1:] + order[:1]


class SceneShuffledPairDataset(ScenePairDataset):
    def __init__(self, paths: list[str | Path], seed: int):
        super().__init__(paths)
        self.donor_order = scene_donor_order(len(self.scenes), seed)

    def __getitem__(self, item):
        result = super().__getitem__(item)
        scene_id, _ = self.index[item]
        donor = self.scenes[self.donor_order[scene_id]]
        result["dino"] = donor["dino"]
        result["dino_imputed"] = donor["dino_imputed"]
        return result


def move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def apply_dino_mode(batch, mode: str):
    if mode in {"zero", "geometry_only"}:
        batch["dino"] = torch.zeros_like(batch["dino"])
    elif mode == "shuffled_particles":
        batch["dino"] = torch.roll(batch["dino"], shifts=1, dims=1)
        batch["dino_imputed"] = torch.roll(batch["dino_imputed"], shifts=1, dims=1)
    elif mode not in {"final", "scene_shuffled"}:
        raise ValueError(f"unsupported DINO mode: {mode}")
    return batch


def run_epoch(model, loader, device, args, optimizer=None):
    training = optimizer is not None
    model.train(training)
    names = ["objective", "particle_loss", "occupancy", "grid_velocity", "consistency", "edge_vector", "edge_length"]
    totals = {name: 0.0 for name in names}
    error_sum = persistence_sum = 0.0
    valid_count = batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw in loader:
            batch = apply_dino_mode(move(raw, device), args.dino_mode)
            prediction = model(**{key: batch[key] for key in MODEL_KEYS})
            losses = one_step_loss(
                prediction, batch, model.spec, args.particle_beta,
                args.edge_vector_weight, args.edge_length_weight,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses.total.backward()
                optimizer.step()
            values = {
                "objective": losses.total, "particle_loss": losses.particle,
                "occupancy": losses.occupancy, "grid_velocity": losses.grid_velocity,
                "consistency": losses.consistency, "edge_vector": losses.edge_vector,
                "edge_length": losses.edge_length,
            }
            for name, value in values.items():
                totals[name] += float(value.detach().cpu())
            mask = batch["target_mask"]
            error = torch.linalg.vector_norm(prediction.displacement - batch["target_displacement"], dim=-1)
            persistence = torch.linalg.vector_norm(batch["target_displacement"], dim=-1)
            error_sum += float((error * mask).sum().detach().cpu())
            persistence_sum += float((persistence * mask).sum().detach().cpu())
            valid_count += int(mask.sum().detach().cpu())
            batches += 1
    metrics = {key: value / max(batches, 1) for key, value in totals.items()}
    metrics.update({
        "weighted_edge_vector": args.edge_vector_weight * metrics["edge_vector"],
        "weighted_edge_length": args.edge_length_weight * metrics["edge_length"],
        "particle_mean": error_sum / max(valid_count, 1),
        "persistence_mean": persistence_sum / max(valid_count, 1),
        "valid_points": valid_count,
    })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("caches", nargs="+")
    parser.add_argument("--val-caches", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", default="v3/runs/one_step")
    parser.add_argument("--checkpoint", type=Path, help="optional V3 checkpoint to initialize from")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--particle-beta", type=float, default=0.01)
    parser.add_argument("--edge-vector-weight", type=float, default=0.25)
    parser.add_argument("--edge-length-weight", type=float, default=0.10)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--min-relative-improvement", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", choices=("graph_direct", "latent_graph", "action_token_graph"), required=True)
    parser.add_argument("--dino-mode", choices=DINO_MODES, default="final")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--dino-embed-dim", type=int, default=16)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--latent-geometry-mode", choices=("full", "bottleneck", "none"), default="full")
    parser.add_argument("--latent-geometry-dim", type=int, default=3)
    args = parser.parse_args()
    if args.early_stop_patience <= args.lr_patience:
        raise ValueError("early-stop-patience must exceed lr-patience")
    if args.dino_mode == "geometry_only" and args.variant != "latent_graph":
        raise ValueError("geometry_only DINO mode is only defined for the latent_graph variant")
    torch.manual_seed(args.seed)
    dataset_cls = SceneShuffledPairDataset if args.dino_mode == "scene_shuffled" else ScenePairDataset
    if args.dino_mode == "scene_shuffled":
        train_data = dataset_cls(args.caches, args.seed)
        val_data = dataset_cls(args.val_caches, args.seed)
    else:
        train_data, val_data = dataset_cls(args.caches), dataset_cls(args.val_caches)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else None
    if checkpoint is not None:
        config = checkpoint["args"]
        for key in (
            "variant", "hidden_dim", "latent_dim", "dino_embed_dim", "layers",
            "attention_heads", "resolution", "latent_geometry_mode", "latent_geometry_dim",
        ):
            if getattr(args, key) != config.get(key, getattr(args, key)):
                raise ValueError(f"--{key.replace('_', '-')}={getattr(args, key)!r} does not match checkpoint {config.get(key)!r}")
        if args.dino_mode != config.get("dino_mode", args.dino_mode):
            raise ValueError(f"--dino-mode={args.dino_mode!r} does not match checkpoint {config.get('dino_mode')!r}")
    model = V3ParticleSurrogate(
        dino_dim=train_data.scenes[0]["dino"].shape[-1], dino_embed_dim=args.dino_embed_dim,
        hidden_dim=args.hidden_dim, latent_dim=args.latent_dim, layers=args.layers,
        variant=args.variant, attention_heads=args.attention_heads, resolution=args.resolution,
        latent_geometry_mode=args.latent_geometry_mode, latent_geometry_dim=args.latent_geometry_dim,
    ).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience,
        threshold=args.min_relative_improvement, threshold_mode="rel", min_lr=args.min_lr,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best_particle = best_objective = reference_best = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, train_loader, device, args, optimizer)
        val = run_epoch(model, val_loader, device, args)
        scheduler.step(val["particle_mean"])
        exact, objective_best = val["particle_mean"] < best_particle, val["objective"] < best_objective
        meaningful = val["particle_mean"] < reference_best * (1 - args.min_relative_improvement)
        if meaningful:
            reference_best, stale = val["particle_mean"], 0
        else:
            stale += 1
        best_particle, best_objective = min(best_particle, val["particle_mean"]), min(best_objective, val["objective"])
        ratio = val["particle_mean"] / max(val["persistence_mean"], 1e-12)
        record = {"epoch": epoch, "train": train, "val": val, "lr": optimizer.param_groups[0]["lr"], "persistence_ratio": ratio}
        print(f"epoch={epoch:03d} train={train['particle_mean']:.6g} val={val['particle_mean']:.6g} "
              f"ratio={ratio:.3f} edge=({val['edge_vector']:.4g},{val['edge_length']:.4g}) lr={record['lr']:.3g}", flush=True)
        with (output / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        state = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "selection_metric": SELECTION_METRIC,
            "best_particle": best_particle, "best_objective": best_objective,
            "early_stop_reference": reference_best, "stale_epochs": stale, "args": vars(args),
            "model_family": "v3_particle",
        }
        torch.save(state, output / "last.pt")
        if exact:
            torch.save(state, output / "best.pt")
        if objective_best:
            torch.save(state, output / "best_objective.pt")
        if stale >= args.early_stop_patience:
            print(f"early stopping after {stale} epochs without 0.5% relative improvement", flush=True)
            break


if __name__ == "__main__":
    main()
