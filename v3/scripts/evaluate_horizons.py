#!/usr/bin/env python3
"""Evaluate V3 recurrent particle candidates with fixed initial action summaries."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from mpm_dino_v2.cache import load_v2_cache
from mpm_dino_v2.losses import edge_deformation_losses
from mpm_dino_v3.model import V3ParticleSurrogate


def scene_donor_order(scene_count: int, seed: int) -> list[int]:
    if scene_count <= 1:
        return list(range(scene_count))
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(scene_count, generator=generator).tolist()
    return order[1:] + order[:1]


def apply_dino_mode(batch, mode: str):
    if mode in {"zero", "geometry_only"}:
        batch["dino"] = torch.zeros_like(batch["dino"])
    elif mode == "shuffled_particles":
        batch["dino"] = torch.roll(batch["dino"], shifts=1, dims=1)
        batch["dino_imputed"] = torch.roll(batch["dino_imputed"], shifts=1, dims=1)
    elif mode not in {"final", "scene_shuffled"}:
        raise ValueError(f"unsupported DINO mode: {mode}")
    return batch


def predict(model, positions, velocities, scene, start, step, mask, device, dino_mode):
    dt = scene["dt"].to(device)
    controller = scene["controller"].to(device)
    controller_velocity = (controller[start + 1] - controller[start]) / dt
    batch = {
        "positions": positions[None],
        "velocities": velocities[None],
        "dino": scene.get("_dino_override", scene["dino"]).to(device)[None],
        "particle_mask": mask[None],
        "dino_imputed": scene.get("_dino_imputed_override", scene["dino_imputed"]).to(device)[None],
        "controller_positions": controller[start][None],
        "controller_velocity": controller_velocity[None],
        "controller_mask": torch.ones((1, controller.shape[1]), dtype=torch.bool, device=device),
        "scale": scene["scale"].to(device)[None],
        "dt": dt[None],
        "x0": scene["x0"].to(device)[None],
        "neighbour_indices": scene["neighbour_indices"].to(device)[None],
        "neighbour_mask": scene["neighbour_mask"].to(device)[None],
        "rest_edge_vectors": scene["rest_edge_vectors"].to(device)[None],
        "rest_edge_lengths": scene["rest_edge_lengths"].to(device)[None],
        "action_time": (dt * step)[None],
    }
    batch = apply_dino_mode(batch, dino_mode)
    return model(**batch).displacement[0]


def evaluate(model, scenes, horizon, device, dino_mode):
    windows = []
    total_error = total_persistence = 0.0
    total_edge_vector = total_edge_length = 0.0
    total_count = edge_steps = calls = 0
    peak_memory = 0
    started = time.perf_counter()
    with torch.no_grad():
        for scene in scenes:
            points, dt = scene["points"].to(device), scene["dt"].to(device)
            neighbour_indices = scene["neighbour_indices"].to(device)[None]
            neighbour_mask = scene["neighbour_mask"].to(device)[None]
            for start in range(1, points.shape[0] - horizon):
                positions = points[start].clone()
                velocities = (positions - points[start - 1]) / dt
                initial = (scene["particle_mask"] & scene["visible"][start] & scene["motion_valid"][start]).to(device)
                origin = positions.clone()
                window_error = window_persistence = window_motion = 0.0
                window_count = 0
                for step in range(horizon):
                    displacement = predict(model, positions, velocities, scene, start, step, initial, device, dino_mode)
                    positions = positions + displacement
                    velocities = displacement / dt
                    target = points[start + step + 1]
                    valid = initial & scene["visible"][start + step + 1].to(device) & scene["motion_valid"][start + step + 1].to(device)
                    count = int(valid.sum())
                    error = torch.linalg.vector_norm(positions[valid] - target[valid], dim=-1).sum()
                    persistence = torch.linalg.vector_norm(origin[valid] - target[valid], dim=-1).sum()
                    motion = torch.linalg.vector_norm(
                        points[start + step + 1][valid] - points[start + step][valid], dim=-1).sum()
                    edge_vector, edge_length = edge_deformation_losses(
                        positions[None], target[None], valid[None], neighbour_indices, neighbour_mask,
                    )
                    total_edge_vector += float(edge_vector.cpu())
                    total_edge_length += float(edge_length.cpu())
                    edge_steps += 1
                    window_error += float(error.cpu()); window_persistence += float(persistence.cpu())
                    window_motion += float(motion.cpu()); window_count += count; calls += 1
                    if device.type == "mps":
                        peak_memory = max(peak_memory, int(torch.mps.current_allocated_memory()))
                total_error += window_error; total_persistence += window_persistence; total_count += window_count
                windows.append((window_motion / max(window_count, 1), window_error, window_persistence, window_count))
    elapsed = time.perf_counter() - started
    scores = np.asarray([window[0] for window in windows])
    quartiles = []
    order = np.argsort(scores)
    for ids in np.array_split(order, 4):
        count = sum(windows[i][3] for i in ids)
        error = sum(windows[i][1] for i in ids) / max(count, 1)
        persistence = sum(windows[i][2] for i in ids) / max(count, 1)
        quartiles.append({"particle_mean": error, "persistence_mean": persistence,
                          "persistence_ratio": error / max(persistence, 1e-12), "windows": len(ids)})
    mean = total_error / max(total_count, 1)
    persistence = total_persistence / max(total_count, 1)
    return {
        "particle_mean": mean, "persistence_mean": persistence,
        "persistence_ratio": mean / max(persistence, 1e-12), "valid_points": total_count,
        "edge_vector": total_edge_vector / max(edge_steps, 1),
        "edge_length": total_edge_length / max(edge_steps, 1),
        "windows": len(windows), "motion_quartiles": quartiles,
        "forward_calls_per_second": calls / max(elapsed, 1e-12), "elapsed_seconds": elapsed,
        "peak_mps_allocated_bytes_observed": peak_memory,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("caches", nargs="+", type=Path)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    scenes = [load_v2_cache(path) for path in args.caches]
    if config["dino_mode"] == "scene_shuffled":
        donor_order = scene_donor_order(len(scenes), int(config.get("seed", 0)))
        for scene_id, donor_id in enumerate(donor_order):
            scenes[scene_id]["_dino_override"] = scenes[donor_id]["dino"]
            scenes[scene_id]["_dino_imputed_override"] = scenes[donor_id]["dino_imputed"]
    model = V3ParticleSurrogate(
        dino_dim=scenes[0]["dino"].shape[-1], dino_embed_dim=config["dino_embed_dim"],
        hidden_dim=config["hidden_dim"], latent_dim=config["latent_dim"], layers=config["layers"],
        variant=config["variant"], attention_heads=config["attention_heads"], resolution=config["resolution"],
        latent_geometry_mode=config.get("latent_geometry_mode", "full"),
        latent_geometry_dim=config.get("latent_geometry_dim", 3),
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model = model.to(device).eval()
    result = {
        "checkpoint": str(args.checkpoint), "variant": config["variant"], "dino_mode": config["dino_mode"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()), "horizons": {},
    }
    for horizon in args.horizons:
        result["horizons"][str(horizon)] = evaluate(model, scenes, horizon, device, config["dino_mode"])
        print(horizon, result["horizons"][str(horizon)], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
