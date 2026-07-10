#!/usr/bin/env python3
"""Evaluate recurrent particle error at fixed horizons with motion quartiles."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from mpm_dino_v2.cache import load_v2_cache
from mpm_dino_v2.model import ParticleGridSurrogate


def predict(model, positions, velocities, scene, frame, mask, device):
    dt = scene["dt"].to(device)
    controller = scene["controller"].to(device)
    controller_velocity = (controller[frame + 1] - controller[frame]) / dt
    return model(
        positions[None], velocities[None], scene["dino"].to(device)[None], mask[None],
        scene["dino_imputed"].to(device)[None], controller[frame][None], controller_velocity[None],
        torch.ones((1, controller.shape[1]), dtype=torch.bool, device=device), scene["scale"].to(device)[None],
        dt[None], scene["x0"].to(device)[None], scene["neighbour_indices"].to(device)[None],
        scene["neighbour_mask"].to(device)[None], scene["rest_edge_lengths"].to(device)[None]).displacement[0]


def evaluate(model, scenes, horizon, device):
    windows = []
    total_error = total_persistence = 0.0
    total_count = calls = 0
    peak_memory = 0
    started = time.perf_counter()
    with torch.no_grad():
        for scene in scenes:
            points, dt = scene["points"].to(device), scene["dt"].to(device)
            for start in range(1, points.shape[0] - horizon):
                positions = points[start].clone()
                velocities = (positions - points[start - 1]) / dt
                initial = (scene["particle_mask"] & scene["visible"][start] & scene["motion_valid"][start]).to(device)
                origin = positions.clone()
                window_error = window_persistence = window_motion = 0.0
                window_count = 0
                for step in range(horizon):
                    displacement = predict(model, positions, velocities, scene, start + step, initial, device)
                    positions = positions + displacement
                    velocities = displacement / dt
                    target = points[start + step + 1]
                    valid = initial & scene["visible"][start + step + 1].to(device) & scene["motion_valid"][start + step + 1].to(device)
                    count = int(valid.sum())
                    error = torch.linalg.vector_norm(positions[valid] - target[valid], dim=-1).sum()
                    persistence = torch.linalg.vector_norm(origin[valid] - target[valid], dim=-1).sum()
                    motion = torch.linalg.vector_norm(
                        points[start + step + 1][valid] - points[start + step][valid], dim=-1).sum()
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
    return {"particle_mean": mean, "persistence_mean": persistence,
            "persistence_ratio": mean / max(persistence, 1e-12), "valid_points": total_count,
            "windows": len(windows), "motion_quartiles": quartiles,
            "forward_calls_per_second": calls / max(elapsed, 1e-12), "elapsed_seconds": elapsed,
            "peak_mps_allocated_bytes_observed": peak_memory}


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
    model = ParticleGridSurrogate(dino_dim=scenes[0]["dino"].shape[-1], base=config["base"],
                                  resolution=config["resolution"], variant=config.get("variant", "fused"))
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model = model.to(device).eval()
    result = {"checkpoint": str(args.checkpoint), "variant": config.get("variant", "fused"),
              "parameters": sum(parameter.numel() for parameter in model.parameters()), "horizons": {}}
    for horizon in args.horizons:
        result["horizons"][str(horizon)] = evaluate(model, scenes, horizon, device)
        print(horizon, result["horizons"][str(horizon)], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
