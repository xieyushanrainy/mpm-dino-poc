"""Measure recurrent sensitivity to derived-velocity feedback damping."""
import argparse
import csv
from pathlib import Path

import torch

from mpm_dino.model import ParticleGridSurrogate
from train_rollout import model_call


def default_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def load_model(path, scene, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ParticleGridSurrogate(
        dino_dim=scene["dino"].shape[-1], base=checkpoint["args"]["base"],
        resolution=checkpoint["args"]["resolution"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def batched_scene(scene, device):
    keys = ["points", "visible", "motion_valid", "controller", "dino", "dino_imputed"]
    batch = {key: scene[key][None].to(device) for key in keys}
    batch["scale"] = scene["scale"][None].to(device)
    batch["dt"] = scene["dt"][None].to(device)
    return batch


def evaluate_window(model, batch, start, horizons, alpha):
    dt = batch["dt"]
    position = batch["points"][:, start].clone()
    start_position = position.clone()
    velocity = (position - batch["points"][:, start - 1]) / dt[:, None, None]
    initial_mask = batch["visible"][:, start] & batch["motion_valid"][:, start]
    results = {}
    with torch.no_grad():
        for offset in range(max(horizons)):
            output = model_call(model, position, velocity, batch, start + offset, initial_mask)
            position = position + output.displacement
            derived_velocity = output.displacement / dt[:, None, None]
            velocity = alpha * derived_velocity + (1 - alpha) * velocity
            horizon = offset + 1
            if horizon not in horizons: continue
            target = batch["points"][:, start + horizon]
            valid = initial_mask & batch["visible"][:, start + horizon] & batch["motion_valid"][:, start + horizon]
            count = int(valid.sum().cpu())
            if count == 0: continue
            error = torch.linalg.vector_norm(position - target, dim=-1)
            persistence = torch.linalg.vector_norm(start_position - target, dim=-1)
            outside = ((position.abs() > 1).any(-1) & valid).sum()
            results[horizon] = {
                "error_sum": float((error * valid).sum().cpu()),
                "persistence_sum": float((persistence * valid).sum().cpu()),
                "metric_error_sum": float((error * valid).sum().cpu() * batch["scale"].cpu()),
                "count": count, "outside": int(outside.cpu()),
                "step_sum": float((torch.linalg.vector_norm(output.displacement, dim=-1) * valid).sum().cpu()),
            }
    return results


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("checkpoint")
    parser.add_argument("caches", nargs="+")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--horizons", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--output", default="runs/velocity_feedback_sweep.csv")
    args = parser.parse_args()
    if any(not 0 <= alpha <= 1 for alpha in args.alphas):
        raise ValueError("alphas must lie in [0,1]")
    device = torch.device(args.device)
    scenes = [torch.load(path, map_location="cpu", weights_only=True) for path in args.caches]
    model = load_model(args.checkpoint, scenes[0], device)
    aggregate = {(alpha, horizon): {k: 0.0 for k in ["error_sum", "persistence_sum", "metric_error_sum", "count", "outside", "step_sum"]} for alpha in args.alphas for horizon in args.horizons}
    max_horizon = max(args.horizons)
    for scene_index, scene in enumerate(scenes, 1):
        batch = batched_scene(scene, device)
        starts = range(1, scene["points"].shape[0] - max_horizon, args.stride)
        for start in starts:
            for alpha in args.alphas:
                result = evaluate_window(model, batch, start, set(args.horizons), alpha)
                for horizon, values in result.items():
                    for key, value in values.items(): aggregate[(alpha, horizon)][key] += value
        print(f"processed scene {scene_index}/{len(scenes)}", flush=True)
    rows = []
    for alpha in args.alphas:
        for horizon in sorted(args.horizons):
            values = aggregate[(alpha, horizon)]; count = max(values["count"], 1)
            error = values["error_sum"] / count; persistence = values["persistence_sum"] / count
            rows.append({
                "alpha": alpha, "horizon": horizon, "particle_mean": error,
                "persistence_mean": persistence, "persistence_ratio": error / max(persistence, 1e-12),
                "metric_mean_mm": values["metric_error_sum"] / count * 1000,
                "outside_fraction": values["outside"] / count,
                "final_step_mean": values["step_sum"] / count, "valid_points": int(values["count"]),
            })
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(f"alpha={row['alpha']:.2f} h={row['horizon']:2d} error={row['particle_mean']:.6g} ratio={row['persistence_ratio']:.3f} mm={row['metric_mean_mm']:.2f} outside={row['outside_fraction']:.3%}")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
