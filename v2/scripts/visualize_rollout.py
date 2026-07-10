#!/usr/bin/env python3
"""Render ground truth, recurrent, and teacher-forced V2 particle predictions."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from mpm_dino_v2.cache import load_v2_cache
from mpm_dino_v2.model import ParticleGridSurrogate


def load_model(path, scene, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    model = ParticleGridSurrogate(
        dino_dim=scene["dino"].shape[-1], base=config["base"], resolution=config["resolution"],
        variant=config.get("variant", "fused"),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint["epoch"]


def predict(model, positions, velocities, scene, t, mask, device):
    dt = scene["dt"].to(device)
    controller = scene["controller"].to(device)
    controller_velocity = (controller[t + 1] - controller[t]) / dt
    with torch.no_grad():
        output = model(
            positions[None], velocities[None], scene["dino"].to(device)[None], mask[None],
            scene["dino_imputed"].to(device)[None], controller[t][None], controller_velocity[None],
            torch.ones((1, controller.shape[1]), dtype=torch.bool, device=device),
            scene["scale"].to(device)[None], dt[None], scene["x0"].to(device)[None],
            scene["neighbour_indices"].to(device)[None], scene["neighbour_mask"].to(device)[None],
            scene["rest_edge_lengths"].to(device)[None],
        )
    return output.displacement[0]


def project(points):
    azimuth, elevation = np.deg2rad(42), np.deg2rad(24)
    forward = np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    up = np.cross(forward, right)
    return np.stack((points @ right, points @ up), axis=1)


def draw_panel(canvas, bounds, title, points, valid, controller, color, error_mm=None):
    x0, y0, width, height = bounds
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (30, 30, 30), -1)

    def pixels(xy):
        uv = (xy + 1.35) / 2.7
        return np.stack((x0 + 18 + uv[:, 0] * (width - 36),
                         y0 + height - 18 - uv[:, 1] * (height - 55)), axis=1).astype(int)

    for u, v in pixels(project(points[valid])):
        if x0 <= u < x0 + width and y0 <= v < y0 + height:
            cv2.circle(canvas, (u, v), 2, color, -1, lineType=cv2.LINE_AA)
    for u, v in pixels(project(controller)):
        if x0 <= u < x0 + width and y0 <= v < y0 + height:
            cv2.circle(canvas, (u, v), 4, (70, 70, 245), -1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, title, (x0 + 12, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (240, 240, 240), 1, cv2.LINE_AA)
    if error_mm is not None:
        cv2.putText(canvas, f"mean error: {error_mm:.2f} mm", (x0 + 12, y0 + 51),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    device = torch.device(args.device)
    scene = load_v2_cache(args.cache)
    model, epoch = load_model(args.checkpoint, scene, device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    points, dt = scene["points"].to(device), scene["dt"].to(device)
    start, stop = 1, points.shape[0] - 1
    if args.max_frames:
        stop = min(stop, start + args.max_frames)
    initial_mask = (scene["particle_mask"] & scene["visible"][start] & scene["motion_valid"][start]).to(device)
    recurrent_position = points[start].clone()
    recurrent_velocity = (points[start] - points[start - 1]) / dt
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1440, 540))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {output}")
    rows = []
    for t in range(start, stop):
        recurrent_displacement = predict(model, recurrent_position, recurrent_velocity, scene, t, initial_mask, device)
        recurrent_position = recurrent_position + recurrent_displacement
        recurrent_velocity = recurrent_displacement / dt
        true_velocity = (points[t] - points[t - 1]) / dt
        teacher_position = points[t] + predict(model, points[t], true_velocity, scene, t, initial_mask, device)
        target = points[t + 1]
        valid = initial_mask & scene["visible"][t + 1].to(device) & scene["motion_valid"][t + 1].to(device)
        scale = float(scene["scale"])
        rec_error = float(torch.linalg.vector_norm(recurrent_position[valid] - target[valid], dim=-1).mean().cpu()) * scale * 1000
        tf_error = float(torch.linalg.vector_norm(teacher_position[valid] - target[valid], dim=-1).mean().cpu()) * scale * 1000
        rows.append({"frame": t + 1, "recurrent_mean_mm": rec_error,
                     "teacher_forced_mean_mm": tf_error, "valid_points": int(valid.sum())})
        canvas = np.full((540, 1440, 3), 18, dtype=np.uint8)
        controller = scene["controller"][t + 1].numpy()
        valid_numpy = valid.cpu().numpy()
        draw_panel(canvas, (0, 0, 480, 500), "Ground truth", target.cpu().numpy(), valid_numpy,
                   controller, (230, 190, 60))
        draw_panel(canvas, (480, 0, 480, 500), "V2 recurrent rollout", recurrent_position.cpu().numpy(),
                   valid_numpy, controller, (80, 150, 255), rec_error)
        draw_panel(canvas, (960, 0, 480, 500), "V2 teacher-forced", teacher_position.cpu().numpy(),
                   valid_numpy, controller, (100, 220, 120), tf_error)
        cv2.putText(canvas, f"retained rollout epoch {epoch} | predicted frame {t + 1} | red = controller",
                    (20, 528), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (225, 225, 225), 1, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()
    with output.with_suffix(".csv").open("w", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    print(f"saved video: {output}")
    print(f"saved metrics: {output.with_suffix('.csv')}")
    last_valid = next((row for row in reversed(rows) if math.isfinite(row["recurrent_mean_mm"])), None)
    if last_valid:
        print(f"last evaluable frame {last_valid['frame']} recurrent={last_valid['recurrent_mean_mm']:.3f} mm "
              f"teacher={last_valid['teacher_forced_mean_mm']:.3f} mm")


if __name__ == "__main__":
    main()
