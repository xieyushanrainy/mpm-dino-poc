"""Render ground truth, recurrent, and teacher-forced particle predictions."""
import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from mpm_dino.model import ParticleGridSurrogate


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(checkpoint_path, scene, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    model = ParticleGridSurrogate(
        dino_dim=scene["dino"].shape[-1], base=config["base"], resolution=config["resolution"]
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint["epoch"]


def predict(model, positions, velocities, scene, t, particle_mask, device):
    dt = scene["dt"].to(device)
    controller = scene["controller"].to(device)
    controller_velocity = (controller[t + 1] - controller[t]) / dt
    with torch.no_grad():
        output = model(
            positions[None], velocities[None], scene["dino"].to(device)[None], particle_mask[None],
            scene["dino_imputed"].to(device)[None], controller[t][None], controller_velocity[None],
            torch.ones(1, controller.shape[1], dtype=torch.bool, device=device),
            scene["scale"].to(device)[None], dt[None],
        )
    return output.displacement[0]


def camera_project(points):
    # Fixed isometric camera: world z remains visually vertical.
    azimuth, elevation = np.deg2rad(42), np.deg2rad(24)
    forward = np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    up = np.cross(forward, right)
    return np.stack([points @ right, points @ up], 1)


def draw_panel(canvas, bounds, title, points, valid, controller, color, error_mm=None):
    x0, y0, width, height = bounds
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (30, 30, 30), -1)
    projected = camera_project(points[valid])
    controller_2d = camera_project(controller)

    def pixels(xy):
        # The cache normalization is designed around [-1,1]; allow margin for motion.
        uv = (xy + 1.35) / 2.7
        return np.stack([x0 + 18 + uv[:, 0] * (width - 36), y0 + height - 18 - uv[:, 1] * (height - 55)], 1).astype(int)

    for u, v in pixels(projected):
        if x0 <= u < x0 + width and y0 <= v < y0 + height:
            cv2.circle(canvas, (u, v), 2, color, -1, lineType=cv2.LINE_AA)
    for u, v in pixels(controller_2d):
        if x0 <= u < x0 + width and y0 <= v < y0 + height:
            cv2.circle(canvas, (u, v), 4, (70, 70, 245), -1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, title, (x0 + 12, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
    if error_mm is not None:
        cv2.putText(canvas, f"mean error: {error_mm:.2f} mm", (x0 + 12, y0 + 51), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("cache")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--velocity-feedback-alpha", type=float, default=1.0,
                        help="Blend derived displacement velocity with the previous recurrent velocity")
    args = parser.parse_args()
    if not 0 <= args.velocity_feedback_alpha <= 1:
        parser.error("--velocity-feedback-alpha must be in [0,1]")
    device = torch.device(args.device)
    scene = torch.load(args.cache, map_location="cpu", weights_only=True)
    model, epoch = load_model(args.checkpoint, scene, device)
    output = Path(args.output or (Path(args.checkpoint).parent / "rollout_visualization.mp4"))
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")

    points = scene["points"].to(device)
    dt = scene["dt"].to(device)
    start = 1
    stop = points.shape[0] - 1
    if args.max_frames is not None:
        stop = min(stop, start + args.max_frames)
    initial_mask = (scene["visible"][start] & scene["motion_valid"][start]).to(device)
    recurrent_position = points[start].clone()
    recurrent_velocity = (points[start] - points[start - 1]) / dt
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1440, 540))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {output}")
    rows = []
    for t in range(start, stop):
        recurrent_dx = predict(model, recurrent_position, recurrent_velocity, scene, t, initial_mask, device)
        recurrent_position = recurrent_position + recurrent_dx
        derived_velocity = recurrent_dx / dt
        recurrent_velocity = args.velocity_feedback_alpha * derived_velocity + (1 - args.velocity_feedback_alpha) * recurrent_velocity
        true_velocity = (points[t] - points[t - 1]) / dt
        teacher_position = points[t] + predict(model, points[t], true_velocity, scene, t, initial_mask, device)
        target = points[t + 1]
        valid = initial_mask & scene["visible"][t + 1].to(device) & scene["motion_valid"][t + 1].to(device)
        scale = float(scene["scale"])
        rec_error = float(torch.linalg.vector_norm(recurrent_position[valid] - target[valid], dim=-1).mean().cpu()) * scale * 1000
        tf_error = float(torch.linalg.vector_norm(teacher_position[valid] - target[valid], dim=-1).mean().cpu()) * scale * 1000
        rows.append({"frame": t + 1, "recurrent_mean_mm": rec_error, "teacher_forced_mean_mm": tf_error, "valid_points": int(valid.sum())})
        canvas = np.full((540, 1440, 3), 18, dtype=np.uint8)
        ctrl = scene["controller"][t + 1].numpy()
        mask_np = valid.cpu().numpy()
        draw_panel(canvas, (0, 0, 480, 500), "Ground truth", target.cpu().numpy(), mask_np, ctrl, (230, 190, 60))
        draw_panel(canvas, (480, 0, 480, 500), "Recurrent rollout", recurrent_position.cpu().numpy(), mask_np, ctrl, (80, 150, 255), rec_error)
        draw_panel(canvas, (960, 0, 480, 500), "Teacher-forced one step", teacher_position.cpu().numpy(), mask_np, ctrl, (100, 220, 120), tf_error)
        cv2.putText(canvas, f"epoch {epoch} | predicted frame {t + 1} | velocity alpha {args.velocity_feedback_alpha:g} | red = prescribed controller", (20, 528), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (225, 225, 225), 1, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()
    with open(csv_path, "w", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer_csv.writeheader(); writer_csv.writerows(rows)
    print(f"saved video: {output}")
    print(f"saved metrics: {csv_path}")
    last_valid = next((row for row in reversed(rows) if math.isfinite(row["recurrent_mean_mm"])), None)
    if last_valid:
        print(f"last valid frame {last_valid['frame']} recurrent mean: {last_valid['recurrent_mean_mm']:.3f} mm")
        print(f"last valid frame {last_valid['frame']} teacher-forced mean: {last_valid['teacher_forced_mean_mm']:.3f} mm")


if __name__ == "__main__":
    main()
