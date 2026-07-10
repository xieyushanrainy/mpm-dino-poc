#!/usr/bin/env python3
"""Extract multiple frozen DINO block feature maps with one model load."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from transformers import AutoImageProcessor, AutoModel


def first_rgb(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    ok, bgr = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def project(points, intrinsic, c2w):
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    camera = (np.linalg.inv(c2w) @ homogeneous.T).T[:, :3]
    uv = camera[:, :2] / np.maximum(camera[:, 2:3], 1e-8)
    uv[:, 0] = uv[:, 0] * intrinsic[0, 0] + intrinsic[0, 2]
    uv[:, 1] = uv[:, 1] * intrinsic[1, 1] + intrinsic[1, 2]
    return uv, camera[:, 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blocks", type=int, nargs="+", default=[6, 9, 11])
    parser.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--depth-tolerance", type=float, default=0.015)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True).eval().to(args.device)
    if any(block < 0 or block >= model.config.num_hidden_layers for block in args.blocks):
        raise ValueError(f"blocks must be in [0, {model.config.num_hidden_layers - 1}]")
    registers = int(getattr(model.config, "num_register_tokens", 4))
    patch = int(getattr(model.config, "patch_size", 16))
    scene_paths = sorted(path for path in args.scenes.iterdir() if (path / "final_data.pkl").exists())
    for scene in scene_paths:
        with (scene / "final_data.pkl").open("rb") as handle:
            raw = pickle.load(handle)
        points = np.asarray(raw["object_points"][0], dtype=np.float32)
        metadata = json.loads((scene / "metadata.json").read_text())
        intrinsic = np.asarray(metadata["intrinsics"][0], dtype=np.float32)
        with (scene / "calibrate.pkl").open("rb") as handle:
            c2w = np.asarray(pickle.load(handle)[0], dtype=np.float32)
        image = first_rgb(scene / "color" / "0.mp4")
        depth = np.load(scene / "depth" / "0" / "0.npy").astype(np.float32) / 1000.0
        inputs = processor(images=image, return_tensors="pt", do_resize=False, do_center_crop=False)
        pixels = inputs["pixel_values"].to(args.device)
        with torch.no_grad():
            outputs = model(pixel_values=pixels, output_hidden_states=True)
        uv, z = project(points, intrinsic, c2w)
        height, width = image.shape[:2]
        rounded = np.rint(uv).astype(int)
        inside = ((z > 0) & (rounded[:, 0] >= 0) & (rounded[:, 0] < width)
                  & (rounded[:, 1] >= 0) & (rounded[:, 1] < height))
        direct = np.zeros(len(points), dtype=bool)
        ids = np.where(inside)[0]
        direct[ids] = np.abs(depth[rounded[ids, 1], rounded[ids, 0]] - z[ids]) <= args.depth_tolerance
        if not direct.any():
            raise RuntimeError(f"no camera-0-visible tracks in {scene.name}")
        grid = torch.from_numpy(uv).to(args.device, torch.float32)
        grid[:, 0] = 2 * grid[:, 0] / max(width - 1, 1) - 1
        grid[:, 1] = 2 * grid[:, 1] / max(height - 1, 1) - 1
        missing = ~direct
        nearest = cKDTree(points[direct]).query(points[missing], k=1)[1]
        for block in args.blocks:
            tokens = outputs.hidden_states[block + 1]
            h, w = pixels.shape[-2] // patch, pixels.shape[-1] // patch
            feature_map = tokens[:, 1 + registers:].transpose(1, 2).reshape(1, -1, h, w)
            sampled = torch.nn.functional.grid_sample(
                feature_map, grid.view(1, 1, -1, 2), align_corners=True,
            )[0, :, 0].T.float().cpu().numpy()
            sampled[missing] = sampled[direct][nearest]
            output = args.output_dir / f"block{block:02d}" / f"{scene.name}.npz"
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output, features=sampled.astype(np.float32), imputed=missing)
        print(f"{scene.name}: tracks={len(points)} imputed={missing.mean():.1%}", flush=True)


if __name__ == "__main__":
    main()
