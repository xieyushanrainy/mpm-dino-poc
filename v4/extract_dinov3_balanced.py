#!/usr/bin/env python3
"""Add point-aligned DINOv3 and matched reprojected DINOv2 features to V4 samples."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


MODELS = (
    ("dinov3_features", "dinov3_valid", "facebook/dinov3-vitb16-pretrain-lvd1689m", 512),
    ("dinov2_reprojected_features", "dinov2_reprojected_valid", "facebook/dinov2-small", 518),
)
VIEWS = ("back", "front", "left", "right")
AZIMUTH = {"front": 0.0, "right": 90.0, "back": 180.0, "left": 270.0}


def camera_to_world(azimuth_degrees: float) -> np.ndarray:
    azimuth = math.radians(azimuth_degrees); elevation = math.radians(15.0); distance = 3.2
    location = np.asarray([
        distance * math.cos(elevation) * math.cos(azimuth),
        distance * math.cos(elevation) * math.sin(azimuth),
        distance * math.sin(elevation),
    ])
    z_axis = location / np.linalg.norm(location)
    x_axis = np.asarray([-math.sin(azimuth), math.cos(azimuth), 0.0])
    y_axis = np.cross(z_axis, x_axis)
    matrix = np.eye(4); matrix[:3, 0] = x_axis; matrix[:3, 1] = y_axis
    matrix[:3, 2] = z_axis; matrix[:3, 3] = location
    return matrix


CAMERAS = {view: camera_to_world(AZIMUTH[view]) for view in VIEWS}


def projections(points: np.ndarray, resolution: int, occlusion_grid: int, tolerance: float):
    # PBR rendering centres every object in a 1.6-unit cube. Physics scales this by 0.625,
    # centres X/Y, and shifts the lower Z bound to the drop height.
    vertical_shift = 0.5 * (float(points[:, 2].min()) + float(points[:, 2].max()))
    render = (points - np.asarray([0.0, 0.0, vertical_shift])) / 0.625
    homogeneous = np.concatenate([render, np.ones((len(render), 1))], axis=1)
    pixels, depths, in_frames = [], [], []
    for view in VIEWS:
        camera = (np.linalg.inv(CAMERAS[view]) @ homogeneous.T).T[:, :3]
        depth = -camera[:, 2]
        u = (0.5 + camera[:, 0] / np.maximum(depth, 1e-12) * 52.0 / 36.0) * resolution
        v = (0.5 - camera[:, 1] / np.maximum(depth, 1e-12) * 52.0 / 36.0) * resolution
        pixel = np.stack([u, v], axis=1)
        in_frame = (depth > 0) & (u >= 0) & (u < resolution) & (v >= 0) & (v < resolution)
        pixels.append(pixel); depths.append(depth); in_frames.append(in_frame)
    pixels = np.stack(pixels, axis=1); depths = np.stack(depths, axis=1)
    in_frames = np.stack(in_frames, axis=1); visible = np.zeros_like(in_frames)
    for view_index in range(len(VIEWS)):
        cell = np.clip((pixels[:, view_index] / resolution * occlusion_grid).astype(int), 0, occlusion_grid - 1)
        key = cell[:, 1] * occlusion_grid + cell[:, 0]
        minimum = np.full(occlusion_grid * occlusion_grid, np.inf)
        np.minimum.at(minimum, key[in_frames[:, view_index]], depths[in_frames[:, view_index], view_index])
        visible[:, view_index] = in_frames[:, view_index] & (
            depths[:, view_index] <= minimum[key] + tolerance
        )
    return pixels.astype(np.float32), visible


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False) as file:
        temporary = Path(file.name)
    try:
        np.savez_compressed(temporary, **arrays); temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(model_name: str, device_name: str):
    import torch
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))
    model = AutoModel.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))
    model.eval().to(device_name)
    return torch, processor, model


def dense_maps(torch, processor, model, images: list[Image.Image], input_size: int, device: str):
    import torch.nn.functional as functional
    resized = [image.resize((input_size, input_size), Image.Resampling.LANCZOS) for image in images]
    inputs = processor(images=resized, return_tensors="pt", do_resize=False, do_center_crop=False)
    pixels = inputs.pixel_values.to(device=device, dtype=next(model.parameters()).dtype)
    with torch.inference_mode():
        hidden = model(pixel_values=pixels).last_hidden_state
    patch = int(model.config.patch_size); registers = int(getattr(model.config, "num_register_tokens", 0))
    grid = input_size // patch; tokens = hidden[:, 1 + registers:, :]
    if tokens.shape[1] != grid * grid:
        raise ValueError(f"unexpected patch-token count {tokens.shape[1]} for {model.config._name_or_path}")
    maps = tokens.reshape(len(images), grid, grid, -1).permute(0, 3, 1, 2).float().cpu()
    return functional.normalize(maps, dim=1)


def sample_maps(torch, maps, projected: np.ndarray, visible: np.ndarray, resolution: int):
    import torch.nn.functional as functional
    count, dimension = len(projected), maps.shape[1]
    sums = torch.zeros((count, dimension), dtype=torch.float32); observations = torch.zeros(count, dtype=torch.int16)
    for view in range(len(VIEWS)):
        indices_np = np.flatnonzero(visible[:, view])
        if not len(indices_np): continue
        xy = projected[indices_np, view]
        grid = np.empty((1, len(indices_np), 1, 2), dtype=np.float32)
        grid[0, :, 0, 0] = 2.0 * (xy[:, 0] + 0.5) / resolution - 1.0
        grid[0, :, 0, 1] = 2.0 * (xy[:, 1] + 0.5) / resolution - 1.0
        sampled = functional.grid_sample(
            maps[view:view + 1], torch.from_numpy(grid), mode="bilinear",
            padding_mode="border", align_corners=False,
        )[0, :, :, 0].transpose(0, 1)
        sampled = functional.normalize(sampled, dim=-1); indices = torch.from_numpy(indices_np)
        sums[indices] += sampled; observations[indices] += 1
    valid = observations > 0; result = torch.zeros_like(sums)
    result[valid] = functional.normalize(sums[valid] / observations[valid, None].float(), dim=-1)
    return result.numpy().astype(np.float16), valid.numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--occlusion-grid", type=int, default=64)
    parser.add_argument("--occlusion-tolerance", type=float, default=0.05)
    parser.add_argument("--overwrite-output", action="store_true")
    args = parser.parse_args()
    source, output = Path(args.source_dir).resolve(), Path(args.output_dir).resolve()
    if output == source: raise SystemExit("output-dir must differ from source-dir")
    if not output.exists(): shutil.copytree(source, output)
    elif args.overwrite_output:
        shutil.rmtree(output); shutil.copytree(source, output)
    dataset = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    total = len(dataset["objects"])
    import torch
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else: device = args.device
    for feature_key, valid_key, model_name, input_size in MODELS:
        torch_module, processor, model = load_model(model_name, device)
        for index, record in enumerate(dataset["objects"], 1):
            sample_path = output / record["sample"]
            with np.load(sample_path, allow_pickle=False) as saved: arrays = {name: saved[name] for name in saved.files}
            if feature_key in arrays and valid_key in arrays:
                print(f"[{index}/{total}] {feature_key} {record['uid']}: already complete", flush=True); continue
            points = arrays["trajectory_positions_m"][0].astype(np.float64)
            images = []
            for view in VIEWS:
                path = output / "objects" / record["uid"] / "images" / f"{view}.webp"
                with Image.open(path) as opened: images.append(opened.convert("RGB").copy())
            resolution = images[0].width
            projected, visible = projections(points, resolution, args.occlusion_grid, args.occlusion_tolerance)
            maps = dense_maps(torch_module, processor, model, images, input_size, device)
            features, valid = sample_maps(torch_module, maps, projected, visible, resolution)
            arrays[feature_key] = features; arrays[valid_key] = valid; atomic_npz(sample_path, arrays)
            metadata_path = sample_path.with_name("metadata.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.setdefault("arrays", {})[feature_key] = {"shape": list(features.shape), "dtype": str(features.dtype)}
            metadata["arrays"][valid_key] = {"shape": list(valid.shape), "dtype": str(valid.dtype)}
            metadata.setdefault("dino_versions", {})[feature_key] = {
                "model": model_name, "input_size": input_size, "feature_dimension": features.shape[1],
                "aggregation": "mean of visible-view L2-normalized patch features, then L2-normalized",
                "projection": "deterministic packaged-camera reconstruction with point-cloud depth mask",
                "occlusion_grid": args.occlusion_grid, "occlusion_tolerance_render_units": args.occlusion_tolerance,
                "valid_point_count": int(valid.sum()),
            }
            metadata.setdefault("packaging", {})["sample_sha256"] = sha256(sample_path)
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"[{index}/{total}] {feature_key} {record['uid']}: {features.shape}, valid={valid.sum()}", flush=True)
        del model, processor; gc.collect()
        if device == "mps": torch.mps.empty_cache()
        if device == "cuda": torch.cuda.empty_cache()
    dataset.setdefault("contents", {})["feature_versions"] = {
        "dino_features": "facebook/dinov2-small; original mesh-exact correspondence",
        "dinov2_reprojected_features": "facebook/dinov2-small; reconstructed correspondence control",
        "dinov3_features": "facebook/dinov3-vitb16-pretrain-lvd1689m; reconstructed correspondence",
    }
    (output / "dataset.json").write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes = {
        "dataset.json": sha256(output / "dataset.json"),
        "samples": {record["sample"]: sha256(output / record["sample"]) for record in dataset["objects"]},
    }
    audit_path = output / "SELECTION_AUDIT.json"
    if audit_path.exists(): hashes["SELECTION_AUDIT.json"] = sha256(audit_path)
    (output / "PACKAGE_SHA256.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "DINOV3_COMPLETE").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__": main()
