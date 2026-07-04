"""Extract camera-0 DINOv3 features and attach them to PhysTwin tracks.

This command downloads/loads the requested Hugging Face checkpoint. Users are
responsible for accepting and complying with its model-weight licence.
"""
import argparse
import json
import os
import pickle
from pathlib import Path

# This project is PyTorch-only. Prevent Transformers from importing an unrelated
# TensorFlow installation, which may have an incompatible compiled NumPy ABI.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from transformers import AutoImageProcessor, AutoModel


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def first_rgb(video: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    ok, bgr = capture.read(); capture.release()
    if not ok:
        raise RuntimeError(f"could not read {video}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def project(points_world, intrinsic, c2w):
    homogeneous = np.concatenate([points_world, np.ones((len(points_world), 1))], 1)
    camera = (np.linalg.inv(c2w) @ homogeneous.T).T[:, :3]
    uv = camera[:, :2] / np.maximum(camera[:, 2:3], 1e-8)
    uv[:, 0] = uv[:, 0] * intrinsic[0, 0] + intrinsic[0, 2]
    uv[:, 1] = uv[:, 1] * intrinsic[1, 1] + intrinsic[1, 2]
    return uv, camera[:, 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", help="PhysTwin scenario directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--depth-tolerance", type=float, default=0.015)
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()
    scene = Path(args.scene)
    raw = pickle.load(open(scene / "final_data.pkl", "rb"))
    points = np.asarray(raw["object_points"][0], dtype=np.float32)
    metadata = json.loads((scene / "metadata.json").read_text())
    intrinsic = np.asarray(metadata["intrinsics"][0], dtype=np.float32)
    c2w = np.asarray(pickle.load(open(scene / "calibrate.pkl", "rb"))[0], dtype=np.float32)
    image = first_rgb(scene / "color" / "0.mp4")
    depth = np.load(scene / "depth" / "0" / "0.npy").astype(np.float32) / 1000.0

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval().to(args.device)
    inputs = processor(images=image, return_tensors="pt", do_resize=False, do_center_crop=False)
    pixel_values = inputs["pixel_values"].to(args.device)
    with torch.no_grad():
        # We extract the final transformer block output via `last_hidden_state`.
        # These are the top-layer DINO token features before patch-grid sampling.
        tokens = model(pixel_values=pixel_values).last_hidden_state
    registers = int(getattr(model.config, "num_register_tokens", 4))
    patch = int(getattr(model.config, "patch_size", 16))
    h, w = pixel_values.shape[-2] // patch, pixel_values.shape[-1] // patch
    feature_map = tokens[:, 1 + registers:].transpose(1, 2).reshape(1, -1, h, w)

    uv, z = project(points, intrinsic, c2w)
    height, width = image.shape[:2]
    rounded = np.rint(uv).astype(int)
    inside = (z > 0) & (rounded[:, 0] >= 0) & (rounded[:, 0] < width) & (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
    direct = np.zeros(len(points), dtype=bool)
    ids = np.where(inside)[0]
    direct[ids] = np.abs(depth[rounded[ids, 1], rounded[ids, 0]] - z[ids]) <= args.depth_tolerance
    if not direct.any():
        raise RuntimeError("no camera-0-visible tracks; verify calibration and depth convention")
    grid = torch.from_numpy(uv).to(feature_map.device, torch.float32)
    grid[:, 0] = 2 * grid[:, 0] / max(width - 1, 1) - 1
    grid[:, 1] = 2 * grid[:, 1] / max(height - 1, 1) - 1
    sampled = torch.nn.functional.grid_sample(feature_map, grid.view(1, 1, -1, 2), align_corners=True)
    features = sampled[0, :, 0].T.float().cpu().numpy()
    missing = ~direct
    nearest = cKDTree(points[direct]).query(points[missing], k=1)[1]
    features[missing] = features[direct][nearest]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, features=features.astype(np.float32), imputed=missing)
    print(f"saved {features.shape} to {args.output}; imputed {missing.mean():.1%}")


if __name__ == "__main__":
    main()
