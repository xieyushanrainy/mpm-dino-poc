from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mpm_dino_v4.v41_data import MODEL_INPUT_KEYS, V41TrajectoryDataset
from mpm_dino_v4.v42_gate2 import load_gate1e_source


DEFAULT_UIDS = (
    "6132c68594544694b753b305b50bf66b",
    "877658e8ec83457d9c3673c651ac1395",
)


def _sample_indices(mask: np.ndarray, count: int) -> np.ndarray:
    valid = np.flatnonzero(mask)
    if len(valid) <= count:
        return valid
    return valid[np.linspace(0, len(valid) - 1, count).round().astype(int)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="v41/dataset")
    parser.add_argument("--manifest", default="v41/manifests/v41_uid_splits.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gate1e-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--uids", nargs="+", default=list(DEFAULT_UIDS))
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    dataset = V41TrajectoryDataset(
        args.dataset, manifest, "validation", dino_mode="zero", seed=456,
        families=("soft_body",),
    )
    model, _ = load_gate1e_source(
        args.gate1e_checkpoint, local_mode="geometry", device="cpu",
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    name_by_uid = {
        "6132c68594544694b753b305b50bf66b": "Creeper Pillow",
        "877658e8ec83457d9c3673c651ac1395": "Poly Pillow",
    }
    requested = set(args.uids)
    records = []
    for index, row in enumerate(dataset.rows):
        if row["uid"] not in requested or row["initial_velocity_regime"] != "zero":
            continue
        item = dataset[index]
        inputs = {
            key: item[key].unsqueeze(0) if torch.is_tensor(item[key]) else item[key]
            for key in MODEL_INPUT_KEYS
        }
        with torch.no_grad():
            output = model(**inputs)
        truth = torch.cat((item["x0"][None], item["x1"][None], item["target"]), dim=0)
        prediction = torch.cat((item["x0"][None], item["x1"][None], output.position[0]), dim=0)
        indices = _sample_indices(item["input_mask"].numpy(), args.points)
        truth_np = truth[:, indices].numpy()
        prediction_np = prediction[:, indices].numpy()
        error = np.sqrt(np.mean((prediction_np - truth_np) ** 2, axis=(1, 2)))
        center = truth_np.reshape(-1, 3).mean(axis=0)
        extent = float(np.max(np.abs(truth_np - center)))
        records.append({
            "uid": row["uid"],
            "name": name_by_uid.get(row["uid"], row["uid"][:8]),
            "episode": row["episode_id"],
            "truth": np.round(truth_np, 5).tolist(),
            "prediction": np.round(prediction_np, 5).tolist(),
            "rmseMm": np.round(error * 1000.0, 3).tolist(),
            "center": np.round(center, 5).tolist(),
            "extent": round(extent, 5),
        })
    if {record["uid"] for record in records} != requested:
        raise RuntimeError("not all requested validation zero-velocity episodes were found")
    records.sort(key=lambda record: args.uids.index(record["uid"]))
    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "epoch": int(checkpoint["epoch"]),
        "seed": int(checkpoint["config"]["seed"]),
        "space": "world coordinates",
        "frames": 61,
        "fps": 30,
        "sampledPoints": int(len(records[0]["truth"][0])),
        "objects": records,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    main()
