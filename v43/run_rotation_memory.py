#!/usr/bin/env python3
"""Build and validate the isolated V4.3 rigid+soft rotation-memory branch.

The full three-seed matrix is deliberately not launched by default.  `--smoke`
builds the real train-only bank and performs one CPU optimizer step through the
identity-anchored reader; lab training expansion remains an explicit action.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "v2" / "src"), str(ROOT / "v3" / "src"),
                str(ROOT / "v4" / "src")]

from mpm_dino_v4.v42_rotation_audit import proper_kabsch, rotation_vector  # noqa: E402
from mpm_dino_v4.v43_rotation_memory import (  # noqa: E402
    CompactRotationReader, RotationBank, RotationMemoryEntry, geodesic_radians,
)


def build_rotation_bank(dataset_root: Path, manifest: dict) -> RotationBank:
    entries = []
    episode_ids = manifest["splits"]["train"]["panel_z"]
    for episode_id in episode_ids:
        row = manifest["episodes"][episode_id]
        if row["family"] not in {"rigid", "soft_body"}:
            continue
        with np.load(dataset_root / row["trajectory"], allow_pickle=False) as data:
            positions = data["trajectory_positions_m"].astype(np.float64)
            active = data["point_active"].astype(bool)
        with np.load(dataset_root / row["object_static"], allow_pickle=False) as data:
            dino = torch.from_numpy(data["dino_features"].astype(np.float32))
            dino_valid = torch.from_numpy(data["dino_valid"].astype(bool))
        point_valid_np = active[0] & active[1]
        reference = positions[1]
        centre = reference[point_valid_np].mean(0)
        radius = max(float(np.linalg.norm(reference[point_valid_np] - centre, axis=1).max()), 1e-12)
        coordinates = torch.from_numpy(((reference - centre) / radius).astype(np.float32))
        vectors, valid = [], []
        for frame in range(2, 61):
            mask = point_valid_np & active[frame]
            rotation, _, ratio = proper_kabsch(reference, positions[frame], mask)
            vectors.append(rotation_vector(rotation))
            valid.append(ratio >= 1e-3)
        normalized_dino = torch.nn.functional.normalize(dino, dim=-1)
        observed = torch.from_numpy(point_valid_np) & dino_valid
        pooled = torch.nn.functional.normalize(normalized_dino[observed].mean(0), dim=0)
        entries.append(RotationMemoryEntry(
            uid=row["uid"], family=row["family"], split="train", panel=row.get("panel", "Z"),
            coordinates=coordinates, dino=normalized_dino, dino_valid=dino_valid,
            point_valid=torch.from_numpy(point_valid_np), pooled_dino=pooled,
            rotation_vectors=torch.from_numpy(np.asarray(vectors, dtype=np.float32)),
            kabsch_valid=torch.tensor(valid), event_phase=torch.linspace(0, 1, 59),
            geometry_scale=radius, representation="proper_kabsch_rotvec_frame1",
            target_provenance=episode_id + ":trajectory_positions_m:frame1_kabsch",
        ))
    return RotationBank(entries)


def optimizer_smoke(bank: RotationBank, max_degrees: float) -> float:
    torch.manual_seed(42)
    # Compact bank tokens: pooled DINO, trajectory rotvec, validity and source family audit bit.
    memory_dim = bank.entries[0].pooled_dino.numel() + 5
    model = CompactRotationReader(32, memory_dim, hidden_dim=32, heads=4,
                                  max_degrees=max_degrees)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    query = torch.randn(1, 59, 32)
    selected = bank.entries[:min(3, len(bank.entries))]
    rows = []
    for entry in selected:
        family_bit = 1. if entry.family == "soft_body" else 0.
        rows.append(torch.cat((entry.pooled_dino[None].expand(59, -1), entry.rotation_vectors,
                              entry.kabsch_valid[:, None].float(),
                              torch.full((59, 1), family_bit)), -1))
    memory = torch.stack(rows, 1)[None]
    valid = torch.stack([entry.kabsch_valid for entry in selected], 1)[None]
    prediction, _, _ = model(query, memory, valid)
    target = torch.eye(3).expand_as(prediction).clone()
    loss = geodesic_radians(prediction, target).mean()
    loss.backward(); optimizer.step()
    return float(loss.detach())


def main(args):
    matrix = json.loads(args.matrix.read_text())
    if not matrix.get("frozen") or matrix.get("test_data_used"):
        raise ValueError("rotation matrix must be frozen and test-sealed")
    manifest = json.loads(args.manifest.read_text())
    bank = build_rotation_bank(args.dataset, manifest)
    args.runs.mkdir(parents=True, exist_ok=True)
    bank.save(args.runs / "rotation_bank.pt")
    report = {"status": "bank_complete", "bank_sha256": bank.content_sha256,
              "uids": len(bank.entries), "families": {family: sum(e.family == family for e in bank.entries)
              for family in ("rigid", "soft_body")}, "test_data_used": False}
    if args.smoke:
        report["cpu_optimizer_smoke_loss_rad"] = optimizer_smoke(bank, matrix["max_residual_degrees"])
        report["status"] = "smoke_complete"
    (args.runs / "RUN_COMPLETE.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("v41/dataset"))
    parser.add_argument("--manifest", type=Path, default=Path("v41/manifests/v41_uid_splits.json"))
    parser.add_argument("--matrix", type=Path, default=Path("v43/ROTATION_MATRIX.json"))
    parser.add_argument("--runs", type=Path, default=Path("v43/run/rotation_memory"))
    parser.add_argument("--device", default="cuda")  # reserved for explicit full-matrix runner
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=True)
    main(parser.parse_args())
