from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from mpm_dino_v4.v41_data import V41TrajectoryDataset
from mpm_dino_v4.v42_geometry import canonical_targets
from mpm_dino_v4.v42_stages import ImpactStage, derive_impact_stages


def summarize_split(dataset):
    rows = []
    for index in range(len(dataset)):
        sample = dataset[index]
        target = sample["target"][None]
        target_mask = sample["target_mask"][None]
        geometry = canonical_targets(
            sample["x1"][None], target, sample["input_mask"][None],
            target_mask,
        )
        stages = derive_impact_stages(
            sample["x1"][None], target, sample["input_mask"][None],
            target_mask, sample["neighbour_indices"][None],
            sample["neighbour_mask"][None], sample["rest_edge_lengths"][None],
            sample["dt"][None], sample["gravity"][None],
            sample["floor_z"][None],
        )
        labels = stages.labels[0].cpu().tolist()
        displacement = geometry.displacement[0]
        mask = target_mask[0] & geometry.valid_rotation[0, :, None]
        canonical_rms = torch.sqrt(
            (displacement.square().sum(-1) * mask).sum(1)
            / mask.sum(1).clamp_min(1)
        )
        rows.append({
            "uid": sample["uid"],
            "episode_id": sample["episode_id"],
            "family": sample["family"],
            "panel": sample["panel"],
            "rotation_valid_frames": int(geometry.valid_rotation.sum()),
            "rotation_invalid_frames": int((~geometry.valid_rotation).sum()),
            "minimum_singular_ratio": float(
                (
                    geometry.singular_values[..., 1]
                    / geometry.singular_values[..., 0].clamp_min(1e-12)
                ).min()
            ),
            "contact_onset": int(stages.contact_onset[0]),
            "peak_start": int(stages.peak_start[0]),
            "peak_end": int(stages.peak_end[0]),
            "recovery_end": int(stages.recovery_end[0]),
            "stage_counts": {
                ImpactStage(key).name.lower(): value
                for key, value in sorted(Counter(labels).items())
            },
            "canonical_rms_m": [float(value) for value in canonical_rms],
            "deformation": [
                float(value) for value in stages.deformation[0].cpu()
            ],
            "floor_gap_m": [
                float(value) for value in stages.floor_gap[0].cpu()
            ],
            "excess_acceleration_m_s2": [
                float(value)
                for value in stages.excess_acceleration[0].cpu()
            ],
            "minimum_floor_gap_m": float(stages.floor_gap.min()),
            "maximum_excess_acceleration_m_s2": float(
                stages.excess_acceleration.max()
            ),
        })
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    summary = {}
    for family, values in sorted(by_family.items()):
        detected = [row for row in values if row["contact_onset"] >= 0]
        summary[family] = {
            "episodes": len(values),
            "uids": len({row["uid"] for row in values}),
            "contact_detected": len(detected),
            "contact_fraction": len(detected) / len(values),
            "rotation_invalid_frames": sum(
                row["rotation_invalid_frames"] for row in values
            ),
            "median_contact_onset": (
                float(np.median([row["contact_onset"] for row in detected]))
                if detected else None
            ),
        }
    return {"summary": summary, "episodes": rows}


def main():
    parser = argparse.ArgumentParser(
        description="V4.2 training/validation-only target and stage audit"
    )
    parser.add_argument("--dataset", type=Path, default=Path("v41/dataset"))
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("v41/manifests/v41_uid_splits.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("v42/target_stage_audit.json"),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    payload = {
        "experiment": "v42_gate0_target_stage_audit",
        "splits": ["train", "validation"],
        "test_data_used": False,
        "stage_thresholds": {
            "floor_gap_statistic": "mean of lowest 4 active surface points",
            "floor_gap_tail_points": 4,
            "adaptive_gap_m": "max(0.010, 0.25 * initial_gap)",
            "raw_excess_acceleration_gravity_fraction": 0.2,
            "contact_frame": "centred impulse frame + 1 saved frame",
            "peak_fraction": 0.95,
            "recovery_fraction": 0.2,
        },
        "kabsch_degeneracy_ratio": 1e-3,
        "results": {},
    }
    for split in ("train", "validation"):
        dataset = V41TrajectoryDataset(
            args.dataset, manifest, split, "zero", 42,
            families=("soft_body", "rigid"),
        )
        payload["results"][split] = summarize_split(dataset)
        print(split, payload["results"][split]["summary"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
