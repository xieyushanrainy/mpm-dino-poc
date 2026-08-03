from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mpm_dino_v4.v41_data import MODEL_INPUT_KEYS, V41TrajectoryDataset
from mpm_dino_v4.v41_train import move
from mpm_dino_v4.v42_gate2 import load_gate1e_source
from mpm_dino_v4.v42_geometry import canonical_targets, rotation_geodesic
from mpm_dino_v4.v42_stages import derive_impact_stages


HORIZONS = (1, 8, 16, 30, 40, 59)


def point_rmse(prediction, target, mask):
    squared = (prediction - target).square().sum(-1)
    return torch.sqrt(
        (squared * mask).sum() / mask.sum().clamp_min(1)
    )


def frame_point_rmse(prediction, target, mask):
    squared = (prediction - target).square().sum(-1)
    return torch.sqrt(
        (squared * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    )


def mean_or_none(values):
    return sum(values) / len(values) if values else None


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[row["family"]].append(row)
        groups[f'{row["panel"]}/{row["family"]}'].append(row)
    output = {}
    scalar_keys = (
        "rigid_proxy_rmse_mm", "oracle_rigid_floor_rmse_mm",
        "predicted_com_rmse_mm", "predicted_rotation_error_deg",
        "predicted_com_oracle_rotation_rmse_mm",
        "oracle_com_predicted_rotation_rmse_mm",
        "rigid_proxy_nrmse", "oracle_rigid_floor_nrmse",
    )
    for group, records in groups.items():
        summary = {"episodes": len(records), "uids": len({r["uid"] for r in records})}
        for key in scalar_keys:
            summary[key] = mean_or_none([r[key] for r in records if r[key] is not None])
        onset = [r["contact_onset_error_frames"] for r in records if r["contact_onset_error_frames"] is not None]
        summary["contact_onset_detected"] = len(onset)
        summary["contact_onset_mae_frames"] = mean_or_none([abs(v) for v in onset])
        summary["horizon_rigid_proxy_rmse_mm"] = {
            str(h): mean_or_none([r["horizon_rigid_proxy_rmse_mm"][str(h)] for r in records])
            for h in HORIZONS
        }
        summary["horizon_oracle_rigid_floor_rmse_mm"] = {
            str(h): mean_or_none([r["horizon_oracle_rigid_floor_rmse_mm"][str(h)] for r in records])
            for h in HORIZONS
        }
        output[group] = summary
    return output


@torch.no_grad()
def main(args):
    device = torch.device(args.device)
    manifest = json.loads(Path(args.manifest).read_text())
    dataset = V41TrajectoryDataset(
        args.dataset, manifest, args.split, dino_mode="zero", seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    model, _ = load_gate1e_source(
        args.gate1e_checkpoint, local_mode="geometry", device=device,
        oracle_condition_dim=int(config["oracle_condition_dim"]),
        oracle_injection=config["oracle_injection"],
    )
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    rows = []
    for raw in loader:
        batch = move(raw, device)
        inputs = {key: batch[key] for key in MODEL_INPUT_KEYS}
        output = model(**inputs)
        targets = canonical_targets(
            batch["x1"], batch["target"], batch["input_mask"],
            batch["target_mask"],
        )
        mask = batch["target_mask"]
        q = targets.reference_shape
        predicted_rigid = output.com[:, :, None] + torch.einsum(
            "bni,btij->btnj", q, output.rotation,
        )
        oracle_rigid = targets.com[:, :, None] + torch.einsum(
            "bni,btij->btnj", q, targets.rotation,
        )
        predicted_com_oracle_rotation = output.com[:, :, None] + torch.einsum(
            "bni,btij->btnj", q, targets.rotation,
        )
        oracle_com_predicted_rotation = targets.com[:, :, None] + torch.einsum(
            "bni,btij->btnj", q, output.rotation,
        )
        rigid_frame_error = frame_point_rmse(
            predicted_rigid, batch["target"], mask,
        )
        oracle_frame_error = frame_point_rmse(
            oracle_rigid, batch["target"], mask,
        )
        valid_rotation = targets.valid_rotation
        rotation_error = rotation_geodesic(
            output.rotation, targets.rotation,
        ).masked_select(valid_rotation)
        truth_stages = derive_impact_stages(
            batch["x1"], batch["target"], batch["input_mask"], mask,
            batch["neighbour_indices"], batch["neighbour_mask"],
            batch["rest_edge_lengths"], batch["dt"], batch["gravity"],
            batch["floor_z"],
        )
        # The proxy must not consume future target activity. Persistent input
        # points are propagated causally across the predicted horizon.
        proxy_mask = batch["input_mask"][:, None].expand_as(mask)
        proxy_stages = derive_impact_stages(
            batch["x1"], predicted_rigid, batch["input_mask"], proxy_mask,
            batch["neighbour_indices"], batch["neighbour_mask"],
            batch["rest_edge_lengths"], batch["dt"], batch["gravity"],
            batch["floor_z"],
        )
        truth_onset = int(truth_stages.contact_onset[0])
        proxy_onset = int(proxy_stages.contact_onset[0])
        onset_error = (
            proxy_onset - truth_onset
            if truth_onset >= 0 and proxy_onset >= 0 else None
        )
        radius = float(targets.radius[0])
        row = {
            "uid": batch["uid"][0],
            "episode_id": batch["episode_id"][0],
            "family": batch["family"][0],
            "panel": batch["panel"][0],
            "radius_m": radius,
            "rigid_proxy_rmse_mm": float(point_rmse(
                predicted_rigid, batch["target"], mask,
            ) * 1000),
            "oracle_rigid_floor_rmse_mm": float(point_rmse(
                oracle_rigid, batch["target"], mask,
            ) * 1000),
            "predicted_com_rmse_mm": float(torch.sqrt(
                (output.com - targets.com).square().sum(-1).mean()
            ) * 1000),
            "predicted_rotation_error_deg": (
                float(rotation_error.mean() * 180 / math.pi)
                if rotation_error.numel() else None
            ),
            "predicted_com_oracle_rotation_rmse_mm": float(point_rmse(
                predicted_com_oracle_rotation, batch["target"], mask,
            ) * 1000),
            "oracle_com_predicted_rotation_rmse_mm": float(point_rmse(
                oracle_com_predicted_rotation, batch["target"], mask,
            ) * 1000),
            "rigid_proxy_nrmse": float(point_rmse(
                predicted_rigid, batch["target"], mask,
            ) / radius),
            "oracle_rigid_floor_nrmse": float(point_rmse(
                oracle_rigid, batch["target"], mask,
            ) / radius),
            "truth_contact_onset": truth_onset if truth_onset >= 0 else None,
            "proxy_contact_onset": proxy_onset if proxy_onset >= 0 else None,
            "contact_onset_error_frames": onset_error,
            "horizon_rigid_proxy_rmse_mm": {
                str(h): float(rigid_frame_error[0, h - 1] * 1000)
                for h in HORIZONS
            },
            "horizon_oracle_rigid_floor_rmse_mm": {
                str(h): float(oracle_frame_error[0, h - 1] * 1000)
                for h in HORIZONS
            },
        }
        rows.append(row)
    payload = {
        "experiment": "v42_frozen_com_rotation_rigid_proxy_audit",
        "split": args.split,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(state["epoch"]),
        "checkpoint_seed": int(config["seed"]),
        "test_data_used": args.split == "test",
        "definition": (
            "predicted_rigid[t,i] = frozen_COM[t] + reference_shape[i] "
            "@ frozen_rotation[t]; canonical deformation is exactly zero"
        ),
        "oracle_rigid_floor_definition": (
            "ground-truth COM plus ground-truth Kabsch rotation with zero "
            "canonical deformation; this is the irreducible rigid-only floor"
        ),
        "summary": summarize(rows),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="v41/dataset")
    parser.add_argument("--manifest", default="v41/manifests/v41_uid_splits.json")
    parser.add_argument(
        "--checkpoint",
        default="v42/checkpoints/v42_adapter_full_seed42_best.pt",
    )
    parser.add_argument("--gate1e-checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output", default="v42/run/rigid_proxy_audit/RESULTS.json",
    )
    main(parser.parse_args())
