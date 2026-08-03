from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mpm_dino_v4.v41_data import MODEL_INPUT_KEYS, V41TrajectoryDataset
from mpm_dino_v4.v41_train import move
from mpm_dino_v4.v42_gate2 import load_gate1e_source
from mpm_dino_v4.v42_geometry import canonical_targets
from mpm_dino_v4.v42_stages import ImpactStage, derive_impact_stages


METHODS = ("rigid_proxy", "com_only", "ballistic_identity")
EVENT_STAGES = (
    int(ImpactStage.CONTACT_ONSET),
    int(ImpactStage.COMPRESSION),
    int(ImpactStage.PEAK_DEFORMATION),
)


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def first_true(mask):
    indices = torch.where(mask)[0]
    return int(indices[0]) if len(indices) else None


def patch_metrics(
    prediction, truth, point_mask, event_frames, reference_shape, radius,
    floor_z, threshold_fraction,
):
    threshold = threshold_fraction * radius
    truth_gap = truth[..., 2] - floor_z
    prediction_gap = prediction[..., 2] - floor_z
    truth_patch = (truth_gap <= threshold) & point_mask
    predicted_patch = (prediction_gap <= threshold) & point_mask
    selected = event_frames[:, None] & point_mask
    true_selected = truth_patch & event_frames[:, None]
    predicted_selected = predicted_patch & event_frames[:, None]
    true_positive = int((true_selected & predicted_selected).sum())
    predicted_positive = int(predicted_selected.sum())
    actual_positive = int(true_selected.sum())
    precision = safe_ratio(true_positive, predicted_positive)
    recall = safe_ratio(true_positive, actual_positive)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0
    )
    gap_error = (prediction_gap - truth_gap).abs()
    pointwise_gap_mae_mm = float(
        gap_error.masked_select(selected).mean() * 1000
    ) if selected.any() else None
    contact_gap_mae_mm = float(
        gap_error.masked_select(true_selected).mean() * 1000
    ) if true_selected.any() else None
    canonical_centres, world_centres = [], []
    frame_f1 = []
    matched_patch_frames = 0
    for frame in torch.where(event_frames)[0].tolist():
        true_mask = truth_patch[frame]
        pred_mask = predicted_patch[frame]
        tp = int((true_mask & pred_mask).sum())
        pred_count, true_count = int(pred_mask.sum()), int(true_mask.sum())
        p = safe_ratio(tp, pred_count)
        r = safe_ratio(tp, true_count)
        frame_f1.append(
            2 * p * r / (p + r)
            if p is not None and r is not None and p + r else 0.0
        )
        if pred_count and true_count:
            matched_patch_frames += 1
            canonical_centres.append(float(torch.linalg.vector_norm(
                reference_shape[pred_mask].mean(0)
                - reference_shape[true_mask].mean(0)
            ) / radius))
            world_centres.append(float(torch.linalg.vector_norm(
                prediction[frame, pred_mask].mean(0)
                - truth[frame, true_mask].mean(0)
            ) * 1000))
    truth_any = truth_patch.any(1)
    predicted_any = predicted_patch.any(1)
    truth_onset = first_true(truth_any)
    predicted_onset = first_true(predicted_any)
    onset_error = (
        predicted_onset - truth_onset
        if truth_onset is not None and predicted_onset is not None else None
    )
    onset_f1 = None
    if truth_onset is not None:
        true_mask = truth_patch[truth_onset]
        pred_mask = predicted_patch[truth_onset]
        tp = int((true_mask & pred_mask).sum())
        p = safe_ratio(tp, int(pred_mask.sum()))
        r = safe_ratio(tp, int(true_mask.sum()))
        onset_f1 = (
            2 * p * r / (p + r)
            if p is not None and r is not None and p + r else 0.0
        )
    return {
        "threshold_radius_fraction": threshold_fraction,
        "threshold_mm": float(threshold * 1000),
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "mean_frame_f1": sum(frame_f1) / len(frame_f1) if frame_f1 else None,
        "onset_patch_f1": onset_f1,
        "truth_geometric_onset": truth_onset,
        "predicted_geometric_onset": predicted_onset,
        "geometric_onset_error_frames": onset_error,
        "pointwise_gap_mae_mm": pointwise_gap_mae_mm,
        "true_contact_gap_mae_mm": contact_gap_mae_mm,
        "canonical_patch_centre_error_radius": (
            sum(canonical_centres) / len(canonical_centres)
            if canonical_centres else None
        ),
        "world_patch_centre_error_mm": (
            sum(world_centres) / len(world_centres)
            if world_centres else None
        ),
        "matched_patch_frames": matched_patch_frames,
        "event_frames": int(event_frames.sum()),
        "predicted_event_patch_points": predicted_positive,
        "true_event_patch_points": actual_positive,
    }


def mean_present(records, key):
    values = [r[key] for r in records if r[key] is not None]
    return sum(values) / len(values) if values else None


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[row["family"]].append(row)
        groups[f'{row["panel"]}/{row["family"]}'].append(row)
    summary = {}
    metric_keys = (
        "event_precision", "event_recall", "event_f1", "mean_frame_f1",
        "onset_patch_f1", "pointwise_gap_mae_mm",
        "true_contact_gap_mae_mm", "canonical_patch_centre_error_radius",
        "world_patch_centre_error_mm",
    )
    for group, group_rows in groups.items():
        summary[group] = {}
        for method in METHODS:
            summary[group][method] = {}
            thresholds = group_rows[0]["methods"][method]
            for threshold_key in thresholds:
                records = [
                    row["methods"][method][threshold_key] for row in group_rows
                ]
                values = {key: mean_present(records, key) for key in metric_keys}
                onset = [
                    record["geometric_onset_error_frames"]
                    for record in records
                    if record["geometric_onset_error_frames"] is not None
                ]
                values.update({
                    "episodes": len(records),
                    "onset_detected": len(onset),
                    "geometric_onset_mae_frames": (
                        sum(abs(value) for value in onset) / len(onset)
                        if onset else None
                    ),
                })
                summary[group][method][threshold_key] = values
    return summary


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
        output = model(**{key: batch[key] for key in MODEL_INPUT_KEYS})
        targets = canonical_targets(
            batch["x1"], batch["target"], batch["input_mask"],
            batch["target_mask"],
        )
        stages = derive_impact_stages(
            batch["x1"], batch["target"], batch["input_mask"],
            batch["target_mask"], batch["neighbour_indices"],
            batch["neighbour_mask"], batch["rest_edge_lengths"], batch["dt"],
            batch["gravity"], batch["floor_z"],
        )
        q = targets.reference_shape
        trajectories = {
            "rigid_proxy": output.com[:, :, None] + torch.einsum(
                "bni,btij->btnj", q, output.rotation,
            ),
            "com_only": output.com[:, :, None] + q[:, None],
            "ballistic_identity": output.ballistic_com[:, :, None] + q[:, None],
        }
        event_frames = torch.zeros_like(stages.labels[:, 1:], dtype=torch.bool)
        for stage in EVENT_STAGES:
            event_frames |= stages.labels[:, 1:].eq(stage)
        # Only initial persistent activity is supplied to causal methods.
        causal_mask = batch["input_mask"][:, None].expand_as(batch["target_mask"])
        methods = {}
        for method, trajectory in trajectories.items():
            methods[method] = {}
            for threshold in args.thresholds:
                key = f"{threshold:g}"
                methods[method][key] = patch_metrics(
                    trajectory[0], batch["target"][0], causal_mask[0],
                    event_frames[0], q[0], float(targets.radius[0]),
                    float(batch["floor_z"][0]), threshold,
                )
        rows.append({
            "uid": batch["uid"][0],
            "episode_id": batch["episode_id"][0],
            "family": batch["family"][0],
            "panel": batch["panel"][0],
            "stage_contact_onset": int(stages.contact_onset[0]),
            "methods": methods,
        })
    payload = {
        "experiment": "v42_causal_contact_patch_audit",
        "split": args.split,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(state["epoch"]),
        "checkpoint_seed": int(config["seed"]),
        "thresholds_radius_fraction": args.thresholds,
        "primary_threshold_radius_fraction": args.primary_threshold,
        "event_stages": ["contact_onset", "compression", "peak_deformation"],
        "test_data_used": args.split == "test",
        "causal_inputs": [
            "initial persistent points", "frozen predicted COM",
            "frozen predicted rotation", "ballistic COM", "floor height",
        ],
        "ground_truth_role": "evaluation only",
        "summary": aggregate(rows),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    primary = f"{args.primary_threshold:g}"
    print(json.dumps({
        group: {method: values[primary] for method, values in methods.items()}
        for group, methods in payload["summary"].items()
    }, indent=2))


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
        "--thresholds", type=float, nargs="+", default=[0.005, 0.01, 0.02],
    )
    parser.add_argument("--primary-threshold", type=float, default=0.01)
    parser.add_argument(
        "--output", default="v42/run/causal_contact_patch_audit/RESULTS.json",
    )
    args = parser.parse_args()
    if args.primary_threshold not in args.thresholds:
        parser.error("--primary-threshold must be included in --thresholds")
    main(args)
