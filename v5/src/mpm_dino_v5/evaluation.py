from __future__ import annotations

from collections import defaultdict

import torch

from .data import model_inputs, targets_and_stages
from .losses import event_normalized_deformation_mse, rotation_geodesic_mean


@torch.no_grad()
def evaluate_causal_model(model, loader, device="cpu", use_identity_rotation=False) -> dict:
    device = torch.device(device)
    model.eval()
    totals = defaultdict(float)
    uid_scores = defaultdict(list)
    batches = 0
    for raw in loader:
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in raw.items()}
        targets, stages = targets_and_stages(batch)
        output = model(**model_inputs(batch), use_identity_rotation=use_identity_rotation)
        objective = event_normalized_deformation_mse(
            output.canonical_displacement, targets.displacement,
            batch["target_mask"], targets.valid_rotation,
            stages.labels[:, 1:], targets.radius,
        )
        com_error = torch.sqrt(
            ((output.com - targets.com).square().sum(-1) * batch["target_mask"].any(2)).sum()
            / batch["target_mask"].any(2).sum().clamp_min(1)
        )
        rotation = rotation_geodesic_mean(
            output.rotation, targets.rotation,
            targets.valid_rotation & batch["target_mask"].any(2),
        ) * (180 / torch.pi)
        totals["event_normalized_mse"] += float(objective)
        totals["com_rmse_m"] += float(com_error)
        totals["rotation_geodesic_deg"] += float(rotation)
        for uid in batch["uid"]:
            uid_scores[uid].append(float(objective))
        batches += 1
    return {
        "event_normalized_mse": totals["event_normalized_mse"] / max(batches, 1),
        "com_rmse_m": totals["com_rmse_m"] / max(batches, 1),
        "rotation_geodesic_deg": totals["rotation_geodesic_deg"] / max(batches, 1),
        "per_uid_event_normalized_mse": {
            uid: sum(values) / len(values) for uid, values in sorted(uid_scores.items())
        },
        "batches": batches,
        "test_data_used": False,
    }


def aggregate_seed_scores(results: dict[int, dict]) -> dict:
    required = {42, 123, 456}
    if set(results) != required:
        raise ValueError("V5 aggregation requires seeds 42, 123 and 456")
    values = {seed: float(result["event_normalized_mse"]) for seed, result in results.items()}
    mean = sum(values.values()) / 3
    return {
        "seeds": values,
        "mean": mean,
        "base_qualified": mean < 0.91098,
        "success": mean <= 0.7025,
        "memory_eligible": 0.7025 < mean < 0.91098,
    }

