from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .model import masked_mean
from .v41_data import V41TrajectoryDataset
from .v41_train import move
from .v42_gate2 import _batch_targets_and_stages, train_v42_gate2
from .v42_stages import ImpactStage


TEMPORAL_DIM = 8  # seven stage indicators plus event-relative time
MATERIAL_DIM = 7
ORACLE_DIM = TEMPORAL_DIM + MATERIAL_DIM
ORACLE_VARIANTS = {
    "geometry_control": (False, False),
    "oracle_temporal": (True, False),
    "oracle_material": (False, True),
    "oracle_both": (True, True),
}
LOSS_CONTRACT = {
    "soft_deformation_amplification_cap": 1.0,
    "soft_deformation_quantile": 0.95,
    "soft_deformation_floor_fraction": 0.005,
    "family_balanced": True,
    "rigid_family_weight": 0.25,
    "rigid_zero_weight": 0.0,
}


def material_vector(metadata):
    """Fixed-scale simulator-source material descriptor (no fitted stats)."""
    soft = metadata["index_record"]["solver_route"] == "soft_body"
    bodies = metadata.get("simulation", {}).get("body_parameters", [])
    values = bodies[0] if len(bodies) == 1 else {}
    density = float(values.get("density_kg_m3", 0.0))
    young = float(values.get("youngs_modulus_pa", 0.0))
    damping = float(values.get("damping", 0.0))
    return torch.tensor([
        float(not soft), float(soft),
        (math.log10(density) - 2.5) / 1.5 if density > 0 else 0.0,
        float(values.get("friction", 0.0)),
        (math.log10(young) - 5.0) / 3.0 if young > 0 else 0.0,
        damping / 0.5,
        float(young > 0 or damping > 0),
    ], dtype=torch.float32)


def load_material_table(dataset_root):
    table = {}
    for path in Path(dataset_root).glob("objects/*/source_metadata.json"):
        metadata = json.loads(path.read_text())
        table[metadata["index_record"]["uid"]] = material_vector(metadata)
    if not table:
        raise FileNotFoundError("no object source_metadata.json files found")
    return table


def temporal_features(stages, target_frames=59):
    labels = stages.labels[:, 1:1 + target_frames]
    one_hot = F.one_hot(labels, num_classes=7).to(stages.deformation.dtype)
    saved_frames = torch.arange(
        1, target_frames + 1, device=labels.device,
        dtype=stages.deformation.dtype,
    )[None]
    onset = stages.contact_onset[:, None]
    relative = (saved_frames - onset.to(saved_frames.dtype)) / max(
        target_frames - 1, 1,
    )
    relative = relative.clamp(-1, 1)
    relative = torch.where(onset >= 0, relative, torch.zeros_like(relative))
    return torch.cat((one_hot, relative[..., None]), dim=-1)


class OracleConditionBuilder:
    def __init__(self, dataset_root, temporal, material):
        self.temporal = bool(temporal)
        self.material = bool(material)
        self.materials = load_material_table(dataset_root)

    def __call__(self, batch, stages):
        batch_size, frames = stages.labels.shape[0], stages.labels.shape[1] - 1
        result = stages.deformation.new_zeros(batch_size, frames, ORACLE_DIM)
        if self.temporal:
            result[..., :TEMPORAL_DIM] = temporal_features(stages, frames)
        if self.material:
            values = torch.stack([self.materials[uid] for uid in batch["uid"]])
            result[..., TEMPORAL_DIM:] = values.to(result)[:, None]
        return result.detach()


def train_oracle_variant(
    dataset_root, manifest, checkpoint, output, seed, variant, **kwargs,
):
    if variant not in ORACLE_VARIANTS:
        raise ValueError(f"unknown oracle variant: {variant}")
    temporal, material = ORACLE_VARIANTS[variant]
    builder = OracleConditionBuilder(dataset_root, temporal, material)
    return train_v42_gate2(
        dataset_root, manifest, checkpoint, output, seed,
        loss_options=LOSS_CONTRACT, stage_weight_mode="total_mass",
        oracle_condition_dim=ORACLE_DIM, condition_builder=builder,
        condition_name=variant, exploratory_control=True,
        experiment_name="v42_oracle_controlled_group_" + variant,
        model_contract_version="oracle_controlled_group_v1",
        **kwargs,
    )


EVENT_STAGES = (
    ImpactStage.CONTACT_ONSET,
    ImpactStage.COMPRESSION,
    ImpactStage.PEAK_DEFORMATION,
)


def event_normalized_canonical_mse(
    output, batch, targets, stages, _frame_weights, _losses,
):
    """Episode-amplitude-normalized MSE on soft event frames only.

    The detached RMS target displacement over contact/compression/peak frames
    is the scale. Consequently, predicting zero has loss one (up to numerical
    precision) for every nonzero episode, independent of deformation amplitude.
    """
    labels = stages.labels[:, 1:]
    selected = torch.zeros_like(labels, dtype=torch.bool)
    for stage in EVENT_STAGES:
        selected |= labels.eq(int(stage))
    valid = (
        batch["target_mask"]
        & targets.valid_rotation[:, :, None]
        & selected[:, :, None]
    )
    count = valid.sum((1, 2)).clamp_min(1)
    target_energy = (
        targets.displacement.square().sum(-1) * valid
    ).sum((1, 2)) / count
    floor = (1e-6 * targets.radius).square()
    scale_squared = target_energy.maximum(floor).detach()
    error_energy = (
        (output.canonical_displacement - targets.displacement)
        .square().sum(-1) * valid
    ).sum((1, 2)) / count
    return (error_energy / scale_squared).mean()


def train_event_normalized_variant(
    dataset_root, manifest, checkpoint, output, seed, variant, **kwargs,
):
    if variant not in {"geometry_control", "oracle_temporal"}:
        raise ValueError("focused diagnostic supports geometry and temporal only")
    temporal = variant == "oracle_temporal"
    builder = OracleConditionBuilder(dataset_root, temporal, False)
    return train_v42_gate2(
        dataset_root, manifest, checkpoint, output, seed,
        loss_options=LOSS_CONTRACT, stage_weight_mode="total_mass",
        oracle_condition_dim=ORACLE_DIM, condition_builder=builder,
        condition_name=variant, exploratory_control=True,
        objective_builder=event_normalized_canonical_mse,
        objective_name="event_frame_amplitude_normalized_canonical_mse_v1",
        dataset_families=("soft_body",), selection_mode="optimized",
        experiment_name="v42_event_normalized_" + variant,
        model_contract_version="event_normalized_temporal_control_v1",
        **kwargs,
    )


def train_upstream_temporal_variant(
    dataset_root, manifest, checkpoint, output, seed, variant, **kwargs,
):
    if variant not in {"geometry_control", "oracle_temporal"}:
        raise ValueError("upstream diagnostic supports geometry and temporal only")
    builder = OracleConditionBuilder(
        dataset_root, variant == "oracle_temporal", False,
    )
    return train_v42_gate2(
        dataset_root, manifest, checkpoint, output, seed,
        loss_options=LOSS_CONTRACT, stage_weight_mode="total_mass",
        oracle_condition_dim=ORACLE_DIM, oracle_injection="adapter",
        condition_builder=builder, condition_name=variant,
        exploratory_control=True,
        objective_builder=event_normalized_canonical_mse,
        objective_name="event_frame_amplitude_normalized_canonical_mse_v1",
        dataset_families=("soft_body",), selection_mode="optimized",
        experiment_name="v42_upstream_temporal_" + variant,
        model_contract_version="upstream_temporal_adapter_control_v1",
        **kwargs,
    )


def _event_mask(batch, targets, stages, stage=None):
    labels = stages.labels[:, 1:]
    if stage is None:
        selected = torch.zeros_like(labels, dtype=torch.bool)
        for value in EVENT_STAGES:
            selected |= labels.eq(int(value))
    else:
        selected = labels.eq(int(stage))
    return (
        batch["target_mask"]
        & targets.valid_rotation[:, :, None]
        & selected[:, :, None]
    )


@torch.no_grad()
def stage_affine_template_baseline(
    dataset_root, manifest, device="cpu", ridge=1e-4,
):
    """Fit training-only stage affine fields and evaluate validation UIDs."""
    device = torch.device(device)
    train = V41TrajectoryDataset(
        dataset_root, manifest, "train", "zero", 42,
        families=("soft_body",),
    )
    validation = V41TrajectoryDataset(
        dataset_root, manifest, "validation", "zero", 42,
        families=("soft_body",),
    )
    normal = {
        int(stage): [torch.zeros(4, 4, device=device),
                     torch.zeros(4, 3, device=device)]
        for stage in EVENT_STAGES
    }
    for raw in DataLoader(train, batch_size=1, shuffle=False):
        batch = move(raw, device)
        targets, stages = _batch_targets_and_stages(batch)
        coordinates = targets.reference_shape / targets.radius[:, None, None]
        design = torch.cat((coordinates, torch.ones_like(coordinates[..., :1])), -1)
        normalized_target = targets.displacement / targets.radius[:, None, None, None]
        for stage in EVENT_STAGES:
            valid = _event_mask(batch, targets, stages, stage)[0]
            frames, points = torch.where(valid)
            if not len(frames):
                continue
            x = design[0, points]
            y = normalized_target[0, frames, points]
            normal[int(stage)][0].add_(x.T @ x)
            normal[int(stage)][1].add_(x.T @ y)
    coefficients = {}
    identity = torch.eye(4, device=device)
    identity[-1, -1] = 0  # do not regularize the intercept
    for stage, (xtx, xty) in normal.items():
        coefficients[stage] = torch.linalg.solve(xtx + ridge * identity, xty)

    rows = []
    for raw in DataLoader(validation, batch_size=1, shuffle=False):
        batch = move(raw, device)
        targets, stages = _batch_targets_and_stages(batch)
        coordinates = targets.reference_shape / targets.radius[:, None, None]
        design = torch.cat((coordinates, torch.ones_like(coordinates[..., :1])), -1)
        predicted = torch.zeros_like(targets.displacement)
        for stage in EVENT_STAGES:
            selected = stages.labels[:, 1:].eq(int(stage))
            field = design @ coefficients[int(stage)]
            predicted = torch.where(
                selected[:, :, None, None],
                field[:, None] * targets.radius[:, None, None, None],
                predicted,
            )
        predicted = (
            predicted - masked_mean(
                predicted,
                batch["target_mask"], dim=2,
            )[:, :, None]
        ) * batch["target_mask"][..., None]
        valid = _event_mask(batch, targets, stages)
        target = targets.displacement
        pred_flat = predicted[valid]
        target_flat = target[valid]
        target_energy = target_flat.square().sum() / valid.sum().clamp_min(1)
        error_energy = (
            (pred_flat - target_flat).square().sum()
            / valid.sum().clamp_min(1)
        )
        cosine = F.cosine_similarity(pred_flat, target_flat, dim=-1)
        rows.append({
            "uid": batch["uid"][0],
            "episode_id": batch["episode_id"][0],
            "event_normalized_mse": float((error_energy / target_energy).cpu()),
            "event_spatial_cosine": float(cosine.mean().cpu()),
            "event_predicted_to_target_rms": float(
                torch.sqrt(pred_flat.square().mean() / target_flat.square().mean()).cpu()
            ),
            "stage_metrics": {
                stage.name.lower(): _template_stage_metrics(
                    predicted, target, _event_mask(batch, targets, stages, stage),
                ) for stage in EVENT_STAGES
            },
        })
    return {
        "baseline": "training_fitted_per_stage_affine_canonical_field",
        "fit_split": "train",
        "evaluation_split": "validation",
        "test_used": False,
        "ridge": ridge,
        "features": ["normalized_x", "normalized_y", "normalized_z", "intercept"],
        "stages": [stage.name.lower() for stage in EVENT_STAGES],
        "validation_uids": len(rows),
        "summary": {
            key: float(np.mean([row[key] for row in rows]))
            for key in (
                "event_normalized_mse", "event_spatial_cosine",
                "event_predicted_to_target_rms",
            )
        },
        "rows": rows,
    }


def _template_stage_metrics(predicted, target, valid):
    if not valid.any():
        return None
    pred = predicted[valid]
    truth = target[valid]
    target_energy = truth.square().sum() / valid.sum().clamp_min(1)
    error_energy = (pred - truth).square().sum() / valid.sum().clamp_min(1)
    return {
        "normalized_mse": float((error_energy / target_energy).cpu()),
        "spatial_cosine": float(F.cosine_similarity(pred, truth, dim=-1).mean().cpu()),
        "predicted_to_target_rms": float(
            torch.sqrt(pred.square().mean() / truth.square().mean()).cpu()
        ),
    }


EFFECT_METRICS = (
    "stage_weighted_canonical_nrmse",
    "stage_weighted_strain_rmse",
    "uid_balanced_magnitude_correlation",
    "uid_balanced_predicted_to_target_peak_ratio",
    "median_onset_error_frames",
    "median_peak_error_frames",
)


def summarize_controlled_matrix(root, variants, seeds):
    """Write descriptive factorial effects; never produce a pass/fail verdict."""
    root = Path(root)
    cells = {}
    for variant in variants:
        reports = []
        for seed in seeds:
            path = root / variant / f"seed{seed}" / "VALIDATION_SCREEN.json"
            if not path.exists():
                return None
            reports.append(json.loads(path.read_text())["geometry_only"])
        cells[variant] = {}
        groups = sorted(set.intersection(*(set(report) for report in reports)))
        for group in groups:
            cells[variant][group] = {}
            for metric in EFFECT_METRICS:
                values = [report[group].get(metric) for report in reports]
                usable = [value for value in values if value is not None]
                cells[variant][group][metric] = {
                    "per_seed": dict(zip(map(str, seeds), values)),
                    "mean": float(np.mean(usable)) if usable else None,
                }
    effects = {}
    required = set(ORACLE_VARIANTS)
    if required.issubset(cells):
        for group in cells["geometry_control"]:
            effects[group] = {}
            for metric in EFFECT_METRICS:
                means = {
                    variant: cells[variant][group][metric]["mean"]
                    for variant in required
                }
                if any(value is None for value in means.values()):
                    continue
                c, t = means["geometry_control"], means["oracle_temporal"]
                m, b = means["oracle_material"], means["oracle_both"]
                effects[group][metric] = {
                    "temporal_main_effect": ((t - c) + (b - m)) / 2,
                    "material_main_effect": ((m - c) + (b - t)) / 2,
                    "interaction": b - t - m + c,
                    "direction": (
                        "higher_is_better" if metric ==
                        "uid_balanced_magnitude_correlation" else
                        "closer_to_one_is_better" if metric ==
                        "uid_balanced_predicted_to_target_peak_ratio" else
                        "lower_is_better"
                    ),
                }
    report = {
        "analysis": "descriptive_2x2_controlled_group_not_a_gate",
        "decision": None,
        "variants": ORACLE_VARIANTS,
        "cells": cells,
        "factorial_effects": effects,
        "interpretation": {
            "temporal_only_helps": "timing/state ambiguity is implicated",
            "material_only_helps": "material ambiguity is implicated",
            "both_only_helps": "the factors interact or both are required",
            "neither_helps": (
                "prioritize representation, decoder optimization, targets, "
                "or signal-to-noise rather than missing oracle context"
            ),
        },
    }
    path = root / "CONTROLLED_EFFECTS.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path
