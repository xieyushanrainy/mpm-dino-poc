from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mpm_dino_v2.deformation import edge_validity, gather_neighbours
from .v41_data import MODEL_INPUT_KEYS, UIDBalancedSampler, V41TrajectoryDataset
from .v41_train import (
    atomic_torch_save, move, restore_rng_state, rng_state, seed_all,
)
from .v42_geometry import canonical_targets
from .v42_losses import compute_v42_local_losses
from .v42_model import V42RotationAwareSurrogate
from .v42_stages import (
    STAGE_RAW_WEIGHTS, ImpactStage, derive_impact_stages,
    total_mass_stage_weights,
)


LOCAL_PREFIXES = (
    "dino_projection.", "region_encoder.", "region_adapter.",
    "canonical_head.", "oracle_canonical_head.",
)
HORIZONS = (1, 8, 16, 30, 40, 59)
TIMING_FLOOR_NRMSE = 1e-4


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_parameters(model):
    return [
        parameter for name, parameter in model.named_parameters()
        if name.startswith(LOCAL_PREFIXES)
    ]


def protected_snapshot(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith(LOCAL_PREFIXES)
    }


def protected_is_identical(model, snapshot):
    state = model.state_dict()
    return all(
        torch.equal(state[name].detach().cpu(), value)
        for name, value in snapshot.items()
    )


def set_gate2_mode(model, training):
    model.eval()
    if training:
        model.dino_projection.train()
        model.region_encoder.train()
        model.region_adapter.train()
        model.canonical_head.train()


def load_gate1e_source(
    checkpoint, local_mode="geometry", device="cpu", oracle_condition_dim=0,
):
    """Strictly load the reviewed Gate-1E model and protect its global path."""
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    if config.get("experiment") != "v42_gate1e_protected_angular_dynamics":
        raise ValueError("Gate 2 requires a Gate-1E source checkpoint")
    if config.get("model_contract_version") != "gate1e_v1":
        raise ValueError("unsupported Gate-1E model contract")
    source = state["model"]
    hidden_dim = int(source["v42_com_head.1.weight"].shape[0])
    block_indices = {
        int(name.split(".")[1])
        for name in source if name.startswith("blocks.")
    }
    blocks = max(block_indices) + 1
    # gate1e_v1 fixed these values, but its historical config accidentally
    # omitted them. Width/block count are independently checked from tensors.
    heads = 4
    dropout = 0.1
    model = V42RotationAwareSurrogate(
        local_mode=local_mode,
        hidden_dim=hidden_dim, blocks=blocks,
        heads=heads, dropout=dropout, frames=59,
        local_trunk_alpha=0.0, rotation_parameterization="axis_angle",
        rotation_attention=True, rotation_dynamics=True,
        oracle_condition_dim=oracle_condition_dim,
    ).to(device)
    if oracle_condition_dim:
        incompatible = model.load_state_dict(source, strict=False)
        expected_missing = {
            name for name in model.state_dict()
            if name.startswith("oracle_canonical_head.")
        }
        if set(incompatible.missing_keys) != expected_missing or (
            incompatible.unexpected_keys
        ):
            raise RuntimeError("unexpected Gate-1E/oracle state mismatch")
    else:
        model.load_state_dict(source, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if local_mode == "geometry":
        for parameter in local_parameters(model):
            parameter.requires_grad_(True)
    state = {**state, "_gate2_reconstruction": {
        "hidden_dim": hidden_dim, "blocks": blocks, "heads": heads,
        "dropout": dropout,
        "source": (
            "tensor-inferred width/block count; gate1e_v1 fixed heads/dropout"
        ),
    }}
    return model, state


@torch.no_grad()
def verify_global_bit_identity(model, source_model, loader, device):
    model.eval()
    source_model.eval()
    checked = 0
    for raw in loader:
        batch = move(raw, device)
        inputs = {key: batch[key] for key in MODEL_INPUT_KEYS}
        candidate = model(**inputs)
        source = source_model(**inputs)
        if not torch.equal(candidate.com, source.com):
            raise RuntimeError("Gate-2 COM differs from frozen Gate-1E source")
        if not torch.equal(candidate.rotation, source.rotation):
            raise RuntimeError(
                "Gate-2 rotation differs from frozen Gate-1E source"
            )
        checked += candidate.com.shape[0]
    return checked


def _batch_targets_and_stages(batch):
    targets = canonical_targets(
        batch["x1"], batch["target"], batch["input_mask"],
        batch["target_mask"],
    )
    stages = derive_impact_stages(
        batch["x1"], batch["target"], batch["input_mask"],
        batch["target_mask"], batch["neighbour_indices"],
        batch["neighbour_mask"], batch["rest_edge_lengths"],
        batch["dt"], batch["gravity"], batch["floor_z"],
    )
    return targets, stages


def run_gate2_epoch(
    model, loader, device, optimizer=None, accumulation=4, max_batches=None,
    loss_options=None, stage_weight_mode="per_frame",
    condition_builder=None,
):
    if stage_weight_mode not in {"per_frame", "total_mass"}:
        raise ValueError(f"unknown stage weight mode: {stage_weight_mode}")
    training = optimizer is not None
    set_gate2_mode(model, training)
    parameters = local_parameters(model)
    totals, batches = {}, 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.enable_grad() if training else torch.no_grad():
        for raw in loader:
            batch = move(raw, device)
            targets, stages = _batch_targets_and_stages(batch)
            # Kabsch targets, labels and weights are detached preprocessing.
            frame_weights = (
                total_mass_stage_weights(stages.labels[:, 1:])
                if stage_weight_mode == "total_mass"
                else stages.weights[:, 1:]
            ).detach()
            inputs = {key: batch[key] for key in MODEL_INPUT_KEYS}
            if condition_builder is not None:
                inputs["oracle_condition"] = condition_builder(batch, stages)
            output = model(**inputs)
            losses = compute_v42_local_losses(
                output, batch, targets=targets,
                frame_weights=frame_weights,
                **(loss_options or {}),
            )
            if training:
                (losses.total / accumulation).backward()
                if (batches + 1) % accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            for name in losses.__dataclass_fields__:
                value = getattr(losses, name)
                totals[name] = totals.get(name, 0.0) + float(
                    value.detach().cpu()
                )
            batches += 1
            if max_batches and batches >= max_batches:
                break
    if training and batches % accumulation:
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {
        key: value / max(batches, 1) for key, value in totals.items()
    }


def _weighted_rmse(error, mask, frame_weight=None):
    squared = error.square()
    if squared.ndim == mask.ndim + 1:
        squared = squared.sum(-1)
    weight = mask.to(squared.dtype)
    if frame_weight is not None:
        while frame_weight.ndim < weight.ndim:
            frame_weight = frame_weight[..., None]
        weight = weight * frame_weight
    return torch.sqrt(
        (squared * weight).sum() / weight.sum().clamp_min(1)
    )


def _first_at_or_above(values, threshold, start):
    indices = torch.where(values[start:] >= threshold)[0]
    return int(indices[0] + start) if len(indices) else None


@torch.no_grad()
def gate2_rows(model, loader, device, condition, condition_builder=None):
    model.eval()
    rows = []
    for raw in loader:
        batch = move(raw, device)
        targets, stages = _batch_targets_and_stages(batch)
        inputs = {key: batch[key] for key in MODEL_INPUT_KEYS}
        if condition_builder is not None:
            inputs["oracle_condition"] = condition_builder(batch, stages)
        output = model(**inputs)
        mask = batch["target_mask"]
        radius = targets.radius
        predicted_magnitude = torch.sqrt(
            (
                output.canonical_displacement.square().sum(-1)
                * mask
            ).sum(2) / mask.sum(2).clamp_min(1)
        ) / radius[:, None]
        target_magnitude = torch.sqrt(
            (
                targets.displacement.square().sum(-1) * mask
            ).sum(2) / mask.sum(2).clamp_min(1)
        ) / radius[:, None]
        batch_size, frames, points, _ = output.canonical_shape.shape
        indices = batch["neighbour_indices"][:, None].expand(
            -1, frames, -1, -1
        ).reshape(batch_size * frames, points, -1)
        graph_mask = batch["neighbour_mask"][:, None].expand(
            -1, frames, -1, -1
        ).reshape(batch_size * frames, points, -1)
        flat_mask = mask.reshape(batch_size * frames, points)
        valid_edges = edge_validity(flat_mask, indices, graph_mask)
        pred = output.canonical_shape.reshape(batch_size * frames, points, 3)
        truth = (
            targets.reference_shape[:, None] + targets.displacement
        ).reshape(batch_size * frames, points, 3)
        pred_lengths = torch.linalg.vector_norm(
            gather_neighbours(pred, indices) - pred[:, :, None], dim=-1,
        )
        target_lengths = torch.linalg.vector_norm(
            gather_neighbours(truth, indices) - truth[:, :, None], dim=-1,
        )
        rest = batch["rest_edge_lengths"][:, None].expand(
            -1, frames, -1, -1
        ).reshape_as(pred_lengths).clamp_min(1e-8)
        strain_error = (
            (pred_lengths - rest) / rest
            - (target_lengths - rest) / rest
        ).reshape(batch_size, frames, points, -1)
        edge_mask = valid_edges.reshape(batch_size, frames, points, -1)
        for sample in range(batch_size):
            valid_rotation = targets.valid_rotation[sample]
            canonical_mask = mask[sample] & valid_rotation[:, None]
            weights = stages.weights[sample, 1:]
            stage_values = stages.labels[sample, 1:]
            record = {
                "condition": condition,
                "uid": batch["uid"][sample],
                "episode_id": batch["episode_id"][sample],
                "family": batch["family"][sample],
                "panel": batch["panel"][sample],
                "stage_weighted_canonical_nrmse": float(
                    _weighted_rmse(
                        (
                            output.canonical_displacement[sample]
                            - targets.displacement[sample]
                        ) / radius[sample],
                        canonical_mask, weights,
                    ).cpu()
                ),
                "stage_weighted_strain_rmse": float(
                    _weighted_rmse(
                        strain_error[sample], edge_mask[sample], weights,
                    ).cpu()
                ),
                "rigid_local_rms_fraction": (
                    float(_weighted_rmse(
                        output.canonical_displacement[sample],
                        mask[sample],
                    ).cpu() / radius[sample].cpu())
                    if batch["family"][sample] == "rigid" else None
                ),
                "target_magnitude": target_magnitude[sample].cpu().tolist(),
                "predicted_magnitude": (
                    predicted_magnitude[sample].cpu().tolist()
                ),
                "stage_metrics": {},
                "timing": None,
            }
            for stage in (
                ImpactStage.COMPRESSION, ImpactStage.PEAK_DEFORMATION,
            ):
                selected = stage_values.eq(int(stage))
                if selected.any():
                    record["stage_metrics"][stage.name.lower()] = {
                        "canonical_nrmse": float(_weighted_rmse(
                            (
                                output.canonical_displacement[sample, selected]
                                - targets.displacement[sample, selected]
                            ) / radius[sample],
                            canonical_mask[selected],
                        ).cpu()),
                        "strain_rmse": float(_weighted_rmse(
                            strain_error[sample, selected],
                            edge_mask[sample, selected],
                        ).cpu()),
                    }
            onset = int(stages.contact_onset[sample]) - 1
            target_curve = target_magnitude[sample]
            peak = float(target_curve.max())
            identifiable = onset >= 0 and peak >= TIMING_FLOOR_NRMSE
            if identifiable:
                target_onset = _first_at_or_above(
                    target_curve, 0.2 * peak, max(onset, 0),
                )
                predicted_onset = _first_at_or_above(
                    predicted_magnitude[sample], 0.2 * peak, max(onset, 0),
                )
                target_peak = int(torch.argmax(target_curve))
                predicted_peak = int(torch.argmax(predicted_magnitude[sample]))
                record["timing"] = {
                    "identifiable": True,
                    "target_onset": target_onset,
                    "predicted_onset": predicted_onset,
                    "onset_error_frames": (
                        abs(predicted_onset - target_onset)
                        if target_onset is not None
                        and predicted_onset is not None else None
                    ),
                    "target_peak": target_peak,
                    "predicted_peak": predicted_peak,
                    "peak_error_frames": abs(predicted_peak - target_peak),
                }
            rows.append(record)
    return rows


def _uid_balanced_mean(rows, key):
    by_uid = defaultdict(list)
    for row in rows:
        if row[key] is not None:
            by_uid[row["uid"]].append(row[key])
    return float(np.mean([
        np.mean(values) for values in by_uid.values()
    ])) if by_uid else None


def summarize_gate2(rows):
    summary = {}
    for panel in ("Z", "V"):
        for family in ("rigid", "soft_body"):
            selected = [
                row for row in rows
                if row["panel"] == panel and row["family"] == family
            ]
            if not selected:
                continue
            item = {
                "uids": len({row["uid"] for row in selected}),
                "episodes": len(selected),
                "stage_weighted_canonical_nrmse": _uid_balanced_mean(
                    selected, "stage_weighted_canonical_nrmse",
                ),
                "stage_weighted_strain_rmse": _uid_balanced_mean(
                    selected, "stage_weighted_strain_rmse",
                ),
                "rigid_local_rms_fraction": _uid_balanced_mean(
                    selected, "rigid_local_rms_fraction",
                ),
            }
            for stage in ("compression", "peak_deformation"):
                for metric in ("canonical_nrmse", "strain_rmse"):
                    values = [
                        {
                            **row,
                            "_value": row["stage_metrics"].get(
                                stage, {},
                            ).get(metric),
                        }
                        for row in selected
                    ]
                    item[f"{stage}_{metric}"] = _uid_balanced_mean(
                        values, "_value",
                    )
            correlations = []
            timing_onset, timing_peak = [], []
            by_uid = defaultdict(list)
            for row in selected:
                target_peak = max(row["target_magnitude"])
                row["predicted_to_target_peak_ratio"] = (
                    max(row["predicted_magnitude"]) / target_peak
                    if target_peak > 0 else None
                )
                by_uid[row["uid"]].append(row)
                timing = row["timing"]
                if timing and timing["identifiable"]:
                    if timing["onset_error_frames"] is not None:
                        timing_onset.append(timing["onset_error_frames"])
                    timing_peak.append(timing["peak_error_frames"])
            for uid_rows in by_uid.values():
                target = np.concatenate([
                    np.asarray(row["target_magnitude"]) for row in uid_rows
                ])
                predicted = np.concatenate([
                    np.asarray(row["predicted_magnitude"]) for row in uid_rows
                ])
                if target.std() > 0 and predicted.std() > 0:
                    correlations.append(float(np.corrcoef(target, predicted)[0, 1]))
            item["uid_balanced_magnitude_correlation"] = (
                float(np.mean(correlations)) if correlations else None
            )
            item["uid_balanced_predicted_to_target_peak_ratio"] = (
                _uid_balanced_mean(
                    selected, "predicted_to_target_peak_ratio",
                )
            )
            item["identifiable_timing_episodes"] = len(timing_peak)
            item["median_onset_error_frames"] = (
                float(np.median(timing_onset)) if timing_onset else None
            )
            item["median_peak_error_frames"] = (
                float(np.median(timing_peak)) if timing_peak else None
            )
            summary[f"panel_{panel}/{family}"] = item
    return summary


def gate2_screen(learned, baseline):
    """Apply the frozen validation screen without weakening any criterion."""
    improvements, stage_improvement = {}, {}
    for group, learned_values in learned.items():
        if group not in baseline:
            continue
        base_values = baseline[group]
        for metric in (
            "stage_weighted_canonical_nrmse",
            "stage_weighted_strain_rmse",
        ):
            base = base_values[metric]
            current = learned_values[metric]
            improvements[f"{group}/{metric}"] = (
                (base - current) / base if base and current is not None else None
            )
        stage_improvement[group] = any(
            base_values.get(f"{stage}_{metric}") is not None
            and learned_values.get(f"{stage}_{metric}") is not None
            and learned_values[f"{stage}_{metric}"]
            < base_values[f"{stage}_{metric}"]
            for stage in ("compression", "peak_deformation")
            for metric in ("canonical_nrmse", "strain_rmse")
        )
    local_groups = [
        group for group in learned if group.endswith("/soft_body")
    ]
    primary = all(
        improvements.get(f"{group}/{metric}", -float("inf")) >= 0.10
        for group in local_groups
        for metric in (
            "stage_weighted_canonical_nrmse",
            "stage_weighted_strain_rmse",
        )
    )
    correlation = all(
        learned[group]["uid_balanced_magnitude_correlation"] is not None
        and learned[group]["uid_balanced_magnitude_correlation"] >= 0.5
        for group in local_groups
    )
    timing = all(
        (
            learned[group]["identifiable_timing_episodes"] == 0
            or (
                learned[group]["median_onset_error_frames"] is not None
                and learned[group]["median_onset_error_frames"] <= 2
                and learned[group]["median_peak_error_frames"] <= 2
            )
        )
        for group in local_groups
    )
    rigid = all(
        values["rigid_local_rms_fraction"] is None
        or values["rigid_local_rms_fraction"] < 0.001
        for group, values in learned.items() if group.endswith("/rigid")
    )
    return {
        "passed": (
            primary and correlation and timing and rigid
            and all(stage_improvement.get(group, False) for group in local_groups)
        ),
        "improvements": improvements,
        "soft_compression_or_peak_improved": {
            group: stage_improvement.get(group, False)
            for group in local_groups
        },
        "magnitude_correlation_passed": correlation,
        "timing_passed": timing,
        "rigid_local_rms_passed": rigid,
        "global_bit_identity_required_separately": True,
    }


def train_v42_gate2(
    root, manifest, gate1e_checkpoint, output, seed, device="cuda",
    epochs=120, draws_per_epoch=40, lr=2e-4, accumulation=4,
    patience=20, plateau_patience=5, resume=True, max_batches=None,
    loss_options=None, experiment_name=None, model_contract_version=None,
    stage_weight_mode="per_frame",
    oracle_condition_dim=0, condition_builder=None,
    condition_name=None, exploratory_control=False,
):
    seed_all(seed)
    device = torch.device(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train_ds = V41TrajectoryDataset(
        root, manifest, "train", "zero", seed,
        families=("soft_body", "rigid"),
    )
    validation_ds = V41TrajectoryDataset(
        root, manifest, "validation", "zero", seed,
        families=("soft_body", "rigid"),
    )
    sampler = UIDBalancedSampler(train_ds, draws_per_epoch, seed)
    train_loader = DataLoader(
        train_ds, batch_size=1, sampler=sampler, num_workers=0,
    )
    validation_loader = DataLoader(
        validation_ds, batch_size=1, shuffle=False, num_workers=0,
    )
    model, source_state = load_gate1e_source(
        gate1e_checkpoint, "geometry", device, oracle_condition_dim,
    )
    zero_model, _ = load_gate1e_source(
        gate1e_checkpoint, "zero", device,
    )
    protected = protected_snapshot(model)
    checked_before = verify_global_bit_identity(
        model, zero_model, validation_loader, device,
    )
    zero_rows = gate2_rows(
        zero_model, validation_loader, device, "zero_local",
    )
    zero_summary = summarize_gate2(zero_rows)
    baseline_path = output / "ZERO_LOCAL_VALIDATION.json"
    baseline_path.write_text(json.dumps({
        "condition": "frozen_zero_local_output",
        "summary": zero_summary, "rows": zero_rows,
    }, indent=2) + "\n")
    parameters = local_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=plateau_patience,
        threshold=0.005, threshold_mode="rel", min_lr=1e-6,
    )
    source_path = Path(gate1e_checkpoint)
    loss_options = dict(loss_options or {})
    gate2b = bool(loss_options)
    gate2c = stage_weight_mode == "total_mass"
    if stage_weight_mode not in {"per_frame", "total_mass"}:
        raise ValueError(f"unknown stage weight mode: {stage_weight_mode}")
    if gate2b:
        required = {
            "soft_deformation_amplification_cap",
            "soft_deformation_quantile",
            "soft_deformation_floor_fraction",
            "family_balanced", "rigid_family_weight", "rigid_zero_weight",
        }
        if set(loss_options) != required:
            raise ValueError(
                "Gate-2B loss options must exactly match the reviewed contract"
            )
    config = {
        "experiment": experiment_name or (
            "v42_gate2c_total_mass_stage_balanced" if gate2c
            else "v42_gate2b_family_balanced_deformation_scaled"
            if gate2b else "v42_gate2_geometry_only_canonical_local"
        ),
        "model_contract_version": model_contract_version or (
            "gate2c_geometry_only_v1" if gate2c
            else "gate2b_geometry_only_v1" if gate2b
            else "gate2_geometry_only_v1"
        ),
        "seed": seed, "device": str(device), "epochs": epochs,
        "draws_per_epoch": draws_per_epoch, "lr": lr,
        "accumulation": accumulation, "patience": patience,
        "plateau_patience": plateau_patience,
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "gate1e_checkpoint": str(source_path),
        "gate1e_checkpoint_sha256": file_sha256(source_path),
        "gate1e_source_epoch": source_state["epoch"],
        "gate1e_model_reconstruction": source_state["_gate2_reconstruction"],
        "gate1e_status": "failed_rotation_screen_operational_placeholder",
        "dino_mode": "zero_with_real_validity_mask",
        "real_dino_trained": False,
        "local_trunk_alpha": 0.0,
        "trainable_prefixes": list(LOCAL_PREFIXES),
        "protected": [
            "physical_trunk", "com_head", "entire_rotation_branch",
        ],
        "world_reconstruction_loss": False,
        "penetration_loss": False,
        "stage_metadata_is_model_input": False,
        "detached_preprocessing": [
            "kabsch_targets", "stage_labels", "stage_weights",
        ],
        "loss_weights": {
            "canonical": 1.0, "strain": 0.5, "edge_length": 0.25,
            "local_velocity": 0.25,
            "rigid_zero": loss_options.get("rigid_zero_weight", 0.25),
        },
        "timing_identifiability": {
            "target_peak_nrmse_floor": TIMING_FLOOR_NRMSE,
            "onset": "first post-contact frame >= 20% target peak",
            "peak": "first maximum",
            "test_data_used_to_set_rule": False,
        },
        "selection": {
            "split": "validation",
            "metric": (
                "stage-weighted canonical NRMSE + stage-weighted strain RMSE"
            ),
            "test_used": False,
        },
        "validation_global_bit_identity_checked_episodes": checked_before,
        "oracle_condition_dim": oracle_condition_dim,
        "condition_name": condition_name,
        "analysis_mode": (
            "exploratory_controlled_group_not_a_gate"
            if exploratory_control else "gate"
        ),
    }
    if gate2b:
        config["gate2b_loss_design"] = {
            **loss_options,
            "soft_scale": (
                f"max(q{100 * loss_options['soft_deformation_quantile']:g} "
                "target canonical RMS over frames, "
                f"{loss_options['soft_deformation_floor_fraction']:g} * "
                "reference radius)"
            ),
            "soft_amplification": (
                "min(reference radius / detached soft scale, variant cap)"
            ),
            "amplified_terms": ["canonical", "local_velocity"],
            "unchanged_terms": ["strain", "edge_length"],
            "batch_size": 1,
            "sampler": "strict alternating family UID-balanced draws",
            "evaluation_screen": "unchanged_gate2_frozen_screen",
        }
    if gate2c:
        config["gate2c_stage_weight_design"] = {
            "training_mode": "total_mass",
            "formula": "w_t proportional to alpha_stage / frames_in_stage",
            "normalization": "per-episode mean frame weight equals one",
            "stage_importance": {
                stage.name.lower(): weight
                for stage, weight in STAGE_RAW_WEIGHTS.items()
            },
            "weight_cap": None,
            "target_frames_only": True,
            "labels_and_weights_detached": True,
            "evaluation_screen": "unchanged_gate2_frozen_screen",
            "parent_loss_contract": "gate2b_balanced_x1",
        }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    best, stale, start_epoch = float("inf"), 0, 1
    last_path = output / "last.pt"
    if resume and last_path.exists() and not (
        output / "RUN_COMPLETE.json"
    ).exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state["config"] != config:
            raise ValueError("refusing to resume changed Gate-2 configuration")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best, stale = float(state["best"]), int(state["stale"])
        start_epoch = int(state["epoch"]) + 1
        sampler.epoch = int(state["sampler_epoch"])
        restore_rng_state(state["rng_state"])
    mode = "a" if start_epoch > 1 else "w"
    epoch = start_epoch - 1
    with (output / "history.jsonl").open(mode) as history:
        for epoch in range(start_epoch, epochs + 1):
            started = time.perf_counter()
            train = run_gate2_epoch(
                model, train_loader, device, optimizer,
                accumulation, max_batches,
                loss_options,
                stage_weight_mode,
                condition_builder,
            )
            validation = run_gate2_epoch(
                model, validation_loader, device, None,
                accumulation, max_batches,
                loss_options,
                stage_weight_mode,
                condition_builder,
            )
            rows = gate2_rows(
                model, validation_loader, device,
                condition_name or (
                    "geometry_only_gate2c" if gate2c
                    else "geometry_only_gate2b" if gate2b else "geometry_only"
                ), condition_builder,
            )
            summary = summarize_gate2(rows)
            soft = summary["panel_Z/soft_body"]
            selection = (
                soft["stage_weighted_canonical_nrmse"]
                + soft["stage_weighted_strain_rmse"]
            )
            scheduler.step(selection)
            improved = selection < best
            best = min(best, selection)
            stale = 0 if improved else stale + 1
            if not protected_is_identical(model, protected):
                raise RuntimeError("protected Gate-1E parameters changed")
            record = {
                "epoch": epoch, "train": train, "validation": validation,
                "validation_summary": summary, "selection": selection,
                "screen": gate2_screen(summary, zero_summary),
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - started,
            }
            history.write(json.dumps(record) + "\n")
            history.flush()
            state = {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": config, "epoch": epoch, "train": train,
                "validation": validation, "validation_summary": summary,
                "selection": selection, "best": best, "stale": stale,
                "sampler_epoch": sampler.epoch, "rng_state": rng_state(),
            }
            atomic_torch_save(state, last_path)
            if improved:
                atomic_torch_save(state, output / "best.pt")
            print(
                f"epoch={epoch:03d} local={selection:.6g} stale={stale}",
                flush=True,
            )
            if stale >= patience:
                break
    best_path = output / "best.pt"
    best_state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model"])
    learned_rows = gate2_rows(
        model, validation_loader, device,
        condition_name or (
            "geometry_only_gate2c" if gate2c
            else "geometry_only_gate2b" if gate2b else "geometry_only"
        ), condition_builder,
    )
    learned_summary = summarize_gate2(learned_rows)
    checked_after = verify_global_bit_identity(
        model, zero_model, validation_loader, device,
    )
    report = {
        "split": "validation", "panel_reporting_separate": True,
        "zero_local": zero_summary, "geometry_only": learned_summary,
        "screen": gate2_screen(learned_summary, zero_summary),
        "screen_interpretation": (
            "descriptive_legacy_metrics_only_not_a_gate"
            if exploratory_control else "frozen_gate_decision"
        ),
        "geometry_rows": learned_rows,
    }
    report_path = output / "VALIDATION_SCREEN.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    completion = {
        "status": "complete", "last_epoch": epoch,
        "best_epoch": best_state["epoch"],
        "best_selection": best_state["selection"],
        "best_checkpoint_sha256": file_sha256(best_path),
        "last_checkpoint_sha256": file_sha256(last_path),
        "history_sha256": file_sha256(output / "history.jsonl"),
        "zero_local_validation_sha256": file_sha256(baseline_path),
        "validation_screen_sha256": file_sha256(report_path),
        "protected_parameters_bit_identical": protected_is_identical(
            model, protected,
        ),
        "validation_com_rotation_bit_identical_to_gate1e": True,
        "validation_global_identity_checked_episodes_before": checked_before,
        "validation_global_identity_checked_episodes_after": checked_after,
        "gate2_screen_passed_this_seed": (
            None if exploratory_control else report["screen"]["passed"]
        ),
    }
    (output / "RUN_COMPLETE.json").write_text(
        json.dumps(completion, indent=2) + "\n"
    )
    return best_path
