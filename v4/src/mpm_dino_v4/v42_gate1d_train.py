from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from .v41_data import MODEL_INPUT_KEYS, UIDBalancedSampler, V41TrajectoryDataset
from .v41_train import (
    atomic_torch_save, move, restore_rng_state, rng_state, seed_all,
    state_sha256,
)
from .v42_geometry import (
    canonical_targets, rotation_geodesic, rotation_matrix_to_vector,
)
from .v42_losses import compute_v42_global_losses
from .v42_model import V42RotationAwareSurrogate
from .v42_stages import ImpactStage, derive_impact_stages
from .v42_train import write_gate1_validation_baselines


ROTATION_PREFIXES = (
    "rotation_contact_projection.", "rotation_attention.",
    "rotation_adapter.", "rotation_head.", "rotation_contact_head.",
    "rotation_contact_score.",
)
PROTECTED_PREFIXES = (
    "initial_node.", "initial_graph.", "time_projection.", "token.",
    "blocks.", "v42_com_head.",
)
HORIZONS = (1, 8, 16, 30, 40, 59)
POST_CONTACT_HORIZONS = (8, 16, 30, 40, 59)
ACTIVE_ROTATION_RAD = 0.05
INACTIVE_ROTATION_RAD = 0.01


def rotation_parameters(model):
    return [
        parameter for name, parameter in model.named_parameters()
        if name.startswith(ROTATION_PREFIXES)
    ]


def set_rotation_training_mode(model, training):
    # The protected physical path remains in deterministic evaluation mode.
    model.eval()
    if training:
        model.rotation_contact_projection.train()
        model.rotation_attention.train()
        model.rotation_adapter.train()
        model.rotation_head.train()
        if model.contact_rotation_mode is not None:
            model.rotation_contact_head.train()
            model.rotation_contact_score.train()


def protected_snapshot(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith(PROTECTED_PREFIXES)
    }


def protected_is_identical(model, snapshot):
    current = model.state_dict()
    return all(torch.equal(current[name].detach().cpu(), value)
               for name, value in snapshot.items())


def load_gate1b_source(model, checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = state["model"]
    target = model.state_dict()
    compatible = {
        name: value for name, value in source.items()
        if name in target
        and target[name].shape == value.shape
        and not name.startswith(ROTATION_PREFIXES)
    }
    missing_protected = [
        name for name in target
        if name.startswith(PROTECTED_PREFIXES) and name not in compatible
    ]
    if missing_protected:
        raise ValueError(
            f"Gate-1B checkpoint lacks protected tensors: {missing_protected}"
        )
    model.load_state_dict(compatible, strict=False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in rotation_parameters(model):
        parameter.requires_grad_(True)
    return state


@torch.no_grad()
def verify_com_identity(model, source_state, loader, device):
    config = source_state["config"]
    source_model = V42RotationAwareSurrogate(
        local_mode="zero", hidden_dim=config["hidden_dim"],
        blocks=config["blocks"], heads=config["heads"],
        dropout=config["dropout"], frames=59,
        rotation_parameterization="6d", rotation_attention=False,
    ).to(device)
    source_model.load_state_dict(source_state["model"])
    source_model.eval()
    model.eval()
    checked = 0
    for raw in loader:
        batch = move(raw, device)
        inputs = {key: batch[key] for key in MODEL_INPUT_KEYS}
        source_com = source_model(**inputs).com
        protected_com = model(**inputs).com
        if not torch.equal(source_com, protected_com):
            raise RuntimeError(
                "Gate-1D COM is not bit-identical to its Gate-1B source"
            )
        checked += source_com.shape[0]
    del source_model
    return checked


def _rotation_metrics(output, batch, targets):
    metrics = {}
    for sample in range(output.rotation.shape[0]):
        group = f"{batch['family'][sample]}/panel_{batch['panel'][sample]}"
        values = metrics.setdefault(group, {})
        for horizon in HORIZONS:
            index = horizon - 1
            if horizon > output.rotation.shape[1]:
                continue
            valid = (
                bool(batch["target_mask"][sample, index].any())
                and bool(targets.valid_rotation[sample, index])
            )
            if not valid:
                continue
            predicted = rotation_geodesic(
                output.rotation[sample, index],
                targets.rotation[sample, index],
            )
            identity = rotation_geodesic(
                torch.eye(
                    3, device=output.rotation.device,
                    dtype=output.rotation.dtype,
                ),
                targets.rotation[sample, index],
            )
            item = values.setdefault(
                f"h{horizon}",
                {"model_sum": 0.0, "identity_sum": 0.0, "count": 0},
            )
            item["model_sum"] += float(predicted.detach().cpu())
            item["identity_sum"] += float(identity.detach().cpu())
            item["count"] += 1
    return metrics


def _merge_metrics(total, update):
    for group, horizons in update.items():
        destination = total.setdefault(group, {})
        for horizon, values in horizons.items():
            item = destination.setdefault(
                horizon, {"model_sum": 0.0, "identity_sum": 0.0, "count": 0},
            )
            for key in item:
                item[key] += values[key]


def _finalize_metrics(metrics):
    result = {}
    for group, horizons in sorted(metrics.items()):
        result[group] = {}
        for horizon, values in sorted(horizons.items()):
            count = values["count"]
            result[group][horizon] = {
                "model_rotation_rad": values["model_sum"] / count,
                "identity_rotation_rad": values["identity_sum"] / count,
                "count": count,
            }
    return result


def _rotation_activity_metrics(output, batch, targets):
    result = {}
    identity = torch.eye(
        3, device=output.rotation.device, dtype=output.rotation.dtype,
    )
    for sample in range(output.rotation.shape[0]):
        group = f"{batch['family'][sample]}/panel_{batch['panel'][sample]}"
        item = result.setdefault(group, {
            "active_model_sum": 0.0, "active_identity_sum": 0.0,
            "active_count": 0, "inactive_prediction_sum": 0.0,
            "inactive_count": 0,
        })
        valid = (
            batch["target_mask"][sample].any(1)
            & targets.valid_rotation[sample]
        )
        target_magnitude = rotation_geodesic(
            identity, targets.rotation[sample]
        )
        model_error = rotation_geodesic(
            output.rotation[sample], targets.rotation[sample]
        )
        prediction_magnitude = rotation_geodesic(
            identity, output.rotation[sample]
        )
        active = valid & target_magnitude.ge(ACTIVE_ROTATION_RAD)
        inactive = valid & target_magnitude.le(INACTIVE_ROTATION_RAD)
        item["active_model_sum"] += float(
            model_error[active].sum().detach().cpu()
        )
        item["active_identity_sum"] += float(
            target_magnitude[active].sum().cpu()
        )
        item["active_count"] += int(active.sum())
        item["inactive_prediction_sum"] += float(
            prediction_magnitude[inactive].sum().detach().cpu()
        )
        item["inactive_count"] += int(inactive.sum())
    return result


def _merge_activity(total, update):
    for group, values in update.items():
        item = total.setdefault(group, {key: 0 for key in values})
        for key, value in values.items():
            item[key] += value


def _finalize_activity(metrics):
    result = {}
    for group, values in sorted(metrics.items()):
        active_count = values["active_count"]
        inactive_count = values["inactive_count"]
        result[group] = {
            "active_model_rad": (
                values["active_model_sum"] / max(active_count, 1)
            ),
            "active_identity_rad": (
                values["active_identity_sum"] / max(active_count, 1)
            ),
            "active_count": active_count,
            "inactive_prediction_rad": (
                values["inactive_prediction_sum"] / max(inactive_count, 1)
            ),
            "inactive_count": inactive_count,
        }
    return result


def angular_dynamics_losses(output, batch, targets, frame_weights):
    if output.angular_velocity is None:
        zero = output.rotation.new_zeros(())
        return zero, zero
    identity = torch.eye(
        3, device=output.rotation.device, dtype=output.rotation.dtype,
    ).expand(output.rotation.shape[0], 1, 3, 3)
    previous = torch.cat((identity, targets.rotation[:, :-1]), dim=1)
    relative = previous.transpose(-1, -2) @ targets.rotation
    target_velocity = rotation_matrix_to_vector(relative) / (
        batch["dt"][:, None, None].clamp_min(1e-8)
    )
    valid = targets.valid_rotation.clone()
    valid[:, 1:] = valid[:, 1:] & targets.valid_rotation[:, :-1]
    valid[:, 0] = False
    family_weight = torch.tensor(
        [1.0 if family == "rigid" else 0.25 for family in batch["family"]],
        device=output.rotation.device, dtype=output.rotation.dtype,
    )[:, None]
    weight = (
        valid.to(output.rotation.dtype) * frame_weights * family_weight
    )
    velocity_error = (
        output.angular_velocity - target_velocity
    ) * batch["dt"][:, None, None]
    velocity_raw = F.smooth_l1_loss(
        velocity_error, torch.zeros_like(velocity_error),
        reduction="none", beta=0.01,
    ).mean(-1)
    velocity_loss = (
        velocity_raw * weight
    ).sum() / weight.sum().clamp_min(1)
    acceleration_valid = valid[:, 1:] & valid[:, :-1]
    acceleration_weight = (
        frame_weights[:, 1:] * family_weight
        * acceleration_valid.to(output.rotation.dtype)
    )
    predicted_change = (
        output.angular_velocity[:, 1:]
        - output.angular_velocity[:, :-1]
    ) * batch["dt"][:, None, None]
    target_change = (
        target_velocity[:, 1:] - target_velocity[:, :-1]
    ) * batch["dt"][:, None, None]
    acceleration_raw = F.smooth_l1_loss(
        predicted_change - target_change,
        torch.zeros_like(predicted_change), reduction="none", beta=0.01,
    ).mean(-1)
    acceleration_loss = (
        acceleration_raw * acceleration_weight
    ).sum() / acceleration_weight.sum().clamp_min(1)
    return velocity_loss, acceleration_loss


def gate1f_auxiliary_losses(output, batch, targets, stages):
    """Training-only contact supervision for the inference-time contact gate."""
    if output.contact_probability is None or output.contact_point is None:
        zero = output.rotation.new_zeros(())
        return zero, zero
    labels = stages.labels[:, 1:]
    target_contact = labels.eq(int(ImpactStage.CONTACT_ONSET))
    known_contact = stages.contact_onset.ge(0)[:, None]
    valid_frames = batch["target_mask"].any(2) & known_contact
    positive = (target_contact & valid_frames).sum(1, keepdim=True).clamp_min(1)
    negative = ((~target_contact) & valid_frames).sum(
        1, keepdim=True
    ).clamp_min(1)
    balanced = torch.where(
        target_contact, 0.5 / positive, 0.5 / negative
    ) * valid_frames
    contact = F.binary_cross_entropy(
        output.contact_probability.clamp(1e-6, 1 - 1e-6),
        target_contact.to(output.rotation.dtype), reduction="none",
    )
    contact = (contact * balanced).sum() / balanced.sum().clamp_min(1e-8)

    # Persistent point identities let the lowest target points identify the
    # contacting reference region without passing future positions to forward.
    count = min(4, batch["target"].shape[2])
    gaps = batch["target"][..., 2].masked_fill(
        ~batch["target_mask"], torch.inf
    )
    indices = gaps.topk(count, dim=2, largest=False).indices
    reference = targets.reference_shape[:, None].expand(
        -1, batch["target"].shape[1], -1, -1
    )
    contact_reference = torch.gather(
        reference, 2, indices[..., None].expand(-1, -1, -1, 3)
    ).mean(2)
    point_mask = target_contact & valid_frames
    point = F.smooth_l1_loss(
        (output.contact_point - contact_reference)
        / targets.radius[:, None, None],
        torch.zeros_like(output.contact_point), reduction="none", beta=0.01,
    ).mean(-1)
    point = (
        point * point_mask
    ).sum() / point_mask.sum().clamp_min(1)
    return contact, point


def run_rotation_epoch(
    model, loader, device, optimizer=None, accumulation=4,
    max_batches=None, angular_dynamics=False, contact_rotation_mode=None,
):
    training = optimizer is not None
    set_rotation_training_mode(model, training)
    parameters = rotation_parameters(model)
    if training:
        optimizer.zero_grad(set_to_none=True)
    totals, batches, metrics, activity = {}, 0, {}, {}
    with torch.enable_grad() if training else torch.no_grad():
        for raw in loader:
            batch = move(raw, device)
            output = model(**{
                key: batch[key] for key in MODEL_INPUT_KEYS
            })
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
            losses = compute_v42_global_losses(
                output, batch, targets, stages.weights[:, 1:].detach(),
            )
            velocity_loss, acceleration_loss = angular_dynamics_losses(
                output, batch, targets, stages.weights[:, 1:].detach(),
            )
            contact_loss, contact_point_loss = gate1f_auxiliary_losses(
                output, batch, targets, stages
            )
            objective = (
                losses.rotation + 0.25 * losses.rotation_key
                + 0.50 * losses.rotation_event
                + 0.25 * losses.rigid_fit
                + (0.50 * velocity_loss + 0.25 * acceleration_loss
                   if angular_dynamics
                   or contact_rotation_mode == "impulse" else 0)
                + (0.25 * contact_loss + 0.10 * contact_point_loss
                   if contact_rotation_mode is not None else 0)
            )
            if training:
                (objective / accumulation).backward()
                if (batches + 1) % accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            values = {
                "rotation_objective": objective,
                "rotation": losses.rotation,
                "rotation_key": losses.rotation_key,
                "rotation_event": losses.rotation_event,
                "rigid_fit": losses.rigid_fit,
                "angular_velocity": velocity_loss,
                "angular_acceleration": acceleration_loss,
                "contact": contact_loss,
                "contact_point": contact_point_loss,
            }
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(
                    value.detach().cpu()
                )
            _merge_metrics(metrics, _rotation_metrics(output, batch, targets))
            _merge_activity(
                activity, _rotation_activity_metrics(output, batch, targets)
            )
            batches += 1
            if max_batches and batches >= max_batches:
                break
    if training and batches % accumulation:
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {
        **{key: value / max(batches, 1) for key, value in totals.items()},
        "strata": _finalize_metrics(metrics),
        "activity": _finalize_activity(activity),
    }


def identity_screen(validation):
    strata = validation["strata"]
    required = (
        "rigid/panel_Z", "rigid/panel_V", "soft_body/panel_Z",
    )
    if any(group not in strata for group in required):
        return False, {"reason": "missing required validation stratum"}
    model_all, identity_all, details = [], [], {}
    passed = True
    for group in required:
        model = sum(
            strata[group][f"h{horizon}"]["model_rotation_rad"]
            for horizon in POST_CONTACT_HORIZONS
        ) / len(POST_CONTACT_HORIZONS)
        identity = sum(
            strata[group][f"h{horizon}"]["identity_rotation_rad"]
            for horizon in POST_CONTACT_HORIZONS
        ) / len(POST_CONTACT_HORIZONS)
        h59_model = strata[group]["h59"]["model_rotation_rad"]
        h59_identity = strata[group]["h59"]["identity_rotation_rad"]
        non_regression = model <= identity
        h59_guard = h59_model <= 1.10 * h59_identity
        passed = passed and non_regression and h59_guard
        model_all.append(model)
        identity_all.append(identity)
        details[group] = {
            "postcontact_model_rad": model,
            "postcontact_identity_rad": identity,
            "improvement_fraction": (
                (identity - model) / max(identity, 1e-12)
            ),
            "non_regression": non_regression,
            "h59_guard": h59_guard,
        }
    overall_improvement = (
        (sum(identity_all) - sum(model_all))
        / max(sum(identity_all), 1e-12)
    )
    passed = passed and overall_improvement >= 0.01
    return passed, {
        "overall_improvement_fraction": overall_improvement,
        "strata": details,
    }


def gate1f_screen(validation):
    """Active-frame learnability plus inactive/H59 safety for rigid panels."""
    required = ("rigid/panel_Z", "rigid/panel_V")
    activity, strata = validation["activity"], validation["strata"]
    if any(group not in activity or group not in strata for group in required):
        return False, {"reason": "missing required rigid validation stratum"}
    details, model_sum, identity_sum, active_count = {}, 0.0, 0.0, 0
    passed = True
    for group in required:
        item = activity[group]
        has_active = item["active_count"] > 0
        active_non_regression = (
            has_active
            and item["active_model_rad"] <= item["active_identity_rad"]
        )
        inactive_safe = (
            item["inactive_count"] == 0
            or item["inactive_prediction_rad"] <= INACTIVE_ROTATION_RAD
        )
        h59_safe = (
            strata[group]["h59"]["model_rotation_rad"]
            <= 1.10 * strata[group]["h59"]["identity_rotation_rad"]
        )
        passed = passed and active_non_regression and inactive_safe and h59_safe
        count = item["active_count"]
        model_sum += item["active_model_rad"] * count
        identity_sum += item["active_identity_rad"] * count
        active_count += count
        details[group] = {
            **item, "active_non_regression": active_non_regression,
            "inactive_safe": inactive_safe, "h59_safe": h59_safe,
        }
    improvement = (
        (identity_sum - model_sum) / max(identity_sum, 1e-12)
    )
    passed = passed and active_count > 0 and improvement >= 0.01
    return passed, {
        "active_improvement_fraction": improvement,
        "active_count": active_count,
        "active_model_rad": model_sum / max(active_count, 1),
        "strata": details,
    }


def train_v42_gate1d(
    root, manifest, gate1b_checkpoint, output, seed, device="cuda",
    epochs=120, draws_per_epoch=40, lr=2e-4, accumulation=4,
    patience=20, plateau_patience=5, min_eligible_epoch=60,
    resume=True, max_batches=None, angular_dynamics=False,
    contact_rotation_mode=None, angular_damping=0.95,
):
    if contact_rotation_mode not in {None, "absolute", "impulse"}:
        raise ValueError(contact_rotation_mode)
    if angular_dynamics and contact_rotation_mode is not None:
        raise ValueError("legacy and contact rotation dynamics are exclusive")
    seed_all(seed)
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
    source_checkpoint = Path(gate1b_checkpoint)
    source_preview = torch.load(
        source_checkpoint, map_location="cpu", weights_only=False,
    )
    source_config = source_preview["config"]
    model = V42RotationAwareSurrogate(
        local_mode="zero", hidden_dim=source_config["hidden_dim"],
        blocks=source_config["blocks"], heads=source_config["heads"],
        dropout=source_config["dropout"], frames=59,
        rotation_parameterization="axis_angle", rotation_attention=True,
        rotation_dynamics=angular_dynamics,
        contact_rotation_mode=contact_rotation_mode,
        angular_damping=angular_damping,
    ).to(device)
    source_state = load_gate1b_source(model, source_checkpoint)
    protected = protected_snapshot(model)
    checked_episodes = verify_com_identity(
        model, source_state, validation_loader, torch.device(device),
    )
    del source_preview
    parameters = rotation_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=plateau_patience,
        threshold=0.005, threshold_mode="rel", min_lr=1e-6,
    )
    config = {
        "experiment": (
            f"v42_gate1f_{contact_rotation_mode}"
            if contact_rotation_mode is not None
            else ("v42_gate1e_protected_angular_dynamics"
                  if angular_dynamics
                  else "v42_gate1d_protected_attention_rotation")
        ),
        "model_contract_version": (
            f"gate1f_{contact_rotation_mode}_v1"
            if contact_rotation_mode is not None
            else ("gate1e_v1" if angular_dynamics else "gate1d_v1")
        ),
        "seed": seed, "device": device, "epochs": epochs,
        "draws_per_epoch": draws_per_epoch, "lr": lr,
        "accumulation": accumulation, "patience": patience,
        "plateau_patience": plateau_patience,
        "min_eligible_epoch": min_eligible_epoch,
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "gate1b_checkpoint": str(source_checkpoint),
        "gate1b_checkpoint_sha256": hashlib.sha256(
            source_checkpoint.read_bytes()
        ).hexdigest(),
        "gate1b_source_epoch": source_state["epoch"],
        "starting_model_sha256": state_sha256(model.state_dict()),
        "trainable_prefixes": list(ROTATION_PREFIXES),
        "protected_prefixes": list(PROTECTED_PREFIXES),
        "validation_com_bit_identity_checked_episodes": checked_episodes,
        "dino_mode": "zero",
        "rotation_parameterization": "axis_angle_exp_map_identity_H1",
        "rotation_output": (
            ("contact-gated absolute residual rotation"
             if contact_rotation_mode == "absolute"
             else "contact-impulse damped angular integration"
             if contact_rotation_mode == "impulse"
             else "integrated angular-velocity dynamics"
             if angular_dynamics
             else "independent absolute rotation per frame")
        ),
        "contact_rotation_mode": contact_rotation_mode,
        "angular_damping": angular_damping,
        "rotation_attention_inputs": [
            "detached_physical_point_features",
            "normalized_reference_position",
            "ballistic_floor_gap",
            "finite_difference_velocity_step",
        ],
        "stage_metadata_is_model_input": False,
        "contact_gate_inputs": [
            "detached_physical_point_features",
            "normalized_reference_position",
            "ballistic_floor_gap",
            "finite_difference_velocity_step",
        ] if contact_rotation_mode is not None else [],
        "loss_weights": {
            "full_trajectory_chordal": 1.0,
            "key_horizon_chordal": 0.25,
            "event_emphasized_chordal": 0.50,
            "rigid_fit": 0.25,
            "angular_velocity": (
                0.50 if angular_dynamics
                or contact_rotation_mode == "impulse" else 0.0
            ),
            "angular_acceleration": (
                0.25 if angular_dynamics
                or contact_rotation_mode == "impulse" else 0.0
            ),
            "contact_bce": 0.25 if contact_rotation_mode is not None else 0.0,
            "contact_point": 0.10 if contact_rotation_mode is not None else 0.0,
        },
        "selection": {
            "minimum_epoch": min_eligible_epoch,
            "gate1f_active_rotation_threshold_rad": (
                ACTIVE_ROTATION_RAD
                if contact_rotation_mode is not None else None
            ),
            "gate1f_inactive_rotation_threshold_rad": (
                INACTIVE_ROTATION_RAD
                if contact_rotation_mode is not None else None
            ),
            "gate1f_active_checkpoint_metric": (
                "UID/stratum-pooled rigid active-frame geodesic"
                if contact_rotation_mode is not None else None
            ),
            "overall_postcontact_identity_improvement": "at least 1%",
            "each_stratum_postcontact": "no regression",
            "each_stratum_h59_relative_to_identity": "at most 1.10",
            "required_strata": [
                "rigid/panel_Z", "rigid/panel_V",
                *([] if contact_rotation_mode is not None
                  else ["soft_body/panel_Z"]),
            ],
            "soft_body_role": (
                "reported safety diagnostic"
                if contact_rotation_mode is not None
                else "required identity-screen stratum"
            ),
            "test_used": False,
        },
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    best_total, best_eligible = float("inf"), float("inf")
    eligible_stale, has_eligible, start_epoch = 0, False, 1
    last_path = output / "last.pt"
    if resume and last_path.exists() and not (
        output / "RUN_COMPLETE.json"
    ).exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state["config"] != config:
            raise ValueError("refusing to resume changed Gate-1D configuration")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best_total = float(state["best_total"])
        best_eligible = float(state["best_eligible"])
        eligible_stale = int(state["eligible_stale"])
        has_eligible = bool(state["has_eligible"])
        start_epoch = int(state["epoch"]) + 1
        sampler.epoch = int(state["sampler_epoch"])
        restore_rng_state(state["rng_state"])
    mode = "a" if start_epoch > 1 else "w"
    epoch = start_epoch - 1
    with (output / "history.jsonl").open(mode) as history:
        for epoch in range(start_epoch, epochs + 1):
            started = time.perf_counter()
            train = run_rotation_epoch(
                model, train_loader, torch.device(device), optimizer,
                accumulation, max_batches,
                angular_dynamics, contact_rotation_mode,
            )
            validation = run_rotation_epoch(
                model, validation_loader, torch.device(device), None,
                accumulation, max_batches,
                angular_dynamics, contact_rotation_mode,
            )
            if contact_rotation_mode is not None:
                rigid_activity = [
                    validation["activity"][group]
                    for group in ("rigid/panel_Z", "rigid/panel_V")
                    if group in validation["activity"]
                ]
                active_count = sum(
                    item["active_count"] for item in rigid_activity
                )
                selection = sum(
                    item["active_model_rad"] * item["active_count"]
                    for item in rigid_activity
                ) / max(active_count, 1)
            else:
                selection = validation["rotation_objective"]
            scheduler.step(selection)
            total_improved = selection < best_total
            best_total = min(best_total, selection)
            screen_passed, screen = (
                gate1f_screen(validation)
                if contact_rotation_mode is not None
                else identity_screen(validation)
            )
            eligible = screen_passed and epoch >= min_eligible_epoch
            eligible_improved = eligible and selection < best_eligible
            if eligible_improved:
                best_eligible = selection
                eligible_stale = 0
                has_eligible = True
            elif has_eligible:
                eligible_stale += 1
            if not protected_is_identical(model, protected):
                raise RuntimeError("protected Gate-1B parameters changed")
            record = {
                "epoch": epoch, "train": train, "validation": validation,
                "selection": selection, "screen": screen,
                "checkpoint_eligible": eligible,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - started,
            }
            history.write(json.dumps(record) + "\n")
            history.flush()
            state = {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "config": config,
                "epoch": epoch, "train": train, "validation": validation,
                "selection": selection, "best_total": best_total,
                "best_eligible": best_eligible,
                "eligible_stale": eligible_stale,
                "has_eligible": has_eligible,
                "sampler_epoch": sampler.epoch,
                "rng_state": rng_state(),
            }
            atomic_torch_save(state, last_path)
            if total_improved:
                atomic_torch_save(state, output / "best_total.pt")
            if eligible_improved:
                atomic_torch_save(state, output / "best.pt")
            print(
                f"epoch={epoch:03d} rotation={selection:.6g} "
                f"screen={screen_passed} eligible={eligible}",
                flush=True,
            )
            if has_eligible and eligible_stale >= patience:
                break
    screen_passed = (output / "best.pt").exists()
    report_checkpoint = (
        output / "best.pt" if screen_passed else output / "best_total.pt"
    )
    report_state = torch.load(
        report_checkpoint, map_location=device, weights_only=False,
    )
    model.load_state_dict(report_state["model"])
    baseline_path = output / "VALIDATION_BASELINES.json"
    write_gate1_validation_baselines(
        model, validation_loader, torch.device(device), baseline_path,
    )
    _, reported_screen = (
        gate1f_screen(report_state["validation"])
        if contact_rotation_mode is not None
        else identity_screen(report_state["validation"])
    )
    completion = {
        "status": "complete",
        "last_epoch": epoch,
        "identity_checkpoint_screen_passed": (
            screen_passed if contact_rotation_mode is None else None
        ),
        "gate1f_checkpoint_screen_passed": (
            screen_passed if contact_rotation_mode is not None else None
        ),
        "best_total_selection": best_total,
        "best_eligible_selection": (
            best_eligible if screen_passed else None
        ),
        "reported_epoch": report_state["epoch"],
        "reported_identity_screen": reported_screen,
        "reported_checkpoint": report_checkpoint.name,
        "reported_checkpoint_sha256": hashlib.sha256(
            report_checkpoint.read_bytes()
        ).hexdigest(),
        "last_checkpoint_sha256": hashlib.sha256(
            last_path.read_bytes()
        ).hexdigest(),
        "history_sha256": hashlib.sha256(
            (output / "history.jsonl").read_bytes()
        ).hexdigest(),
        "validation_baselines_sha256": hashlib.sha256(
            baseline_path.read_bytes()
        ).hexdigest(),
        "protected_parameters_bit_identical": protected_is_identical(
            model, protected
        ),
        "validation_com_bit_identical_to_gate1b": True,
    }
    (output / "RUN_COMPLETE.json").write_text(
        json.dumps(completion, indent=2) + "\n"
    )
    return report_checkpoint
