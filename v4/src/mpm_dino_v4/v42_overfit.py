from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from .v41_data import MODEL_INPUT_KEYS, V41TrajectoryDataset
from .v41_train import atomic_torch_save, move, seed_all
from .v42_gate2 import (
    LOCAL_PREFIXES, _batch_targets_and_stages, file_sha256,
    load_gate1e_source, local_parameters, protected_is_identical,
    protected_snapshot,
)
from .v42_losses import compute_v42_local_losses
from .v42_stages import total_mass_stage_weights


OVERFIT_MODES = ("single_frame", "single_episode")
OVERFIT_OBJECTIVES = ("composite", "canonical_only")
LOSS_OPTIONS = {
    "soft_deformation_amplification_cap": 1.0,
    "soft_deformation_quantile": 0.95,
    "soft_deformation_floor_fraction": 0.005,
    "family_balanced": True,
    "rigid_family_weight": 0.25,
    "rigid_zero_weight": 0.0,
}
PASS_THRESHOLDS = {
    "single_frame": {
        "canonical_error_reduction": 0.95,
        "predicted_to_target_magnitude_ratio_min": 0.90,
        "predicted_to_target_magnitude_ratio_max": 1.10,
    },
    "single_episode": {
        "canonical_error_reduction": 0.95,
        "magnitude_correlation": 0.95,
        "predicted_to_target_peak_ratio_min": 0.90,
        "predicted_to_target_peak_ratio_max": 1.10,
        "peak_timing_error_frames": 1,
    },
}


def select_peak_soft_training_example(dataset: V41TrajectoryDataset) -> dict:
    """Select the strongest canonical deformation using training data only."""
    best = None
    for index, row in enumerate(dataset.rows):
        if row["family"] != "soft_body" or row["initial_velocity_regime"] != "zero":
            continue
        batch = default_collate([dataset[index]])
        targets, _ = _batch_targets_and_stages(batch)
        mask = batch["target_mask"] & targets.valid_rotation[:, :, None]
        magnitude = torch.sqrt(
            (targets.displacement.square().sum(-1) * mask).sum(2)
            / mask.sum(2).clamp_min(1)
        ) / targets.radius[:, None]
        frame = int(magnitude[0].argmax())
        candidate = {
            "dataset_index": index,
            "uid": row["uid"],
            "episode_id": row["episode_id"],
            "target_frame_index": frame,
            "saved_frame_index": frame + 2,
            "target_peak_nrmse": float(magnitude[0, frame]),
        }
        if best is None or candidate["target_peak_nrmse"] > best["target_peak_nrmse"]:
            best = candidate
    if best is None:
        raise RuntimeError("no Panel-Z soft-body training episode is available")
    return best


def _magnitude_curve(displacement, mask, radius):
    return torch.sqrt(
        (displacement.square().sum(-1) * mask).sum(2)
        / mask.sum(2).clamp_min(1)
    ) / radius[:, None]


def _canonical_error(predicted, target, mask, radius):
    return float(torch.sqrt(
        ((predicted - target).square().sum(-1) * mask).sum()
        / mask.sum().clamp_min(1)
    ) / radius[0])


def overfit_metrics(output, targets, target_mask, mode, peak_frame, initial_error):
    valid = target_mask & targets.valid_rotation[:, :, None]
    if mode == "single_frame":
        valid = torch.zeros_like(valid).scatter_(
            1,
            torch.tensor([[[peak_frame]]], device=valid.device).expand(
                valid.shape[0], 1, valid.shape[2]
            ),
            valid[:, peak_frame:peak_frame + 1],
        )
    error = _canonical_error(
        output.canonical_displacement, targets.displacement,
        valid, targets.radius,
    )
    target_curve = _magnitude_curve(targets.displacement, valid, targets.radius)[0]
    predicted_curve = _magnitude_curve(
        output.canonical_displacement, valid, targets.radius,
    )[0]
    reduction = 1.0 - error / max(initial_error, 1e-12)
    if mode == "single_frame":
        target_magnitude = float(target_curve[peak_frame])
        predicted_magnitude = float(predicted_curve[peak_frame])
        ratio = predicted_magnitude / max(target_magnitude, 1e-12)
        values = {
            "canonical_nrmse": error,
            "canonical_error_reduction": reduction,
            "target_magnitude_nrmse": target_magnitude,
            "predicted_magnitude_nrmse": predicted_magnitude,
            "predicted_to_target_magnitude_ratio": ratio,
        }
    else:
        target_np = target_curve.detach().cpu().numpy()
        predicted_np = predicted_curve.detach().cpu().numpy()
        correlation = (
            float(np.corrcoef(target_np, predicted_np)[0, 1])
            if np.std(target_np) > 0 and np.std(predicted_np) > 0 else 0.0
        )
        target_peak = int(target_np.argmax())
        predicted_peak = int(predicted_np.argmax())
        values = {
            "canonical_nrmse": error,
            "canonical_error_reduction": reduction,
            "magnitude_correlation": correlation,
            "target_peak_frame": target_peak,
            "predicted_peak_frame": predicted_peak,
            "peak_timing_error_frames": abs(predicted_peak - target_peak),
            "target_peak_nrmse": float(target_np[target_peak]),
            "predicted_peak_nrmse": float(predicted_np[predicted_peak]),
            "predicted_to_target_peak_ratio": (
                float(predicted_np[predicted_peak])
                / max(float(target_np[target_peak]), 1e-12)
            ),
        }
    values["finite"] = all(
        math.isfinite(value) for value in values.values()
        if isinstance(value, float)
    )
    return values


def overfit_passed(mode, metrics):
    threshold = PASS_THRESHOLDS[mode]
    if mode == "single_frame":
        ratio = metrics["predicted_to_target_magnitude_ratio"]
        return bool(
            metrics["finite"]
            and metrics["canonical_error_reduction"] >= threshold["canonical_error_reduction"]
            and threshold["predicted_to_target_magnitude_ratio_min"]
            <= ratio <= threshold["predicted_to_target_magnitude_ratio_max"]
        )
    ratio = metrics["predicted_to_target_peak_ratio"]
    return bool(
        metrics["finite"]
        and metrics["canonical_error_reduction"] >= threshold["canonical_error_reduction"]
        and metrics["magnitude_correlation"] >= threshold["magnitude_correlation"]
        and threshold["predicted_to_target_peak_ratio_min"]
        <= ratio <= threshold["predicted_to_target_peak_ratio_max"]
        and metrics["peak_timing_error_frames"]
        <= threshold["peak_timing_error_frames"]
    )


def select_overfit_objective(losses, objective):
    if objective == "composite":
        return losses.total
    if objective == "canonical_only":
        return losses.canonical
    raise ValueError(f"unknown overfit objective: {objective}")


def train_overfit_mode(
    dataset_root, manifest, gate1e_checkpoint, output, seed, mode,
    device="cuda", steps=2000, lr=1e-3, log_every=25,
    objective="composite",
):
    if mode not in OVERFIT_MODES:
        raise ValueError(f"unknown overfit mode: {mode}")
    if objective not in OVERFIT_OBJECTIVES:
        raise ValueError(f"unknown overfit objective: {objective}")
    if objective == "canonical_only" and mode != "single_frame":
        raise ValueError("canonical-only audit is defined for single_frame only")
    seed_all(seed)
    device = torch.device(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = V41TrajectoryDataset(
        dataset_root, manifest, "train", "zero", seed,
        families=("soft_body",),
    )
    selection = select_peak_soft_training_example(dataset)
    batch = move(default_collate([dataset[selection["dataset_index"]]]), device)
    targets, stages = _batch_targets_and_stages(batch)
    target_mask = batch["target_mask"].clone()
    peak_frame = selection["target_frame_index"]
    loss_batch = dict(batch)
    if mode == "single_frame":
        loss_batch["target_mask"] = torch.zeros_like(target_mask)
        loss_batch["target_mask"][:, peak_frame] = target_mask[:, peak_frame]
        frame_weights = torch.zeros(
            target_mask.shape[:2], device=device, dtype=targets.radius.dtype,
        )
        frame_weights[:, peak_frame] = 1.0
    else:
        frame_weights = total_mass_stage_weights(stages.labels[:, 1:]).detach()

    model, source_state = load_gate1e_source(gate1e_checkpoint, "geometry", device)
    zero_model, _ = load_gate1e_source(gate1e_checkpoint, "zero", device)
    model.eval()  # deterministic capacity test: dropout remains disabled
    zero_model.eval()
    protected = protected_snapshot(model)
    parameters = local_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    inputs = {key: batch[key] for key in MODEL_INPUT_KEYS}
    with torch.no_grad():
        initial_output = model(**inputs)
        source_output = zero_model(**inputs)
    initial_valid = target_mask & targets.valid_rotation[:, :, None]
    if mode == "single_frame":
        selected_valid = torch.zeros_like(initial_valid)
        selected_valid[:, peak_frame] = initial_valid[:, peak_frame]
        initial_valid = selected_valid
    initial_error = _canonical_error(
        initial_output.canonical_displacement, targets.displacement,
        initial_valid, targets.radius,
    )
    initial_metrics = overfit_metrics(
        initial_output, targets, target_mask, mode, peak_frame, initial_error,
    )
    history_path = output / "history.jsonl"
    best_error = float("inf")
    best_step = 0
    started = time.time()
    with history_path.open("w") as history:
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(**inputs)
            losses = compute_v42_local_losses(
                prediction, loss_batch, targets=targets,
                frame_weights=frame_weights, **LOSS_OPTIONS,
            )
            optimized_loss = select_overfit_objective(losses, objective)
            optimized_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            optimizer.step()
            should_measure = step == 1 or step % log_every == 0 or step == steps
            if not should_measure:
                continue
            with torch.no_grad():
                measured = model(**inputs)
                metrics = overfit_metrics(
                    measured, targets, target_mask, mode, peak_frame,
                    initial_error,
                )
            record = {
                "step": step,
                "objective": objective,
                "optimized_loss": float(optimized_loss.detach()),
                "loss": float(losses.total.detach()),
                "canonical_loss": float(losses.canonical.detach()),
                "strain_loss": float(losses.strain.detach()),
                "gradient_norm": float(gradient_norm),
                "metrics": metrics,
            }
            history.write(json.dumps(record) + "\n")
            history.flush()
            if metrics["canonical_nrmse"] < best_error:
                best_error = metrics["canonical_nrmse"]
                best_step = step
                atomic_torch_save({
                    "model": model.state_dict(), "step": step,
                    "metrics": metrics, "selection": selection,
                }, output / "best.pt")
            print(
                f"mode={mode} objective={objective} step={step:04d} "
                f"optimized={record['optimized_loss']:.6g} "
                f"canonical={metrics['canonical_nrmse']:.6g}",
                flush=True,
            )
            if overfit_passed(mode, metrics):
                break

    best_state = torch.load(
        output / "best.pt", map_location=device, weights_only=False,
    )
    model.load_state_dict(best_state["model"])
    with torch.no_grad():
        final_output = model(**inputs)
    final_metrics = overfit_metrics(
        final_output, targets, target_mask, mode, peak_frame, initial_error,
    )
    global_identity = bool(
        torch.equal(final_output.com, source_output.com)
        and torch.equal(final_output.rotation, source_output.rotation)
    )
    result = {
        "experiment": (
            "v42_decoder_canonical_only_overfit"
            if objective == "canonical_only"
            else "v42_decoder_learnability_overfit"
        ),
        "mode": mode,
        "objective": objective,
        "status": "complete",
        "passed": overfit_passed(mode, final_metrics),
        "thresholds": PASS_THRESHOLDS[mode],
        "selection": selection,
        "initial_metrics": initial_metrics,
        "best_step": best_step,
        "final_metrics": final_metrics,
        "protected_parameters_bit_identical": protected_is_identical(model, protected),
        "com_rotation_bit_identical_to_gate1e": global_identity,
        "test_data_used": False,
        "validation_data_used": False,
        "real_dino_trained": False,
        "elapsed_seconds": time.time() - started,
        "source_checkpoint_sha256": file_sha256(gate1e_checkpoint),
        "source_epoch": source_state["epoch"],
        "history_sha256": file_sha256(history_path),
        "best_checkpoint_sha256": file_sha256(output / "best.pt"),
        "trainable_prefixes": list(LOCAL_PREFIXES),
        "loss_contract": LOSS_OPTIONS,
    }
    result_path = output / "OVERFIT_RESULT.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    (output / "RUN_COMPLETE.json").write_text(json.dumps({
        "status": "complete", "passed": result["passed"],
        "mode": mode, "best_step": best_step,
        "result_sha256": file_sha256(result_path),
    }, indent=2) + "\n")
    return result
