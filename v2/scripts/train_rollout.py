#!/usr/bin/env python3
"""Full-backpropagation V2 rollout fine-tuning with teacher guardrail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from mpm_dino_v2.data import SceneSequenceDataset
from mpm_dino_v2.losses import one_step_loss
from mpm_dino_v2.model import ParticleGridSurrogate


def move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def model_call(model, positions, velocities, batch, step, mask):
    dt = batch["dt"]
    controller_velocity = (batch["controller"][:, step + 1] - batch["controller"][:, step]) / dt[:, None, None]
    return model(
        positions, velocities, batch["dino"], mask, batch["dino_imputed"], batch["controller"][:, step],
        controller_velocity, torch.ones(controller_velocity.shape[:2], dtype=torch.bool, device=positions.device),
        batch["scale"], dt, batch["x0"], batch["neighbour_indices"], batch["neighbour_mask"],
        batch["rest_edge_lengths"],
    )


def loss_at(current, output, batch, step, initial_mask, model, args):
    target = batch["points"][:, step + 1]
    target_mask = initial_mask & batch["visible"][:, step] & batch["visible"][:, step + 1]
    target_mask &= batch["motion_valid"][:, step] & batch["motion_valid"][:, step + 1]
    target_velocity = (batch["points"][:, step + 1] - batch["points"][:, step]) / batch["dt"][:, None, None]
    loss_input = {
        "positions": current, "target_displacement": target - current, "target_velocity": target_velocity,
        "target_mask": target_mask, "neighbour_indices": batch["neighbour_indices"],
        "neighbour_mask": batch["neighbour_mask"],
    }
    losses = one_step_loss(
        output, loss_input, model.spec, args.particle_beta, args.edge_vector_weight, args.edge_length_weight,
    )
    return losses, target, target_mask


def run_epoch(model, loader, device, args, optimizer=None, random_min_steps=None):
    training = optimizer is not None
    model.train(training)
    objective_sum = recurrent_error = persistence_error = teacher_error = 0.0
    edge_vector_sum = edge_length_sum = 0.0
    recurrent_count = teacher_count = batches = total_rollout_steps = 0
    horizon_counts = {step: 0 for step in range(1, args.steps + 1)}
    with torch.enable_grad() if training else torch.no_grad():
        for raw in loader:
            batch = move(raw, device)
            dt = batch["dt"]
            positions = batch["points"][:, 0]
            velocities = (positions - batch["previous_points"]) / dt[:, None, None]
            initial_mask = batch["particle_mask"] & batch["visible"][:, 0] & batch["motion_valid"][:, 0]
            start = positions
            weighted_objective = 0.0
            weight_sum = 0.0
            rollout_steps = args.steps
            if training and random_min_steps is not None:
                rollout_steps = int(torch.randint(random_min_steps, args.steps + 1, ()).item())
            horizon_counts[rollout_steps] += 1
            total_rollout_steps += rollout_steps
            for step in range(rollout_steps):
                output = model_call(model, positions, velocities, batch, step, initial_mask)
                losses, target, target_mask = loss_at(positions, output, batch, step, initial_mask, model, args)
                weight = args.discount ** step
                weighted_objective = weighted_objective + weight * losses.total
                weight_sum += weight
                positions = positions + output.displacement
                velocities = output.displacement / dt[:, None, None]
                recurrent_error += float((torch.linalg.vector_norm(positions - target, dim=-1) * target_mask).sum().detach().cpu())
                persistence_error += float((torch.linalg.vector_norm(start - target, dim=-1) * target_mask).sum().detach().cpu())
                recurrent_count += int(target_mask.sum().detach().cpu())
                edge_vector_sum += float(losses.edge_vector.detach().cpu())
                edge_length_sum += float(losses.edge_length.detach().cpu())

            teacher_step = rollout_steps - 1
            teacher_position = batch["points"][:, teacher_step]
            teacher_previous = batch["previous_points"] if teacher_step == 0 else batch["points"][:, teacher_step - 1]
            teacher_velocity = (teacher_position - teacher_previous) / dt[:, None, None]
            teacher_mask = batch["particle_mask"] & batch["visible"][:, teacher_step] & batch["motion_valid"][:, teacher_step]
            teacher_output = model_call(model, teacher_position, teacher_velocity, batch, teacher_step, teacher_mask)
            teacher_losses, teacher_target, teacher_target_mask = loss_at(
                teacher_position, teacher_output, batch, teacher_step, teacher_mask, model, args,
            )
            objective = weighted_objective / weight_sum + args.teacher_weight * teacher_losses.total
            if training:
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            teacher_prediction = teacher_position + teacher_output.displacement
            teacher_error += float((torch.linalg.vector_norm(teacher_prediction - teacher_target, dim=-1) * teacher_target_mask).sum().detach().cpu())
            teacher_count += int(teacher_target_mask.sum().detach().cpu())
            objective_sum += float(objective.detach().cpu())
            batches += 1
    return {
        "objective": objective_sum / max(batches, 1),
        "recurrent_particle_mean": recurrent_error / max(recurrent_count, 1),
        "recurrent_persistence_mean": persistence_error / max(recurrent_count, 1),
        "teacher_particle_mean": teacher_error / max(teacher_count, 1),
        "edge_vector": edge_vector_sum / max(total_rollout_steps, 1),
        "edge_length": edge_length_sum / max(total_rollout_steps, 1),
        "valid_recurrent_points": recurrent_count,
        "horizon_counts": horizon_counts,
    }


def train_loader(dataset, batch_size, oversample, fraction=1.0):
    scores = np.asarray(dataset.motion_scores)
    weights = np.where(scores >= np.median(scores), oversample, 1.0)
    sample_count = max(1, round(len(weights) * fraction))
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), sample_count, replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def main() -> None:
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("caches", nargs="+")
    parser.add_argument("--val-caches", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, choices=(2, 4), required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", required=True)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--particle-beta", type=float, default=0.01)
    parser.add_argument("--edge-vector-weight", type=float, default=0.25)
    parser.add_argument("--edge-length-weight", type=float, default=0.10)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--teacher-weight", type=float, default=0.5)
    parser.add_argument("--motion-oversample", type=float, default=2.0)
    parser.add_argument("--no-regression-ratio", type=float, default=1.1)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--min-relative-improvement", type=float, default=0.005)
    parser.add_argument("--random-min-steps", type=int, default=None,
                        help="Uniformly sample training horizons from this value through --steps")
    parser.add_argument("--train-fraction", type=float, default=1.0,
                        help="Fraction of a full sampled training epoch between validations")
    args = parser.parse_args()
    if args.early_stop_patience <= args.lr_patience:
        raise ValueError("early-stop-patience must exceed lr-patience")
    if args.random_min_steps is not None and not 1 <= args.random_min_steps <= args.steps:
        raise ValueError("random-min-steps must be between 1 and --steps")
    if not 0 < args.train_fraction <= 1:
        raise ValueError("train-fraction must be in (0, 1]")
    torch.manual_seed(42)
    train_data = SceneSequenceDataset(args.caches, args.steps)
    val_data = SceneSequenceDataset(args.val_caches, args.steps)
    training_loader = train_loader(train_data, args.batch_size, args.motion_oversample, args.train_fraction)
    validation_loader = DataLoader(val_data, batch_size=args.batch_size)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    args.base, args.resolution = checkpoint["args"]["base"], checkpoint["args"]["resolution"]
    model = ParticleGridSurrogate(
        dino_dim=train_data.scenes[0]["dino"].shape[-1], base=args.base, resolution=args.resolution,
        variant=checkpoint["args"].get("variant", "fused"),
    ).to(torch.device(args.device))
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=args.lr_patience, threshold=args.min_relative_improvement,
        threshold_mode="rel", min_lr=1e-6,
    )
    current = run_epoch(model, validation_loader, torch.device(args.device), args)
    baseline_teacher = current["teacher_particle_mean"]
    teacher_limit = baseline_teacher * args.no_regression_ratio
    print(f"initial recurrent={current['recurrent_particle_mean']:.6g} teacher={baseline_teacher:.6g} "
          f"teacher_limit={teacher_limit:.6g}", flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best_recurrent = reference_best = current["recurrent_particle_mean"]
    stale = 0
    scheduler.step(best_recurrent)
    # Epoch zero is the valid guarded incumbent inherited from the previous
    # stage. Fine-tuning must beat it; a regressed epoch is never promoted just
    # because it is the first trained epoch.
    initial_state = {
        "epoch": 0, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "best_recurrent": best_recurrent,
        "baseline_teacher": baseline_teacher, "early_stop_reference": reference_best,
        "stale_epochs": stale, "no_regression": True, "args": vars(args),
    }
    torch.save(initial_state, output / "best.pt")
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(
            model, training_loader, torch.device(args.device), args, optimizer,
            random_min_steps=args.random_min_steps,
        )
        val = run_epoch(model, validation_loader, torch.device(args.device), args)
        if not all(np.isfinite(val[key]) for key in ("objective", "recurrent_particle_mean", "teacher_particle_mean")):
            raise FloatingPointError(f"non-finite rollout validation metrics at epoch {epoch}: {val}")
        scheduler.step(val["recurrent_particle_mean"])
        no_regression = val["teacher_particle_mean"] <= teacher_limit
        improved = val["recurrent_particle_mean"] < best_recurrent and no_regression
        meaningful = val["recurrent_particle_mean"] < reference_best * (1 - args.min_relative_improvement) and no_regression
        if meaningful:
            reference_best, stale = val["recurrent_particle_mean"], 0
        else:
            stale += 1
        if improved:
            best_recurrent = val["recurrent_particle_mean"]
        ratio = val["recurrent_particle_mean"] / max(val["recurrent_persistence_mean"], 1e-12)
        record = {"epoch": epoch, "train": train, "val": val, "lr": optimizer.param_groups[0]["lr"],
                  "no_regression": no_regression, "persistence_ratio": ratio}
        print(f"epoch={epoch:03d} recurrent={val['recurrent_particle_mean']:.6g} ratio={ratio:.3f} "
              f"teacher={val['teacher_particle_mean']:.6g} guard={no_regression} lr={record['lr']:.3g}", flush=True)
        with (output / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        state = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "best_recurrent": best_recurrent,
            "baseline_teacher": baseline_teacher, "early_stop_reference": reference_best,
            "stale_epochs": stale, "no_regression": no_regression, "args": vars(args),
        }
        torch.save(state, output / "last.pt")
        if improved:
            torch.save(state, output / "best.pt")
        if stale >= args.early_stop_patience:
            print(f"early stopping after {stale} epochs without meaningful guarded improvement", flush=True)
            break


if __name__ == "__main__":
    main()
