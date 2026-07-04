"""Full-backpropagation rollout fine-tuning for the particle-grid surrogate."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from mpm_dino.data import SceneSequenceDataset
from mpm_dino.losses import one_step_loss
from mpm_dino.model import ParticleGridSurrogate


MODEL_KEYS = ["positions", "velocities", "dino", "particle_mask", "dino_imputed", "controller_positions", "controller_velocity", "controller_mask", "scale", "dt"]


def default_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def move(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def model_call(model, positions, velocities, batch, step, mask):
    dt = batch["dt"]
    controller_velocity = (batch["controller"][:, step + 1] - batch["controller"][:, step]) / dt[:, None, None]
    return model(
        positions, velocities, batch["dino"], mask, batch["dino_imputed"],
        batch["controller"][:, step], controller_velocity,
        torch.ones(controller_velocity.shape[:2], dtype=torch.bool, device=positions.device),
        batch["scale"], dt,
    )


def loss_batch(current_positions, output, batch, step, initial_mask, model, particle_beta):
    target_position = batch["points"][:, step + 1]
    target_mask = initial_mask & batch["visible"][:, step] & batch["visible"][:, step + 1]
    target_mask &= batch["motion_valid"][:, step] & batch["motion_valid"][:, step + 1]
    target_displacement = target_position - current_positions
    target_velocity = (batch["points"][:, step + 1] - batch["points"][:, step]) / batch["dt"][:, None, None]
    loss_input = {
        "positions": current_positions, "target_displacement": target_displacement,
        "target_velocity": target_velocity, "target_mask": target_mask,
    }
    return one_step_loss(output, loss_input, model.spec, particle_beta), target_position, target_mask


def run_epoch(model, loader, device, steps, particle_beta, discount, teacher_weight, optimizer=None):
    training = optimizer is not None
    model.train(training)
    objective_sum = recurrent_error = persistence_error = teacher_error = 0.0
    recurrent_count = teacher_count = batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw in loader:
            batch = move(raw, device)
            dt = batch["dt"]
            positions = batch["points"][:, 0]
            velocities = (positions - batch["previous_points"]) / dt[:, None, None]
            initial_mask = batch["visible"][:, 0] & batch["motion_valid"][:, 0]
            start_position = positions
            weighted_objective = 0.0; weight_sum = 0.0
            for step in range(steps):
                output = model_call(model, positions, velocities, batch, step, initial_mask)
                losses, target_position, target_mask = loss_batch(
                    positions, output, batch, step, initial_mask, model, particle_beta
                )
                weight = discount ** step
                weighted_objective = weighted_objective + weight * losses.total
                weight_sum += weight
                positions = positions + output.displacement
                velocities = output.displacement / dt[:, None, None]
                error = torch.linalg.vector_norm(positions - target_position, dim=-1)
                persistence = torch.linalg.vector_norm(start_position - target_position, dim=-1)
                recurrent_error += float((error * target_mask).sum().detach().cpu())
                persistence_error += float((persistence * target_mask).sum().detach().cpu())
                recurrent_count += int(target_mask.sum().detach().cpu())

            # Teacher-forced auxiliary update at the final step of this window.
            teacher_step = steps - 1
            teacher_position = batch["points"][:, teacher_step]
            teacher_velocity = (teacher_position - batch["points"][:, teacher_step - 1]) / dt[:, None, None]
            teacher_mask = batch["visible"][:, teacher_step] & batch["motion_valid"][:, teacher_step]
            teacher_output = model_call(model, teacher_position, teacher_velocity, batch, teacher_step, teacher_mask)
            teacher_losses, teacher_target, teacher_target_mask = loss_batch(
                teacher_position, teacher_output, batch, teacher_step, teacher_mask, model, particle_beta
            )
            objective = weighted_objective / weight_sum + teacher_weight * teacher_losses.total
            if training:
                optimizer.zero_grad(set_to_none=True); objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            teacher_prediction = teacher_position + teacher_output.displacement
            teacher_distance = torch.linalg.vector_norm(teacher_prediction - teacher_target, dim=-1)
            teacher_error += float((teacher_distance * teacher_target_mask).sum().detach().cpu())
            teacher_count += int(teacher_target_mask.sum().detach().cpu())
            objective_sum += float(objective.detach().cpu()); batches += 1
    return {
        "objective": objective_sum / max(batches, 1),
        "recurrent_particle_mean": recurrent_error / max(recurrent_count, 1),
        "recurrent_persistence_mean": persistence_error / max(recurrent_count, 1),
        "teacher_particle_mean": teacher_error / max(teacher_count, 1),
        "valid_recurrent_points": recurrent_count,
    }


def make_train_loader(dataset, batch_size, oversample):
    scores = np.asarray(dataset.motion_scores)
    threshold = np.median(scores)
    weights = np.where(scores >= threshold, oversample, 1.0)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("caches", nargs="+")
    parser.add_argument("--val-caches", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--output", default="runs/rollout_s2")
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--particle-beta", type=float, default=0.01)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--teacher-weight", type=float, default=0.5)
    parser.add_argument("--motion-oversample", type=float, default=2.0)
    parser.add_argument("--no-regression-ratio", type=float, default=1.1)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--min-relative-improvement", type=float, default=0.005)
    args = parser.parse_args()
    if args.early_stop_patience <= args.lr_patience:
        raise ValueError("early-stop-patience must exceed lr-patience")
    train_data = SceneSequenceDataset(args.caches, args.steps)
    val_data = SceneSequenceDataset(args.val_caches, args.steps)
    train_loader = make_train_loader(train_data, args.batch_size, args.motion_oversample)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    model = ParticleGridSurrogate(
        dino_dim=train_data.scenes[0]["dino"].shape[-1],
        base=checkpoint["args"]["base"], resolution=checkpoint["args"]["resolution"],
    ).to(device)
    args.base = checkpoint["args"]["base"]
    args.resolution = checkpoint["args"]["resolution"]
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=args.lr_patience,
        threshold=args.min_relative_improvement, threshold_mode="rel", min_lr=1e-6,
    )
    current = run_epoch(model, val_loader, device, args.steps, args.particle_beta, args.discount, args.teacher_weight)
    compatible_resume = checkpoint.get("best_recurrent") is not None and checkpoint.get("args", {}).get("steps") == args.steps
    start_epoch = 1
    if compatible_resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        baseline_teacher = checkpoint["baseline_teacher"]
        best_recurrent = checkpoint["best_recurrent"]
        reference_best = checkpoint.get("early_stop_reference", best_recurrent)
        stale = checkpoint.get("stale_epochs", 0)
        print(f"resumed rollout state at epoch {checkpoint['epoch']} with lr={optimizer.param_groups[0]['lr']:.3g}")
    else:
        baseline_teacher = current["teacher_particle_mean"]
        best_recurrent = reference_best = float("inf")
        stale = 0
        print("weights-only initialization with fresh rollout optimizer/scheduler")
    teacher_limit = baseline_teacher * args.no_regression_ratio
    print(f"current recurrent={current['recurrent_particle_mean']:.6g} teacher={current['teacher_particle_mean']:.6g} teacher_limit={teacher_limit:.6g}")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train = run_epoch(model, train_loader, device, args.steps, args.particle_beta, args.discount, args.teacher_weight, optimizer)
        val = run_epoch(model, val_loader, device, args.steps, args.particle_beta, args.discount, args.teacher_weight)
        scheduler.step(val["recurrent_particle_mean"])
        no_regression = val["teacher_particle_mean"] <= teacher_limit
        improved = val["recurrent_particle_mean"] < best_recurrent and no_regression
        meaningful = val["recurrent_particle_mean"] < reference_best * (1 - args.min_relative_improvement) and no_regression
        if meaningful: reference_best = val["recurrent_particle_mean"]; stale = 0
        else: stale += 1
        if improved: best_recurrent = val["recurrent_particle_mean"]
        ratio = val["recurrent_particle_mean"] / max(val["recurrent_persistence_mean"], 1e-12)
        record = {"epoch": epoch, "train": train, "val": val, "lr": optimizer.param_groups[0]["lr"], "no_regression": no_regression, "persistence_ratio": ratio}
        print(f"epoch={epoch:03d} recurrent={val['recurrent_particle_mean']:.6g} persistence_ratio={ratio:.3f} teacher={val['teacher_particle_mean']:.6g} no_regression={no_regression} lr={record['lr']:.3g}")
        with (output / "history.jsonl").open("a") as handle: handle.write(json.dumps(record) + "\n")
        state = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "best_recurrent": best_recurrent,
            "baseline_teacher": baseline_teacher, "early_stop_reference": reference_best,
            "stale_epochs": stale, "no_regression": no_regression, "args": vars(args),
        }
        torch.save(state, output / "last.pt")
        if improved: torch.save(state, output / "best.pt")
        if stale >= args.early_stop_patience:
            print(f"early stopping after {stale} epochs without meaningful guarded improvement"); break


if __name__ == "__main__":
    main()
