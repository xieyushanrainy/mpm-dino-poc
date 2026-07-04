import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mpm_dino.data import ScenePairDataset
from mpm_dino.losses import one_step_loss
from mpm_dino.model import ParticleGridSurrogate


MODEL_KEYS = ["positions", "velocities", "dino", "particle_mask", "dino_imputed", "controller_positions", "controller_velocity", "controller_mask", "scale", "dt"]
SELECTION_METRIC = "val_particle_mean"


def default_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def run_epoch(model, loader, device, particle_beta, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {k: 0.0 for k in ["objective", "particle_loss", "occupancy", "grid_velocity", "consistency"]}
    particle_error_sum = persistence_error_sum = 0.0
    valid_count = batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = move(batch, device)
            prediction = model(**{k: batch[k] for k in MODEL_KEYS})
            losses = one_step_loss(prediction, batch, model.spec, particle_beta=particle_beta)
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses.total.backward()
                optimizer.step()
            totals["objective"] += float(losses.total.detach().cpu())
            totals["particle_loss"] += float(losses.particle.detach().cpu())
            totals["occupancy"] += float(losses.occupancy.detach().cpu())
            totals["grid_velocity"] += float(losses.grid_velocity.detach().cpu())
            totals["consistency"] += float(losses.consistency.detach().cpu())
            mask = batch["target_mask"]
            count = int(mask.sum().detach().cpu())
            error = torch.linalg.vector_norm(prediction.displacement - batch["target_displacement"], dim=-1)
            persistence = torch.linalg.vector_norm(batch["target_displacement"], dim=-1)
            particle_error_sum += float((error * mask).sum().detach().cpu())
            persistence_error_sum += float((persistence * mask).sum().detach().cpu())
            valid_count += count; batches += 1
    metrics = {key: value / max(batches, 1) for key, value in totals.items()}
    metrics["particle_mean"] = particle_error_sum / max(valid_count, 1)
    metrics["persistence_mean"] = persistence_error_sum / max(valid_count, 1)
    metrics["valid_points"] = valid_count
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train the one-step particle-grid POC.", fromfile_prefix_chars="@")
    parser.add_argument("caches", nargs="+")
    parser.add_argument("--val-caches", nargs="+", default=[])
    parser.add_argument("--epochs", type=int, default=20, help="Maximum additional epochs, including when resuming")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--output", default="runs/one_step")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--particle-beta", type=float, default=0.01)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--min-relative-improvement", type=float, default=0.005,
                        help="Relative validation-particle improvement required to reset early stopping")
    parser.add_argument("--rollout-gate-ratio", type=float, default=0.9,
                        help="Ready when validation particle mean <= this times persistence")
    parser.add_argument("--base", type=int, default=24)
    parser.add_argument("--resolution", type=int, default=32)
    args = parser.parse_args()
    if args.early_stop_patience <= args.lr_patience:
        raise ValueError("early-stop-patience must exceed lr-patience so LR reduction gets a chance")
    train_data = ScenePairDataset(args.caches)
    val_data = ScenePairDataset(args.val_caches) if args.val_caches else None
    dino_dim = train_data.scenes[0]["dino"].shape[-1]
    device = torch.device(args.device)
    resumed = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else None
    if resumed:
        args.base, args.resolution = resumed["args"]["base"], resumed["args"]["resolution"]
    model = ParticleGridSurrogate(dino_dim=dino_dim, base=args.base, resolution=args.resolution).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience,
        threshold=args.min_relative_improvement, threshold_mode="rel", min_lr=args.min_lr,
    )
    start_epoch, best_particle, best_objective, reference_best = 1, float("inf"), float("inf"), float("inf")
    stale_epochs = 0
    compatible_resume = (
        resumed
        and resumed.get("selection_metric") == SELECTION_METRIC
        and resumed.get("args", {}).get("particle_beta") == args.particle_beta
    )
    if resumed:
        model.load_state_dict(resumed["model"])
        start_epoch = resumed["epoch"] + 1
        if compatible_resume:
            optimizer.load_state_dict(resumed["optimizer"])
            scheduler.load_state_dict(resumed["scheduler"])
            best_particle = resumed.get("best_particle", best_particle)
            best_objective = resumed.get("best_objective", best_objective)
            reference_best = resumed.get("early_stop_reference", best_particle)
            stale_epochs = resumed.get("stale_epochs", 0)
        else:
            print("weights-only resume: reset optimizer/scheduler because the particle objective or selection metric changed")
        print(f"resumed {args.checkpoint} at epoch {resumed['epoch']}")
    if args.lr is not None:
        for group in optimizer.param_groups: group["lr"] = args.lr
        print(f"set learning rate to {args.lr:g}")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size) if val_data else None
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.jsonl"
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train = run_epoch(model, train_loader, device, args.particle_beta, optimizer)
        val = run_epoch(model, val_loader, device, args.particle_beta) if val_loader else train
        scheduler.step(val["particle_mean"])
        gate_ratio = val["particle_mean"] / max(val["persistence_mean"], 1e-12)
        rollout_ready = gate_ratio <= args.rollout_gate_ratio
        record = {
            "epoch": epoch, "train": train, "val": val,
            "lr": optimizer.param_groups[0]["lr"], "rollout_gate_ratio": gate_ratio,
            "rollout_ready": rollout_ready,
        }
        print(
            f"epoch={epoch:03d} train_particle={train['particle_mean']:.6g} "
            f"val_particle={val['particle_mean']:.6g} persistence={val['persistence_mean']:.6g} "
            f"ratio={gate_ratio:.3f} objective={val['objective']:.6g} "
            f"lr={record['lr']:.3g} rollout_ready={rollout_ready}"
        )
        with history_path.open("a") as handle: handle.write(json.dumps(record) + "\n")

        exact_improvement = val["particle_mean"] < best_particle
        objective_improvement = val["objective"] < best_objective
        meaningful_improvement = val["particle_mean"] < reference_best * (1 - args.min_relative_improvement)
        if meaningful_improvement:
            reference_best = val["particle_mean"]; stale_epochs = 0
        else:
            stale_epochs += 1
        best_particle = min(best_particle, val["particle_mean"])
        best_objective = min(best_objective, val["objective"])
        state = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "selection_metric": SELECTION_METRIC,
            "best_particle": best_particle, "best_objective": best_objective,
            "early_stop_reference": reference_best, "stale_epochs": stale_epochs,
            "rollout_ready": rollout_ready, "rollout_gate_ratio": gate_ratio, "args": vars(args),
        }
        torch.save(state, output / "last.pt")
        if exact_improvement: torch.save(state, output / "best.pt")
        if objective_improvement: torch.save(state, output / "best_objective.pt")
        if stale_epochs >= args.early_stop_patience:
            print(f"early stopping: <{args.min_relative_improvement:.2%} relative improvement for {stale_epochs} epochs")
            break


if __name__ == "__main__":
    main()
