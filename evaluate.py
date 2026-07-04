import argparse

import torch
from torch.utils.data import DataLoader

from mpm_dino.data import ScenePairDataset
from mpm_dino.model import ParticleGridSurrogate


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def masked_distance(prediction, target, mask):
    distance = torch.linalg.vector_norm(prediction - target, dim=-1)
    weight = mask.to(distance.dtype)
    return (distance * weight).sum() / weight.sum().clamp_min(1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate one-step particle displacement and simple baselines.", fromfile_prefix_chars="@")
    parser.add_argument("checkpoint")
    parser.add_argument("caches", nargs="+")
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    dataset = ScenePairDataset(args.caches)
    dino_dim = dataset.scenes[0]["dino"].shape[-1]
    model = ParticleGridSurrogate(dino_dim=dino_dim, base=config["base"], resolution=config["resolution"]).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    sums = {"model": 0.0, "persistence": 0.0, "constant_velocity": 0.0}; count = 0
    keys = ["positions", "velocities", "dino", "particle_mask", "dino_imputed", "controller_positions", "controller_velocity", "controller_mask", "scale", "dt"]
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**{k: batch[k] for k in keys})
            target, mask = batch["target_displacement"], batch["target_mask"]
            sums["model"] += masked_distance(output.displacement, target, mask).item()
            sums["persistence"] += masked_distance(torch.zeros_like(target), target, mask).item()
            sums["constant_velocity"] += masked_distance(batch["velocities"] * batch["dt"][:, None, None], target, mask).item()
            count += 1
    for name, value in sums.items():
        print(f"{name}_normalized_particle_mean={value / max(count, 1):.8g}")


if __name__ == "__main__":
    main()
