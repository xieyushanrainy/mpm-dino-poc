from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import dataset, targets_and_stages
from .memory import PHASES, SourcePhase, V5MemoryBank


@torch.no_grad()
def build_training_bank(root, manifest, output: str | Path, seed: int = 42) -> V5MemoryBank:
    """Rebuild the permitted 20-soft-UID, three-phase mechanics bank."""
    source = dataset(root, manifest, "train", families=("soft_body",), seed=seed)
    if len(source.uids) != 20:
        raise ValueError(f"expected 20 soft training UIDs, found {len(source.uids)}")
    entries = []
    for uid in source.uids:
        # Panel-Z is the sole soft episode in the present dataset. Choosing the
        # first indexed episode remains deterministic if another is added.
        sample = source[source.by_uid[uid][0]]
        batch = {
            key: (value[None] if torch.is_tensor(value) else [value])
            for key, value in sample.items()
        }
        targets, stages = targets_and_stages(batch)
        labels = stages.labels[0, 1:]
        for phase in PHASES:
            candidates = torch.where(labels.eq(phase))[0]
            if len(candidates):
                frame = int(candidates[len(candidates) // 2])
                point_valid = batch["target_mask"][0, frame]
                deformation = targets.displacement[0, frame]
                contact = (
                    batch["target"][0, frame, :, 2] - batch["floor_z"][0]
                    <= 0.01 * targets.radius[0]
                ) & point_valid
                dino_valid = batch["dino_valid"][0]
            else:
                # Preserve the existing V4.3 three-entry scope. A phase absent
                # for one UID becomes an explicitly empty entry, never a
                # fabricated target or an extra source phase.
                point_valid = torch.zeros_like(batch["input_mask"][0])
                deformation = torch.zeros_like(targets.reference_shape[0])
                contact = torch.zeros_like(point_valid)
                dino_valid = torch.zeros_like(batch["dino_valid"][0])
            entries.append(SourcePhase(
                uid=uid,
                split="train",
                phase=phase,
                coordinates=targets.reference_shape[0].cpu(),
                deformation=deformation.cpu(),
                contact=contact.cpu(),
                point_valid=point_valid.cpu(),
                dino=batch["dino"][0].cpu(),
                dino_valid=dino_valid.cpu(),
            ))
    bank = V5MemoryBank(entries, source.uids)
    bank.save(output)
    return bank
