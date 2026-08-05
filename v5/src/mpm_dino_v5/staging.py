from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn


STAGE_ORDER = ("com", "rotation", "interaction", "deformation", "memory")


def freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def thaw(module: nn.Module) -> nn.Module:
    module.train()
    for parameter in module.parameters():
        parameter.requires_grad_(True)
    return module


def module_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def snapshot(modules: Iterable[tuple[str, nn.Module]]) -> dict[str, str]:
    return {name: module_sha256(module) for name, module in modules}


def assert_unchanged(modules: Iterable[tuple[str, nn.Module]], expected: dict[str, str]) -> None:
    actual = snapshot(modules)
    if actual != expected:
        changed = sorted(name for name in set(actual) | set(expected) if actual.get(name) != expected.get(name))
        raise RuntimeError(f"protected modules changed: {changed}")


@dataclass(frozen=True)
class PromotionDecision:
    qualified: bool
    success: bool
    use_memory: bool
    mean: float


def deformation_promotion(seed_scores: dict[int, float], qualification: float = 0.91098, target: float = 0.7025) -> PromotionDecision:
    required = {42, 123, 456}
    if set(seed_scores) != required:
        raise ValueError(f"promotion requires exactly seeds {sorted(required)}")
    mean = sum(seed_scores.values()) / 3
    qualified = mean < qualification
    success = mean <= target
    return PromotionDecision(qualified, success, qualified and not success, mean)


def choose_rotation(learned_degrees: dict[int, float], identity_degrees: dict[int, float]) -> str:
    required = {42, 123, 456}
    if set(learned_degrees) != required or set(identity_degrees) != required:
        raise ValueError("rotation decision requires seeds 42, 123 and 456")
    learned = sum(learned_degrees.values()) / 3
    identity = sum(identity_degrees.values()) / 3
    return "learned" if learned < identity else "identity"


@torch.no_grad()
def output_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

