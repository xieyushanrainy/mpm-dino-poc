from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class FeatureTransform:
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, components: int) -> "FeatureTransform":
        mean = x.mean(0, keepdims=True)
        scale = x.std(0, keepdims=True)
        scale[scale < 1e-8] = 1.0
        standardized = (x - mean) / scale
        max_components = min(components, standardized.shape[0] - 1, standardized.shape[1])
        if max_components < 1:
            raise ValueError("PCA requires at least two training objects")
        _, _, vh = np.linalg.svd(standardized, full_matrices=False)
        return cls(mean.astype(np.float32), scale.astype(np.float32), vh[:max_components].astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (((x - self.mean) / self.scale) @ self.components.T).astype(np.float32)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": torch.from_numpy(self.mean), "scale": torch.from_numpy(self.scale), "components": torch.from_numpy(self.components)}


class Probe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kind: str, hidden_dim: int, dropout: float):
        super().__init__()
        if kind == "linear":
            self.network = nn.Linear(input_dim, output_dim)
        elif kind == "mlp":
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim)
            )
        else:
            raise ValueError(f"unknown model kind: {kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
