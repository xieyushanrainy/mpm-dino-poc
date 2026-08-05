from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SPLIT_SHA256 = "2f4add1aa6f6f68b965d031cf12bf60f981a10f628a09bdc7f75a574cd42949b"
SEEDS = (42, 123, 456)


@dataclass(frozen=True)
class ModelConfig:
    frames: int = 59
    hidden_dim: int = 128
    interaction_dim: int = 128
    blocks: int = 4
    heads: int = 4
    dropout: float = 0.1

    def validate(self) -> None:
        if self.frames != 59:
            raise ValueError("V5 is fixed to the 59-frame V4.1 target horizon")
        if min(self.hidden_dim, self.interaction_dim, self.blocks, self.heads) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")


@dataclass(frozen=True)
class MemoryConfig:
    enabled: bool = False
    top_k: int = 3
    tokens_per_source: int = 32
    phases: tuple[int, ...] = (2, 3, 4)
    residual_bound: float = 0.05
    gate_logit: float = -4.0
    permitted_uids: int = 20

    def validate(self) -> None:
        if self.top_k != 3 or self.tokens_per_source != 32:
            raise ValueError("V5 memory is fixed to top-3 and 32 tokens/source")
        if self.phases != (2, 3, 4):
            raise ValueError("memory may contain only onset/compression/peak phases")
        if self.permitted_uids != 20:
            raise ValueError("memory scope is fixed to 20 training soft UIDs")
        if self.residual_bound <= 0:
            raise ValueError("residual_bound must be positive")
        if self.gate_logit >= 0:
            raise ValueError("memory gate must initialize conservatively below zero")


@dataclass(frozen=True)
class TrainingConfig:
    seeds: tuple[int, ...] = SEEDS
    base_qualification: float = 0.91098
    final_target: float = 0.7025
    split_sha256: str = SPLIT_SHA256
    allow_v4_weights: bool = False

    def validate(self) -> None:
        if self.seeds != SEEDS:
            raise ValueError(f"V5 promotion seeds are fixed to {SEEDS}")
        if self.allow_v4_weights:
            raise ValueError("V5 forbids V4 neural-weight initialization")
        if self.split_sha256 != SPLIT_SHA256:
            raise ValueError("unexpected split manifest hash")
        if not self.final_target < self.base_qualification:
            raise ValueError("final target must be stricter than base qualification")


@dataclass(frozen=True)
class V5Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    schema_version: str = "v5_causal_deformation_v1"

    def validate(self) -> None:
        self.model.validate()
        self.memory.validate()
        self.training.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "V5Config":
        memory = dict(values.get("memory", {}))
        training = dict(values.get("training", {}))
        if "phases" in memory:
            memory["phases"] = tuple(memory["phases"])
        if "seeds" in training:
            training["seeds"] = tuple(training["seeds"])
        config = cls(
            model=ModelConfig(**values.get("model", {})),
            memory=MemoryConfig(**memory),
            training=TrainingConfig(**training),
            schema_version=values.get("schema_version", "v5_causal_deformation_v1"),
        )
        config.validate()
        return config

    def reject_checkpoint(self, checkpoint: str | Path | None) -> None:
        if checkpoint is not None:
            raise ValueError("V5 components must start from random weights; checkpoint initialization is forbidden")
