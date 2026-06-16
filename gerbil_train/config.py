from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GwENFieldEntry:
    f_index: int
    f_type: int
    vocab_size: int
    emb_dim: int
    enabled: bool = True


@dataclass
class GwENModelConfig:
    target_size: int
    embedding_fields: dict[str, GwENFieldEntry]
    mlp: dict[str, Any] = field(default_factory=dict)
    attention: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any], field_entries: dict[str, GwENFieldEntry]) -> "GwENModelConfig":
        return cls(
            target_size=int(d.get("target_size", 0)),
            embedding_fields=field_entries,
            mlp=dict(d.get("mlp", {})),
            attention=dict(d.get("attention", {})),
        )


@dataclass
class GwENDataConfig:
    batch_size: int = 512
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    prefetch_factor: int = 2


@dataclass
class GwENOptimizerConfig:
    type: str = "adam"
    lr: float = 0.001
    weight_decay: float = 0.0


@dataclass
class GwENSchedulerConfig:
    enabled: bool = False
    mode: str = "max"
    factor: float = 0.5
    patience: int = 1


@dataclass
class GwENCheckpointConfig:
    path: str | None = None
    monitor: str = "hit@1"
    mode: str = "max"


@dataclass
class GwENEarlyStopConfig:
    enabled: bool = False
    patience: int = 3


@dataclass
class GwENLoggingConfig:
    plot_path: str | None = None


@dataclass
class GwENEvalConfig:
    topk: int = 10


@dataclass
class GwENLossConfig:
    type: str = "ce"           # "ce" (cross-entropy) or "nce" (sampled softmax)
    num_sampled: int = 100     # negative samples per batch (only for nce)


@dataclass
class GwENCompileConfig:
    enabled: bool = False
    mode: str = "default"      # "default" | "reduce-overhead" | "max-autotune"


@dataclass
class GwENTrainConfig:
    seed: int = 42
    device: str = "cpu"
    epochs: int = 1
    compile: GwENCompileConfig = field(default_factory=GwENCompileConfig)
    data: GwENDataConfig = field(default_factory=GwENDataConfig)
    optimizer: GwENOptimizerConfig = field(default_factory=GwENOptimizerConfig)
    scheduler: GwENSchedulerConfig = field(default_factory=GwENSchedulerConfig)
    checkpoint: GwENCheckpointConfig = field(default_factory=GwENCheckpointConfig)
    early_stop: GwENEarlyStopConfig = field(default_factory=GwENEarlyStopConfig)
    logging: GwENLoggingConfig = field(default_factory=GwENLoggingConfig)
    evaluation: GwENEvalConfig = field(default_factory=GwENEvalConfig)
    loss: GwENLossConfig = field(default_factory=GwENLossConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GwENTrainConfig":
        return cls(
            seed=int(d.get("seed", 42)),
            device=str(d.get("device", "cpu")),
            epochs=int(d.get("epochs", 1)),
            data=GwENDataConfig(**d.get("data", {})),
            optimizer=GwENOptimizerConfig(**d.get("optimizer", {})),
            scheduler=GwENSchedulerConfig(**d.get("scheduler", {})),
            checkpoint=GwENCheckpointConfig(**d.get("checkpoint", {})),
            early_stop=GwENEarlyStopConfig(**d.get("early_stop", {})),
            logging=GwENLoggingConfig(**d.get("logging", {})),
            evaluation=GwENEvalConfig(**d.get("evaluation", {})),
            loss=GwENLossConfig(**d.get("loss", {})),
            compile=GwENCompileConfig(**d.get("compile")) if isinstance(d.get("compile"), dict) else GwENCompileConfig(enabled=bool(d.get("compile", False))),
        )
