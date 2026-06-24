from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainDataConfig:
    batch_size: int = 512
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    prefetch_factor: int = 2
    shuffle_buffer: int = 0


@dataclass
class TrainOptimizerConfig:
    type: str = "adam"
    lr: float = 0.001
    weight_decay: float = 0.0


@dataclass
class TrainSchedulerConfig:
    enabled: bool = False
    mode: str = "max"
    factor: float = 0.5
    patience: int = 1


@dataclass
class TrainCheckpointConfig:
    path: str | None = None
    monitor: str = "hit@1"
    mode: str = "max"


@dataclass
class TrainEarlyStopConfig:
    enabled: bool = False
    patience: int = 3


@dataclass
class TrainLoggingConfig:
    verbose: bool = False
    plot_path: str | None = None


@dataclass
class TrainEvalConfig:
    topk: int = 10


@dataclass
class TrainLossConfig:
    type: str = "ce"           # "ce" (cross-entropy) or "nce" (sampled softmax)
    num_sampled: int = 100     # negative samples per batch (only for nce)


@dataclass
class TrainInspectorConfig:
    enabled: bool = True
    log_first: int = 3
    log_every: int = 0


@dataclass
class TrainCompileConfig:
    enabled: bool = False
    mode: str = "default"      # "default" | "reduce-overhead" | "max-autotune"



@dataclass
class GwENTrainConfig:
    seed: int = 42
    device: str = "cpu"
    epochs: int = 1
    compile: TrainCompileConfig = field(default_factory=TrainCompileConfig)
    data: TrainDataConfig = field(default_factory=TrainDataConfig)
    optimizer: TrainOptimizerConfig = field(default_factory=TrainOptimizerConfig)
    scheduler: TrainSchedulerConfig = field(default_factory=TrainSchedulerConfig)
    checkpoint: TrainCheckpointConfig = field(default_factory=TrainCheckpointConfig)
    early_stop: TrainEarlyStopConfig = field(default_factory=TrainEarlyStopConfig)
    logging: TrainLoggingConfig = field(default_factory=TrainLoggingConfig)
    evaluation: TrainEvalConfig = field(default_factory=TrainEvalConfig)
    loss: TrainLossConfig = field(default_factory=TrainLossConfig)
    inspector: TrainInspectorConfig = field(default_factory=TrainInspectorConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GwENTrainConfig":
        return cls(
            seed=int(d.get("seed", 42)),
            device=str(d.get("device", "cpu")),
            epochs=int(d.get("epochs", 1)),
            data=TrainDataConfig(**d.get("data", {})),
            optimizer=TrainOptimizerConfig(**d.get("optimizer", {})),
            scheduler=TrainSchedulerConfig(**d.get("scheduler", {})),
            checkpoint=TrainCheckpointConfig(**d.get("checkpoint", {})),
            early_stop=TrainEarlyStopConfig(**d.get("early_stop", {})),
            logging=TrainLoggingConfig(**d.get("logging", {})),
            evaluation=TrainEvalConfig(**d.get("evaluation", {})),
            loss=TrainLossConfig(**d.get("loss", {})),
            compile=TrainCompileConfig(**d.get("compile")) if isinstance(d.get("compile"), dict) else TrainCompileConfig(enabled=bool(d.get("compile", False))),
            inspector=TrainInspectorConfig(**d.get("inspector", {})),
        )
