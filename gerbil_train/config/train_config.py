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
    """Learning rate scheduler configuration.

    - ``type='warmup_decay'``: step-level linear warmup + exponential decay via ``update_learning_rate()``
    - ``type='none'``: fixed learning rate, no scheduling
    """
    type: str = "none"              # "warmup_decay" | "none"
    warmup_steps: int = 0           # linear warmup steps (0=disabled, only for type=warmup_decay)
    decay_rate: float = 0.0         # exponential decay rate (negative, 0.0=disabled, only for type=warmup_decay)
    learning_rate_min: float = 0.0  # minimum learning rate floor (only for type=warmup_decay)


@dataclass
class TrainCheckpointConfig:
    monitor: str
    mode: str
    path: str | None = None


@dataclass
class TrainEarlyStopConfig:
    enabled: bool = False
    patience: int = 3


@dataclass
class TrainLoggingConfig:
    verbose: bool = False
    plot_path: str | None = None



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
class TrainConfig:
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
    loss: TrainLossConfig = field(default_factory=TrainLossConfig)
    inspector: TrainInspectorConfig = field(default_factory=TrainInspectorConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
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
            loss=TrainLossConfig(**d.get("loss", {})),
            compile=TrainCompileConfig(**d.get("compile")) if isinstance(d.get("compile"), dict) else TrainCompileConfig(enabled=bool(d.get("compile", False))),
            inspector=TrainInspectorConfig(**d.get("inspector", {})),
        )
