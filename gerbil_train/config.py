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
    shuffle_buffer: int = 0


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
class GwENInspectorConfig:
    enabled: bool = True
    log_first: int = 3
    log_every: int = 0


@dataclass
class GwENCompileConfig:
    enabled: bool = False
    mode: str = "default"      # "default" | "reduce-overhead" | "max-autotune"


# ── DeepFM ─────────────────────────────────────────────────────────────

@dataclass
class DeepFMFieldConfig:
    """One sparse field for DeepFM."""
    vocab_size: int = 0
    padding_idx: int | None = None


@dataclass
class DeepFMConfig:
    """DeepFM model configuration."""
    dense_input_dim: int = 0
    embedding_dim: int = 16
    sparse_fields: dict[str, DeepFMFieldConfig] = field(default_factory=dict)
    embedding_fields: dict[str, GwENFieldEntry] = field(default_factory=dict)
    deep: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DeepFMConfig":
        sparse: dict[str, DeepFMFieldConfig] = {}
        for name, cfg in d.get("sparse_fields", {}).items():
            if isinstance(cfg, int):
                sparse[name] = DeepFMFieldConfig(vocab_size=cfg)
            elif isinstance(cfg, dict):
                sparse[name] = DeepFMFieldConfig(
                    vocab_size=int(cfg.get("vocab_size", 0)),
                    padding_idx=cfg.get("padding_idx"),
                )
        embedding_fields: dict[str, GwENFieldEntry] = {}
        for name, cfg in d.get("embedding_fields", {}).items():
            embedding_fields[name] = GwENFieldEntry(
                f_index=int(cfg.get("f_index", 0)),
                f_type=int(cfg.get("f_type", 1)),
                vocab_size=int(cfg.get("vocab_size", 0)),
                emb_dim=int(cfg.get("emb_dim", 16)),
                enabled=bool(cfg.get("enabled", True)),
            )
        return cls(
            dense_input_dim=int(d.get("dense_input_dim", 0)),
            embedding_dim=int(d.get("embedding_dim", 16)),
            sparse_fields=sparse,
            embedding_fields=embedding_fields,
            deep=dict(d.get("deep", {})),
            output=dict(d.get("output", {})),
        )

    @property
    def field_names(self) -> list[str]:
        if self.embedding_fields:
            return [n for n, e in self.embedding_fields.items() if e.enabled]
        return list(self.sparse_fields.keys())


@dataclass
class DeepFMTrainConfig:
    """DeepFM training configuration, reusing shared sub-configs."""
    seed: int = 42
    device: str = "cpu"
    epochs: int = 5
    trainer: GwENDataConfig = field(default_factory=lambda: GwENDataConfig(batch_size=256))
    optimizer: GwENOptimizerConfig = field(default_factory=GwENOptimizerConfig)
    scheduler: GwENSchedulerConfig = field(default_factory=GwENSchedulerConfig)
    checkpoint: GwENCheckpointConfig = field(default_factory=GwENCheckpointConfig)
    early_stop: GwENEarlyStopConfig = field(default_factory=GwENEarlyStopConfig)
    logging: GwENLoggingConfig = field(default_factory=GwENLoggingConfig)
    evaluation: GwENEvalConfig = field(default_factory=GwENEvalConfig)
    compile: GwENCompileConfig = field(default_factory=GwENCompileConfig)
    inspector: GwENInspectorConfig = field(default_factory=GwENInspectorConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DeepFMTrainConfig":
        trainer_cfg = d.get("data") or d.get("trainer", {})
        return cls(
            seed=int(d.get("seed", 42)),
            device=str(d.get("device", "cpu")),
            epochs=int(d.get("epochs", 5)),
            trainer=GwENDataConfig(**trainer_cfg),
            optimizer=GwENOptimizerConfig(**d.get("optimizer", {})),
            scheduler=GwENSchedulerConfig(**d.get("scheduler", {})),
            checkpoint=GwENCheckpointConfig(
                path=d.get("checkpoint", {}).get("best_checkpoint_path"),
                monitor=str(d.get("checkpoint", {}).get("monitor", "val_auc")),
                mode=str(d.get("checkpoint", {}).get("mode", "max")),
            ),
            early_stop=GwENEarlyStopConfig(**d.get("early_stop", {})),
            logging=GwENLoggingConfig(**d.get("logging", {})),
            evaluation=GwENEvalConfig(**d.get("evaluation", {})),
            compile=GwENCompileConfig(**d.get("compile")) if isinstance(d.get("compile"), dict) else GwENCompileConfig(enabled=bool(d.get("compile", False))),
            inspector=GwENInspectorConfig(**d.get("inspector", {})),
        )


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
    inspector: GwENInspectorConfig = field(default_factory=GwENInspectorConfig)

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
            inspector=GwENInspectorConfig(**d.get("inspector", {})),
        )


# ── Learning-to-Rank ───────────────────────────────────────────────────

@dataclass
class LTRConfig:
    """Learning-to-rank model configuration."""
    input_dim: int = 136
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    activation: str = "relu"
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LTRConfig":
        return cls(
            input_dim=int(d.get("input_dim", 136)),
            hidden_dims=list(d.get("hidden_dims", [256, 128])),
            activation=str(d.get("activation", "relu")),
            dropout=float(d.get("dropout", 0.1)),
        )


# ── Shared-Bottom Two-Tower ────────────────────────────────────────────

@dataclass
class SBTTConfig:
    """Shared-bottom two-tower model configuration."""
    query_input_dim: int = 32
    item_input_dim: int = 32
    embedding_dim: int = 32
    normalize_embedding: bool = False
    temperature: float = 0.07
    shared_bottom: dict[str, Any] = field(default_factory=dict)
    explicit_tower: dict[str, Any] = field(default_factory=dict)
    implicit_tower: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SBTTConfig":
        return cls(
            query_input_dim=int(d["query_input_dim"]),
            item_input_dim=int(d["item_input_dim"]),
            embedding_dim=int(d.get("embedding_dim", 32)),
            normalize_embedding=bool(d.get("normalize_embedding", False)),
            temperature=float(d.get("temperature", 0.07)),
            shared_bottom=dict(d.get("shared_bottom", {})),
            explicit_tower=dict(d.get("explicit_tower", {})),
            implicit_tower=dict(d.get("implicit_tower", {})),
        )
