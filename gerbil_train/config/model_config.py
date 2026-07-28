from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gerbil_train.exceptions import ConfigError


@dataclass
class FieldEntry:
    """One embedding field in the model config.

    :param field_index:  Feature index matching ``pos_map.txt``
    :param field_type:   0 = continuous (bucketed), 1 = categorical
    :param dim:  Dimension for this field
    :param emb_size:  Embedding dimension for this field
    :param enabled:  Whether to include this field in the model
    """
    field_name: str             # 必须唯一
    field_index: int            # 可以相同，相同时表示词表共享
    field_type: int             # 0表示连续特征，1表示离散特征
    dim: int                    # 特征维度
    concat_type: str = "emb"    # 连续特征拼接方式. direct: 直接concat, emb: 投影后concat
    emb_size: int = -1          # Embedding维度
    enabled: bool = True        # 是否启用
    wide: bool = True           # 是否进入Wide部分（仅W&D/DeepFM模型使用）
    deep: bool = True           # 是否进入Deep部分（仅W&D/DeepFM模型使用）


def load_enabled_field_entries(model_cfg: dict[str, Any]) -> tuple[list[FieldEntry], list[str]]:
    """Load all enabled field entries from the model config."""
    fields = model_cfg.get("embedding", {}).get("fields", {})
    # print(f"[debug] fields: {fields}")

    enabled_field_entries: list[FieldEntry] = []
    disabled_field_names: list[str] = []
    for name, entry in fields.items():
        if entry["enabled"]:
            enabled_field_entries.append(FieldEntry(
                field_index=entry["field_index"],
                field_type=entry["field_type"],
                field_name=name,
                dim=entry["dim"],
                concat_type=entry.get("concat_type", "emb"),
                emb_size=entry["emb_size"],
                enabled=entry["enabled"],
                wide=bool(entry.get("wide", True)),
                deep=bool(entry.get("deep", True)),
            ))
        else:
            print(f"Disabled field {name}")
            disabled_field_names.append(name)
    return enabled_field_entries, disabled_field_names  


@dataclass
class BaseModelConfig:
    target_size: int
    embedding_fields: dict[str, FieldEntry]
    mlp: dict[str, Any] = field(default_factory=dict)
    field_attention: dict[int, Any] = field(default_factory=dict)
    # dict[field_name, dict[field_index, tuple[mean, std]]]
    field_stats: dict[str, dict[int, tuple[float, float]]] = field(default_factory=dict)

    # Shared config: fields used by SLIM, SIM, DIEN, etc.
    slim: dict[str, Any] = field(default_factory=dict)
    behavior_fields: list[str] = field(default_factory=list)
    target_fields: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, model_cfg: dict[str, Any], field_entries: list[FieldEntry]) -> "BaseModelConfig":
        return cls(
            target_size=int(model_cfg.get("target_size", 0)),
            embedding_fields={field.field_name: field for field in field_entries},
            mlp=dict(model_cfg.get("mlp", {})),
            field_attention=dict(model_cfg.get("field_attention", {})),
            field_stats=dict(model_cfg.get("field_stats", {})),
            slim=dict(model_cfg.get("slim", {})),
            behavior_fields=list(model_cfg.get("behavior_fields", [])),
            target_fields=list(model_cfg.get("target_fields", [])),
        )


@dataclass
class AFMModelConfig(BaseModelConfig):
    afm_attention: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, model_cfg: dict[str, Any], field_entries: list[FieldEntry]) -> "AFMModelConfig":
        return cls(
            target_size=int(model_cfg.get("target_size", 0)),
            embedding_fields={field.field_name: field for field in field_entries},
            mlp=dict(model_cfg.get("mlp", {})),
            field_attention=dict(model_cfg.get("field_attention", {})),
            field_stats=dict(model_cfg.get("field_stats", {})),
            afm_attention=dict(model_cfg.get("afm_attention", {})),
        )


@dataclass
class AutoIntModelConfig(BaseModelConfig):
    auto_attention: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, model_cfg: dict[str, Any], field_entries: list[FieldEntry]) -> "AutoIntModelConfig":
        return cls(
            target_size=int(model_cfg.get("target_size", 0)),
            embedding_fields={field.field_name: field for field in field_entries},
            mlp=dict(model_cfg.get("mlp", {})),
            field_attention=dict(model_cfg.get("field_attention", {})),
            field_stats=dict(model_cfg.get("field_stats", {})),
            auto_attention=dict(model_cfg.get("auto_attention", {})),
        )


@dataclass
class DINModelConfig(BaseModelConfig):
    behavior_fields: list[str] = field(default_factory=list)
    target_fields: list[str] = field(default_factory=list)
    softmax_attn: bool = False
    target_merge: str = "mean"
    local_activation_unit: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, model_cfg: dict[str, Any], field_entries: list[FieldEntry]) -> "DINModelConfig":
        return cls(
            target_size=int(model_cfg.get("target_size", 0)),
            embedding_fields={field.field_name: field for field in field_entries},
            behavior_fields=list(model_cfg.get("behavior_fields", [])),
            target_fields=list(model_cfg.get("target_fields", [])),
            softmax_attn=bool(model_cfg.get("softmax_attn", False)),
            target_merge=str(model_cfg.get("target_merge", "mean")),
            local_activation_unit=dict(model_cfg.get("local_activation_unit", {})),
            mlp=dict(model_cfg.get("mlp", {})),
            field_attention=dict(model_cfg.get("field_attention", {})),
            field_stats=dict(model_cfg.get("field_stats", {})),
        )

@dataclass
class DIENModelConfig(DINModelConfig):
    interest_extractor: dict[str, Any] = field(default_factory=dict)
    aux_net: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, model_cfg: dict[str, Any], field_entries: list[FieldEntry]) -> "DIENModelConfig":
        return cls(
            target_size=int(model_cfg.get("target_size", 0)),
            embedding_fields={field.field_name: field for field in field_entries},
            behavior_fields=list(model_cfg.get("behavior_fields", [])),
            target_fields=list(model_cfg.get("target_fields", [])),
            softmax_attn=bool(model_cfg.get("softmax_attn", False)),
            target_merge=str(model_cfg.get("target_merge", "mean")),
            local_activation_unit=dict(model_cfg.get("local_activation_unit", {})),
            interest_extractor=dict(model_cfg.get("interest_extractor", {})),
            aux_net=dict(model_cfg.get("aux_net", {})),
            mlp=dict(model_cfg.get("mlp", {})),
            field_attention=dict(model_cfg.get("field_attention", {})),
            field_stats=dict(model_cfg.get("field_stats", {})),
        )


@dataclass
class DeepFMModelConfig(BaseModelConfig):
    output: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, model_cfg: dict[str, Any], field_entries: list[FieldEntry]) -> "DeepFMModelConfig":
        return cls(
            target_size=int(model_cfg.get("target_size", 0)),
            embedding_fields={field.field_name: field for field in field_entries},
            output=dict(model_cfg.get("output", {})),
            mlp=dict(model_cfg.get("mlp", {})),
            field_attention=dict(model_cfg.get("field_attention", {})),
            field_stats=dict(model_cfg.get("field_stats", {})),
        )


@dataclass
class WideAndDeepModelConfig(BaseModelConfig):
    output: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, model_cfg: dict[str, Any], field_entries: list[FieldEntry]) -> "WideAndDeepModelConfig":
        return cls(
            target_size=int(model_cfg.get("target_size", 0)),
            embedding_fields={field.field_name: field for field in field_entries},
            output=dict(model_cfg.get("output", {})),
            mlp=dict(model_cfg.get("mlp", {})),
            field_attention=dict(model_cfg.get("field_attention", {})),
            field_stats=dict(model_cfg.get("field_stats", {})),
        )


@dataclass
class MFModelConfig(BaseModelConfig):
    mf: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, model_cfg: dict[str, Any], field_entries: list[FieldEntry]) -> "MFModelConfig":
        return cls(
            target_size=int(model_cfg.get("target_size", 0)),
            embedding_fields={field.field_name: field for field in field_entries},
            mf=dict(model_cfg.get("mf", {})),
            mlp=dict(model_cfg.get("mlp", {})),
            field_attention=dict(model_cfg.get("field_attention", {})),
            field_stats=dict(model_cfg.get("field_stats", {})),
        )


@dataclass
class YouTubeDNNModelConfig(BaseModelConfig):
    behavior_fields: list[str] = field(default_factory=list)
    example_age_field: str = ""
    head_bias: bool = False

    @classmethod
    def from_dict(cls, model_cfg: dict[str, Any], field_entries: list[FieldEntry]) -> "YouTubeDNNModelConfig":
        return cls(
            target_size=int(model_cfg.get("target_size", 0)),
            embedding_fields={field.field_name: field for field in field_entries},
            behavior_fields=list(model_cfg.get("behavior_fields", [])),
            example_age_field=str(model_cfg.get("example_age_field", "")),
            head_bias=bool(model_cfg.get("head_bias", False)),
            mlp=dict(model_cfg.get("mlp", {})),
            field_attention=dict(model_cfg.get("field_attention", {})),
            field_stats=dict(model_cfg.get("field_stats", {})),
        )
