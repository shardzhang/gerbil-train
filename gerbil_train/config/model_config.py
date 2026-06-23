from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FieldEntry:
    """One embedding field in the model config.

    :param field_index:  Feature index matching ``nn_pos_map.txt``
    :param field_type:   0 = continuous (bucketed), 1 = categorical
    :param dim:  Dimension for this field
    :param emb_size:  Embedding dimension for this field
    :param enabled:  Whether to include this field in the model
    """
    field_name: str     # 必须唯一
    field_index: int    # 可以相同，相同时表示词表共享
    field_type: int     # 0表示连续特征，1表示离散特征
    dim: int            # 特征维度
    emb_size: int = -1
    enabled: bool = True


def load_enabled_field_entries(model_cfg: dict[str, Any]) -> tuple[list[FieldEntry], list[str]]:
    """Load all enabled field entries from the model config.

    Supports both old format (``embedding_fields``) and new format (``embedding.fields``).
    """
    fields = model_cfg.get("embedding_fields") or model_cfg.get("embedding", {}).get("fields", {})
    enabled_field_entries: list[FieldEntry] = []
    disabled_field_names: list[str] = []
    for field_name, field_entry in fields.items():
        if field_entry["enabled"]:
            enabled_field_entries.append(FieldEntry(
                field_index=field_entry["field_index"],
                field_type=field_entry["field_type"],
                field_name=field_name,
                dim=field_entry["dim"],
                emb_size=field_entry["emb_size"],
                enabled=field_entry["enabled"],
            ))
        else:
            print(f"Disabled field {field_name}")
            disabled_field_names.append(field_name)
    return enabled_field_entries, disabled_field_names  


@dataclass
class GwENModelConfig:
    target_size: int
    # dict[field_name, FieldEntry]
    embedding_fields: dict[str, FieldEntry]
    mlp: dict[str, Any] = field(default_factory=dict)
    # dict[field_index, Any]
    field_attention: dict[int, Any] = field(default_factory=dict)
    # dict[field_name, dict[field_index, tuple[mean, std]]]
    field_stats: dict[str, dict[int, tuple[float, float]]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, model_cfg: dict[str, Any], field_entries: list[FieldEntry]) -> "GwENModelConfig":
        return cls(
            target_size=int(model_cfg.get("target_size", 0)),
            embedding_fields={field.field_name: field for field in field_entries},
            mlp=dict(model_cfg.get("mlp", {})),
            field_attention=dict(model_cfg.get("field_attention", {})),
            field_stats=dict(model_cfg.get("field_stats", {})),
        )
