import argparse
import sys
import yaml
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gerbil_train.config.model_config import FieldEntry
from gerbil_train.data.tfrecord_dataset import load_field_specs
from gerbil_train.utils.config import load_experiment_config

def build_field_entries(model_cfg_path: str | Path, all_field_specs: list[FieldEntry], include_wide_deep: bool = False) -> None:
    """从pos_map.txt中读取字段配置, 更新模型配置文件中的字段配置

    :param model_cfg_path: Path to the model YAML config file.
    :param all_field_specs: List of field spec objects with ``name``, ``field_index``, ``field_type``, ``dim``.
    :param include_wide_deep: Whether to include ``wide``/``deep`` field flags (for W&D/DeepFM).
    """
    raw_cfg: dict[str, Any] = yaml.safe_load(Path(model_cfg_path).read_text(encoding="utf-8"))
    default_emb_size: int = int(raw_cfg.get("embedding", {}).get("default_emb_size", 16))
    existing: dict[str, Any] = raw_cfg.get("embedding", {}).get("fields") or {}

    if existing:
        raise ValueError("Model config already has field entries")

    if all_field_specs is None:
        raise ValueError("all_field_specs must be provided")

    entries: dict[str, FieldEntry] = {}
    for spec in all_field_specs:
        ex = existing.get(spec.field_name, {})
        kw = dict(
            field_index=spec.field_index,
            field_type=spec.field_type,
            field_name=spec.field_name,
            dim=int(spec.dim),
            concat_type=str(ex.get("concat_type", "direct")),
            emb_size=int(ex.get("emb_size", default_emb_size)),
            enabled=bool(ex.get("enabled", True)),
        )
        if include_wide_deep:
            kw["wide"] = bool(ex.get("wide", True))
            kw["deep"] = bool(ex.get("deep", True))
        entries[spec.field_name] = FieldEntry(**kw)

    field_dicts: list[dict[str, Any]] = []
    for f_name, entry in sorted(entries.items(), key=lambda x: x[1].field_index):
        d: dict[str, Any] = {
            "field_index": entry.field_index,
            "field_type": entry.field_type,
            "dim": entry.dim,
        }
        if entry.field_type == 0:
            d["concat_type"] = entry.concat_type
        d["emb_size"] = entry.emb_size
        d["enabled"] = entry.enabled
        if include_wide_deep:
            d["wide"] = entry.wide
            d["deep"] = entry.deep
        field_dicts.append((f_name, d))
    raw_cfg["embedding"]["fields"] = dict(field_dicts)

    with open(model_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"Config written to {model_cfg_path}")


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG: Path = PROJECT_ROOT / "configs/2-gwen_ml1m_binary/experiment.yaml"

if __name__ == "__main__":
    """Main function for building model config."""
    parser = argparse.ArgumentParser(description="Build model config from pos_map.txt")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Experiment YAML path")
    parser.add_argument("--include-wide-deep", type=bool, default=False, action="store_true", help="Include wide/deep field flags (for W&D/DeepFM)")
    args = parser.parse_args()
    exp_cfg: dict[str, Any] = load_experiment_config(args.config)
    model_cfg = str(PROJECT_ROOT / exp_cfg["experiment"]["model"])
    print(f"model_cfg: {model_cfg}")
    pos_map_text_path = exp_cfg["data"]["paths"]["nn_pos_map_txt"]
    print(f"pos_map_text_path: {pos_map_text_path}")
    all_field_specs: list[FieldEntry] = load_field_specs(pos_map_text_path)
    build_field_entries(model_cfg, all_field_specs, include_wide_deep=args.include_wide_deep)
