import sys
import yaml
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gerbil_train.config.model_config import FieldEntry
from gerbil_train.data.tfrecord_dataset import load_field_specs
from gerbil_train.utils.config import parse_args
from gerbil_train.utils.config import load_experiment_config

def build_field_entries(model_cfg_path: str | Path, all_field_specs: list[FieldEntry]) -> None:
    """从pos_map.txt中读取字段配置，更新模型配置文件中的字段配置。

    :param model_cfg_path: Path to the model YAML config file.
    :param all_field_specs: List of field spec objects with ``name``, ``field_index``, ``field_type``, ``dim``.
    :param extra_keys: Optional dict of extra keys to write to the config root (e.g. ``target_size``).
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
        entries[spec.field_name] = FieldEntry(
            field_index=spec.field_index, 
            field_type=spec.field_type, 
            field_name=spec.field_name,
            dim=int(spec.dim),
            concat_type=str(ex.get("concat_type", "direct")),
            emb_size=int(ex.get("emb_size", default_emb_size)),
            enabled=bool(ex.get("enabled", True)),
        )
    raw_cfg["embedding"]["fields"] = {
        f_name: {
            "field_index": entry.field_index,
            "field_type": entry.field_type,
            "dim": entry.dim,
            **({"concat_type": entry.concat_type} if entry.field_type == 0 else {}),
            "emb_size": entry.emb_size,
            "enabled": entry.enabled,
        }
        for f_name, entry in sorted(entries.items(), key=lambda x: x[1].field_index)
    }

    with open(model_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"Config written to {model_cfg_path}")


PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH: Path = PROJECT_ROOT / "configs/2-gwen_ml1m_binary/experiment.yaml"

if __name__ == "__main__":
    """Main function for building model config."""
    args = parse_args(CONFIG_PATH)
    exp_cfg: dict[str, Any] = load_experiment_config(args.config)
    model_cfg = str(PROJECT_ROOT / exp_cfg["experiment"]["model"])
    print(f"model_cfg: {model_cfg}")
    pos_map_text_path = exp_cfg["data"]["paths"]["nn_pos_map_txt"]
    print(f"pos_map_text_path: {pos_map_text_path}")
    all_field_specs: list[FieldEntry] = load_field_specs(pos_map_text_path)
    build_field_entries(model_cfg, all_field_specs)
