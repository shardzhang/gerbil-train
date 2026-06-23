from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DataPathsConfig:
    tfrecord_root: str
    nn_pos_map_txt: str
    nn_pos_map_json: str


@dataclass
class DataSplitSubdirsConfig:
    train: str = "train"
    val: str = "val"
    test: str = "test"


@dataclass
class TFRecordDataConfig:
    name: str = ""
    description: str = ""
    format: str = "tfrecord"
    paths: DataPathsConfig = field(default_factory=DataPathsConfig)
    split_subdirs: DataSplitSubdirsConfig = field(default_factory=DataSplitSubdirsConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TFRecordDataConfig":
        return cls(
            name=str(d.get("name", "")),
            description=str(d.get("description", "")),
            format=str(d.get("format", "tfrecord")),
            paths=DataPathsConfig(**d.get("paths", {})),
            split_subdirs=DataSplitSubdirsConfig(**d.get("split_subdirs", {})),
        )