"""CLI smoke tests: model instantiation and forward pass for each CLI entry point."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from gerbil_train.config.model_config import (
    BaseModelConfig, DINModelConfig, DIENModelConfig, DeepFMModelConfig,
    WideAndDeepModelConfig, YouTubeDNNModelConfig, FieldEntry,
)
from gerbil_train.config.train_config import TrainConfig
from gerbil_train.models.gwen import GwENBinaryModel, GwENMulticlassModel
from gerbil_train.models.fm import FM
from gerbil_train.models.deepfm import DeepFM
from gerbil_train.models.wide_and_deep import WideAndDeep
from gerbil_train.models.din import DIN
from gerbil_train.models.dien import DIEN
from gerbil_train.models.youtube_dnn import YouTubeDNN
from gerbil_train.models.ftrl import FTRLModel
from gerbil_train.trainer.binary_trainer import BinaryClassificationTrainer
from gerbil_train.trainer.multi_trainer import MultiClassClassificationTrainer
from gerbil_train.trainer.dien_trainer import DIENTrainer
from gerbil_train.trainer.ftrl_trainer import FTRLTrainer


class CliModelInstantiationTests(unittest.TestCase):
    """Smoke tests: each CLI's model can be instantiated and runs forward."""

    def _make_field(self, name: str, idx: int, dim: int = 10, **kw) -> FieldEntry:
        return FieldEntry(field_name=name, field_index=idx, field_type=1, dim=dim, emb_size=4, **kw)

    def _make_config(self) -> TrainConfig:
        return TrainConfig.from_dict({
            "seed": 42, "device": "cpu", "epochs": 1,
            "data": {"batch_size": 2},
            "checkpoint": {"monitor": "val_gauc", "mode": "max"},
            "optimizer": {"lr": 0.001},
        })

    def _make_bag(self, batch: int = 2, seq_len: int = 1) -> dict:
        total = batch * seq_len
        return {"indices": torch.arange(total, dtype=torch.long),
                "offsets": torch.arange(0, total + seq_len, seq_len, dtype=torch.long)[:batch],
                "weights": torch.ones(total, dtype=torch.float32)}

    def _make_feature_bags(self, fields: list[str], batch: int = 2) -> dict:
        return {f: self._make_bag(batch) for f in fields}

    def _make_seq_bags(self, fields: list[str], batch: int = 2, seq_len: int = 3) -> dict:
        return {f: self._make_bag(batch, seq_len) for f in fields}

    def test_gwen_binary_smoke(self) -> None:
        fields = {"uid": self._make_field("uid", 1)}
        cfg = BaseModelConfig(target_size=0, embedding_fields=fields, mlp={"hidden_dims": [4]})
        model = GwENBinaryModel(cfg)
        out = model(self._make_feature_bags(["uid"]))
        self.assertEqual(out.shape, (2,))

    def test_gwen_multiclass_smoke(self) -> None:
        fields = {"uid": self._make_field("uid", 1)}
        cfg = BaseModelConfig(target_size=10, embedding_fields=fields, mlp={"hidden_dims": [4]})
        model = GwENMulticlassModel(cfg)
        out = model({"uid": self._make_bag()})
        self.assertEqual(out.shape, (2, 10))

    def test_fm_smoke(self) -> None:
        fields = {"uid": self._make_field("uid", 1), "iid": self._make_field("iid", 2)}
        cfg = BaseModelConfig(target_size=0, embedding_fields=fields)
        model = FM(cfg)
        out = model(self._make_feature_bags(["uid", "iid"]))
        self.assertEqual(out.shape, (2,))

    def test_deepfm_smoke(self) -> None:
        fields = {"uid": self._make_field("uid", 1), "iid": self._make_field("iid", 2)}
        cfg = DeepFMModelConfig(target_size=0, embedding_fields=fields, mlp={"hidden_dims": [4]})
        model = DeepFM(cfg)
        out = model(self._make_feature_bags(["uid", "iid"]))
        self.assertEqual(out.shape, (2,))

    def test_wide_and_deep_smoke(self) -> None:
        fields = {"uid": self._make_field("uid", 1), "iid": self._make_field("iid", 2)}
        cfg = WideAndDeepModelConfig(target_size=0, embedding_fields=fields, mlp={"hidden_dims": [4]})
        model = WideAndDeep(cfg)
        out = model(self._make_feature_bags(["uid", "iid"]))
        self.assertEqual(out.shape, (2,))

    def test_din_smoke(self) -> None:
        fields = {"uid": self._make_field("uid", 1), "target": self._make_field("target", 2),
                  "history": self._make_field("history", 3, dim=20)}
        cfg = DINModelConfig(target_size=5, embedding_fields=fields,
                             behavior_fields=["history"], target_fields=["target"],
                             mlp={"hidden_dims": [4]})
        model = DIN(cfg)
        out = model(self._make_feature_bags(["uid", "target", "history"]))
        self.assertEqual(out.shape, (2,))

    def test_dien_smoke(self) -> None:
        fields = {"uid": self._make_field("uid", 1), "target": self._make_field("target", 2),
                  "history": self._make_field("history", 3, dim=30)}
        cfg = DIENModelConfig(target_size=5, embedding_fields=fields,
                              behavior_fields=["history"], target_fields=["target"],
                              mlp={"hidden_dims": [4]},
                              interest_extractor={"hidden_size": 4})
        model = DIEN(cfg)
        bags = self._make_feature_bags(["uid", "target"])
        bags.update(self._make_seq_bags(["history"], seq_len=3))
        out = model(bags)
        self.assertEqual(out.shape, (2,))

    def test_youtube_dnn_smoke(self) -> None:
        fields = {"uid": self._make_field("uid", 1), "history": self._make_field("history", 2, dim=20)}
        cfg = YouTubeDNNModelConfig(target_size=10, embedding_fields=fields,
                                    behavior_fields=["history"], mlp={"hidden_dims": [4]})
        model = YouTubeDNN(cfg)
        out = model(self._make_feature_bags(["uid", "history"]))
        self.assertEqual(out.shape, (2, 10))

    def test_ftrl_smoke(self) -> None:
        fields = {"uid": self._make_field("uid", 1), "iid": self._make_field("iid", 2)}
        cfg = BaseModelConfig(target_size=0, embedding_fields=fields)
        model = FTRLModel(cfg)
        out = model(self._make_feature_bags(["uid", "iid"]))
        self.assertEqual(out.shape, (2,))

    def test_model_config_from_yaml(self) -> None:
        """Verify model config can be loaded from a minimal YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "model.yaml"
            yaml_path.write_text(yaml.safe_dump({
                "task": "binary",
                "embedding": {
                    "default_emb_size": 4,
                    "fields": {
                        "uid": {"field_index": 1, "field_type": 1, "dim": 10, "emb_size": 4, "enabled": True},
                    },
                },
                "mlp": {"hidden_dims": [4]},
            }))
            # Create a minimal valid pos_map.json
            json_path = Path(tmpdir) / "pos_map.json"
            import json
            json_path.write_text(json.dumps({"target_size": 5}))
            from gerbil_train.utils.training import build_model_config
            exp_cfg = {"model": yaml.safe_load(yaml_path.read_text()),
                       "data": {"paths": {"nn_pos_map_json": str(json_path)}}}
            cfg = build_model_config(exp_cfg, BaseModelConfig)
            self.assertIn("uid", cfg.embedding_fields)


class TrainerInitSmokeTests(unittest.TestCase):
    """Smoke tests: each trainer can be initialized with a model and config."""

    def _make_model_and_config(self) -> tuple:
        fields = {"uid": FieldEntry(field_name="uid", field_index=1, field_type=1, dim=10, emb_size=4)}
        cfg = BaseModelConfig(target_size=2, embedding_fields=fields, mlp={"hidden_dims": [4]})
        return GwENBinaryModel(cfg), cfg

    def test_binary_trainer_init(self) -> None:
        model, cfg = self._make_model_and_config()
        t = BinaryClassificationTrainer(model, self._minimal_train_cfg())
        self.assertEqual(t.model_name, "BinaryModel")

    def test_multi_trainer_init(self) -> None:
        model = GwENMulticlassModel(self._minimal_model_cfg())
        t = MultiClassClassificationTrainer(model, self._minimal_train_cfg())
        self.assertEqual(t.model_name, "MultiClassModel")

    def test_dien_trainer_init(self) -> None:
        fields = {"uid": FieldEntry(field_name="uid", field_index=1, field_type=1, dim=10, emb_size=4),
                  "target": FieldEntry(field_name="target", field_index=2, field_type=1, dim=5, emb_size=4),
                  "history": FieldEntry(field_name="history", field_index=3, field_type=1, dim=20, emb_size=4)}
        cfg = DIENModelConfig(target_size=5, embedding_fields=fields,
                              behavior_fields=["history"], target_fields=["target"],
                              mlp={"hidden_dims": [4]}, interest_extractor={"hidden_size": 4})
        model = DIEN(cfg)
        t = DIENTrainer(model, self._minimal_train_cfg())
        self.assertEqual(t.model_name, "DIEN")

    def test_ftrl_trainer_init(self) -> None:
        model = FTRLModel(self._minimal_model_cfg())
        t = FTRLTrainer(model, self._minimal_train_cfg())
        self.assertEqual(t.model_name, "FTRL")

    def _minimal_model_cfg(self) -> BaseModelConfig:
        fields = {"uid": FieldEntry(field_name="uid", field_index=1, field_type=1, dim=10, emb_size=4)}
        return BaseModelConfig(target_size=2, embedding_fields=fields, mlp={"hidden_dims": [4]})

    def _minimal_train_cfg(self) -> TrainConfig:
        return TrainConfig.from_dict({
            "seed": 42, "device": "cpu", "epochs": 1,
            "data": {"batch_size": 2},
            "checkpoint": {"monitor": "val_gauc", "mode": "max"},
            "optimizer": {"lr": 0.001},
            "logging": {"verbose": False},
        })


if __name__ == "__main__":
    unittest.main()
