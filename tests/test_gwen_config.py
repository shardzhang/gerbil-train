from __future__ import annotations

import unittest

from gerbil_train.config.model_config import FieldEntry, BaseModelConfig
from gerbil_train.config.train_config import (
    TrainCheckpointConfig,
    TrainCompileConfig,
    TrainDataConfig,
    TrainLossConfig,
    TrainOptimizerConfig,
    TrainConfig,
)


class GwENFieldEntryTests(unittest.TestCase):
    """Tests for FieldEntry."""

    def test_field_entry_defaults(self) -> None:
        entry = FieldEntry(field_name="test", field_index=1, field_type=1, dim=10, emb_size=8)
        self.assertEqual(entry.field_index, 1)
        self.assertEqual(entry.field_type, 1)
        self.assertEqual(entry.dim, 10)
        self.assertEqual(entry.emb_size, 8)
        self.assertTrue(entry.enabled)

    def test_field_entry_disabled(self) -> None:
        entry = FieldEntry(field_name="test", field_index=5, field_type=0, dim=3, emb_size=4, enabled=False)
        self.assertFalse(entry.enabled)


class BaseModelConfigTests(unittest.TestCase):
    """Tests for BaseModelConfig."""

    def test_from_dict(self) -> None:
        entries = [
            FieldEntry(field_name="age", field_index=2, field_type=1, dim=8, emb_size=4),
        ]
        cfg = BaseModelConfig.from_dict(
            {"target_size": 100, "mlp": {"hidden_dims": [32]}, "field_attention": {"enabled": False}},
            entries,
        )
        self.assertEqual(cfg.target_size, 100)
        self.assertEqual(len(cfg.embedding_fields), 1)
        self.assertEqual(cfg.mlp["hidden_dims"], [32])
        self.assertEqual(cfg.field_attention["enabled"], False)

    def test_from_dict_minimal(self) -> None:
        entries: list = []
        cfg = BaseModelConfig.from_dict({}, entries)
        self.assertEqual(cfg.target_size, 0)
        self.assertEqual(cfg.embedding_fields, {})


class GwENDataConfigTests(unittest.TestCase):
    """Tests for GwENDataConfig."""

    def test_defaults(self) -> None:
        cfg = TrainDataConfig()
        self.assertEqual(cfg.batch_size, 512)
        self.assertEqual(cfg.num_workers, 0)
        self.assertFalse(cfg.pin_memory)

    def test_kwargs(self) -> None:
        cfg = TrainDataConfig(**{"batch_size": 128, "num_workers": 4, "prefetch_factor": 4})
        self.assertEqual(cfg.batch_size, 128)
        self.assertEqual(cfg.num_workers, 4)
        self.assertEqual(cfg.prefetch_factor, 4)


class GwENOptimizerConfigTests(unittest.TestCase):
    """Tests for GwENOptimizerConfig."""

    def test_defaults(self) -> None:
        cfg = TrainOptimizerConfig()
        self.assertEqual(cfg.type, "adam")
        self.assertEqual(cfg.lr, 0.001)

    def test_kwargs(self) -> None:
        cfg = TrainOptimizerConfig(**{"lr": 0.01, "weight_decay": 0.1})
        self.assertEqual(cfg.lr, 0.01)
        self.assertEqual(cfg.weight_decay, 0.1)


class GwENLossConfigTests(unittest.TestCase):
    """Tests for GwENLossConfig."""

    def test_default_ce(self) -> None:
        cfg = TrainLossConfig()
        self.assertEqual(cfg.type, "ce")

    def test_nce(self) -> None:
        cfg = TrainLossConfig(**{"type": "nce", "num_sampled": 50})
        self.assertEqual(cfg.type, "nce")
        self.assertEqual(cfg.num_sampled, 50)


class GwENCompileConfigTests(unittest.TestCase):
    """Tests for GwENCompileConfig."""

    def test_default_disabled(self) -> None:
        cfg = TrainCompileConfig()
        self.assertFalse(cfg.enabled)

    def test_enabled(self) -> None:
        cfg = TrainCompileConfig(**{"enabled": True, "mode": "reduce-overhead"})
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.mode, "reduce-overhead")


class GwENCheckpointConfigTests(unittest.TestCase):
    """Tests for TrainCheckpointConfig."""

    def test_defaults(self) -> None:
        cfg = TrainCheckpointConfig(monitor="hit@1", mode="max")
        self.assertIsNone(cfg.path)
        self.assertEqual(cfg.monitor, "hit@1")
        self.assertEqual(cfg.mode, "max")


class TrainConfigFromDictTests(unittest.TestCase):
    """Tests for TrainConfig.from_dict."""

    def test_empty_dict(self) -> None:
        cfg = TrainConfig.from_dict({
            "checkpoint": {"monitor": "val_loss", "mode": "min"},
        })
        self.assertEqual(cfg.seed, 42)
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.epochs, 1)

    def test_full_config(self) -> None:
        cfg = TrainConfig.from_dict({
            "seed": 7,
            "device": "cuda",
            "epochs": 10,
            "data": {"batch_size": 256, "num_workers": 2},
            "optimizer": {"lr": 0.01},
            "checkpoint": {"monitor": "hit@10", "mode": "max"},
            "loss": {"type": "nce", "num_sampled": 50},
            "compile": {"enabled": True},
        })
        self.assertEqual(cfg.seed, 7)
        self.assertEqual(cfg.device, "cuda")
        self.assertEqual(cfg.epochs, 10)
        self.assertEqual(cfg.data.batch_size, 256)
        self.assertEqual(cfg.optimizer.lr, 0.01)
        self.assertEqual(cfg.checkpoint.monitor, "hit@10")
        self.assertEqual(cfg.loss.type, "nce")
        self.assertTrue(cfg.compile.enabled)

    def test_compile_bool(self) -> None:
        """from_dict handles compile: true (bool) gracefully."""
        cfg = TrainConfig.from_dict({"compile": True, "checkpoint": {"monitor": "val_loss", "mode": "min"}})
        self.assertTrue(cfg.compile.enabled)


if __name__ == "__main__":
    unittest.main()
