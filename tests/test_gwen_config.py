from __future__ import annotations

import unittest

from gerbil_train.config.train_config import (
    TrainCheckpointConfig,
    TrainCompileConfig,
    TrainDataConfig,
    FieldEntry,
    TrainLossConfig,
    GwENModelConfig,
    TrainOptimizerConfig,
    GwENTrainConfig,
)


class GwENFieldEntryTests(unittest.TestCase):
    """Tests for GwENFieldEntry."""

    def test_field_entry_defaults(self) -> None:
        entry = FieldEntry(f_index=1, f_type=1, vocab_size=10, emb_dim=8)
        self.assertEqual(entry.f_index, 1)
        self.assertEqual(entry.f_type, 1)
        self.assertEqual(entry.vocab_size, 10)
        self.assertEqual(entry.emb_dim, 8)
        self.assertTrue(entry.enabled)

    def test_field_entry_disabled(self) -> None:
        entry = FieldEntry(f_index=5, f_type=0, vocab_size=3, emb_dim=4, enabled=False)
        self.assertFalse(entry.enabled)


class GwENModelConfigTests(unittest.TestCase):
    """Tests for GwENModelConfig."""

    def test_from_dict(self) -> None:
        entries = {
            "age": FieldEntry(f_index=2, f_type=1, vocab_size=8, emb_dim=4),
        }
        cfg = GwENModelConfig.from_dict(
            {"target_size": 100, "mlp": {"hidden_dims": [32]}, "field_attention": {"enabled": False}},
            entries,
        )
        self.assertEqual(cfg.target_size, 100)
        self.assertEqual(len(cfg.embedding_fields), 1)
        self.assertEqual(cfg.mlp["hidden_dims"], [32])
        self.assertEqual(cfg.field_attention["enabled"], False)

    def test_from_dict_minimal(self) -> None:
        entries = {}
        cfg = GwENModelConfig.from_dict({}, entries)
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
    """Tests for GwENCheckpointConfig."""

    def test_defaults(self) -> None:
        cfg = TrainCheckpointConfig()
        self.assertIsNone(cfg.path)
        self.assertEqual(cfg.monitor, "hit@1")
        self.assertEqual(cfg.mode, "max")


class GwENTrainConfigFromDictTests(unittest.TestCase):
    """Tests for GwENTrainConfig.from_dict."""

    def test_empty_dict(self) -> None:
        cfg = GwENTrainConfig.from_dict({})
        self.assertEqual(cfg.seed, 42)
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.epochs, 1)

    def test_full_config(self) -> None:
        cfg = GwENTrainConfig.from_dict({
            "seed": 7,
            "device": "cuda",
            "epochs": 10,
            "data": {"batch_size": 256, "num_workers": 2},
            "optimizer": {"lr": 0.01},
            "checkpoint": {"monitor": "hit@10"},
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
        cfg = GwENTrainConfig.from_dict({"compile": True})
        self.assertTrue(cfg.compile.enabled)


if __name__ == "__main__":
    unittest.main()
