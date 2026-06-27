from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from gerbil_train.losses.classification import nce_loss, sampled_softmax_loss
from gerbil_train.metrics.classification import auc


class CrossEntropyLossTests(unittest.TestCase):
    """Reference tests for cross-entropy (used as baseline for losses)."""

    def test_cross_entropy_basic(self) -> None:
        """Standard CE loss produces expected values."""
        logits = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        targets = torch.tensor([0, 1])
        loss = F.cross_entropy(logits, targets)
        self.assertGreater(loss.item(), 0.0)
        self.assertLess(loss.item(), 1.0)

    def test_cross_entropy_perfect_prediction(self) -> None:
        """CE loss is near zero when prediction is nearly perfect."""
        logits = torch.tensor([[100.0, 0.0], [0.0, 100.0]])
        targets = torch.tensor([0, 1])
        loss = F.cross_entropy(logits, targets)
        self.assertAlmostEqual(loss.item(), 0.0, places=4)


class NceLossTests(unittest.TestCase):
    """Tests for the NCE loss implementation."""

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.emb_dim = 8
        self.num_classes = 20
        self.hidden = torch.randn(self.batch_size, self.emb_dim)
        self.class_weight = torch.randn(self.num_classes, self.emb_dim)
        self.targets = torch.randint(0, self.num_classes, (self.batch_size,))

    def test_nce_loss_returns_scalar(self) -> None:
        """NCE loss returns a scalar float tensor."""
        loss = nce_loss(self.hidden, self.class_weight, self.targets, num_sampled=5)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0.0)

    def test_nce_loss_finite(self) -> None:
        """NCE loss is finite (no NaN or Inf)."""
        loss = nce_loss(self.hidden, self.class_weight, self.targets, num_sampled=5)
        self.assertTrue(torch.isfinite(loss).item())

    def test_nce_loss_decreases_with_better_weights(self) -> None:
        """NCE loss is lower when class_weight matches targets."""
        hidden = torch.randn(2, 8)
        targets = torch.tensor([3, 7])
        # Random weights
        random_w = torch.randn(10, 8)
        loss_random = nce_loss(hidden, random_w, targets, num_sampled=5)
        # Perfect weights (hidden copied into target rows)
        perfect_w = torch.randn(10, 8)
        perfect_w[3] = hidden[0]
        perfect_w[7] = hidden[1]
        loss_perfect = nce_loss(hidden, perfect_w, targets, num_sampled=5)
        self.assertLess(loss_perfect.item(), loss_random.item())

    def test_nce_loss_accepts_small_k(self) -> None:
        """Works with num_sampled=1 (minimum)."""
        loss = nce_loss(self.hidden, self.class_weight, self.targets, num_sampled=1)
        self.assertTrue(torch.isfinite(loss).item())

    def test_nce_loss_batch_size_one(self) -> None:
        """Works with single sample."""
        h = self.hidden[:1]
        t = self.targets[:1]
        loss = nce_loss(h, self.class_weight, t, num_sampled=5)
        self.assertTrue(torch.isfinite(loss).item())


class SampledSoftmaxLossTests(unittest.TestCase):
    """Tests for the Sampled Softmax loss implementation."""

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.emb_dim = 8
        self.num_classes = 20
        self.hidden = torch.randn(self.batch_size, self.emb_dim)
        self.class_weight = torch.randn(self.num_classes, self.emb_dim)
        self.targets = torch.randint(0, self.num_classes, (self.batch_size,))

    def test_sampled_softmax_returns_scalar(self) -> None:
        """Sampled softmax returns a scalar float tensor."""
        loss = sampled_softmax_loss(self.hidden, self.class_weight, self.targets, num_sampled=5)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0.0)

    def test_sampled_softmax_finite(self) -> None:
        """Sampled softmax loss is finite."""
        loss = sampled_softmax_loss(self.hidden, self.class_weight, self.targets, num_sampled=5)
        self.assertTrue(torch.isfinite(loss).item())

    def test_sampled_softmax_with_bias(self) -> None:
        """Works when class_bias is provided."""
        bias = torch.randn(self.num_classes)
        loss = sampled_softmax_loss(
            self.hidden, 
            self.class_weight, 
            self.targets,
            num_sampled=5, 
            class_bias=bias,
        )
        self.assertTrue(torch.isfinite(loss).item())

    def test_sampled_softmax_without_bias(self) -> None:
        """Works when class_bias is None."""
        loss = sampled_softmax_loss(
            self.hidden, 
            self.class_weight, 
            self.targets,
            num_sampled=5, 
            class_bias=None,
        )
        self.assertTrue(torch.isfinite(loss).item())

    def test_sampled_softmax_batch_size_one(self) -> None:
        """Works with single sample."""
        h = self.hidden[:1]
        t = self.targets[:1]
        loss = sampled_softmax_loss(h, self.class_weight, t, num_sampled=5)
        self.assertTrue(torch.isfinite(loss).item())

    def test_sampled_softmax_decreases_with_better_weights(self) -> None:
        """Loss is lower when class_weight matches targets."""
        hidden = torch.randn(2, 8)
        targets = torch.tensor([3, 7])
        random_w = torch.randn(10, 8)
        loss_random = sampled_softmax_loss(hidden, random_w, targets, num_sampled=5)
        perfect_w = torch.randn(10, 8)
        perfect_w[3] = hidden[0]
        perfect_w[7] = hidden[1]
        loss_perfect = sampled_softmax_loss(hidden, perfect_w, targets, num_sampled=5)
        self.assertLess(loss_perfect.item(), loss_random.item())


class AUCTests(unittest.TestCase):
    """Tests for the AUC metric."""

    def test_auc_perfect(self) -> None:
        """AUC is 1.0 when all positive ranks are above all negatives."""
        labels = torch.tensor([1, 1, 0, 0])
        preds = torch.tensor([0.9, 0.8, 0.2, 0.1])
        self.assertAlmostEqual(auc(labels, preds), 1.0, places=4)

    def test_auc_mixed(self) -> None:
        """AUC is computed correctly for mixed predictions."""
        labels = torch.tensor([1, 0, 1, 0])
        preds = torch.tensor([0.9, 0.8, 0.2, 0.1])
        # Sorted: 0.9(1), 0.8(0), 0.2(1), 0.1(0) → ranks [4,3,2,1], sum_pos=4+2=6
        # AUC = (6 - 3) / 4 = 0.75
        self.assertAlmostEqual(auc(labels, preds), 0.75, places=4)

    def test_auc_falling(self) -> None:
        """AUC is 0.0 when all positive ranks are below all negatives."""
        labels = torch.tensor([1, 1, 0, 0])
        preds = torch.tensor([0.1, 0.2, 0.9, 0.8])
        self.assertAlmostEqual(auc(labels, preds), 0.0, places=4)

    def test_auc_single_class(self) -> None:
        """AUC defaults to 0.5 when only one class is present."""
        labels = torch.tensor([1, 1, 1])
        preds = torch.tensor([0.9, 0.8, 0.7])
        self.assertEqual(auc(labels, preds), 0.5)


if __name__ == "__main__":
    unittest.main()
