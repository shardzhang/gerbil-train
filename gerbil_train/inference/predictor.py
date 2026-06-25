"""Unified predictor for offline inference and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from gerbil_train.metrics.classification import auc, average_precision, gauc, map_score, mrr_score

__all__ = ["Predictor"]


class Predictor:
    """Offline inference and evaluation for all model types.

    Usage::

        predictor = Predictor(model, device="cpu")
        predictor.load_checkpoint("checkpoints/.../best_model.pth")
        results = predictor.predict(test_loader)            # → list[dict]
        metrics = predictor.evaluate(test_loader)           # → dict
        predictor.predict_and_eval(test_loader, output)     # → metrics + file
    """

    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.model.eval()


    def load_checkpoint(self, ckpt_path: str | Path) -> None:
        """Load model weights from a saved checkpoint."""
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        else:
            state_dict = state
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Loaded checkpoint from {ckpt_path}")


    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> list[dict[str, Any]]:
        """Run inference and return per-sample results.

        :return: List of dicts with keys ``user_id``, ``score``, ``label``.
        """
        self.model.eval()
        results: list[dict[str, Any]] = []
        for batch in dataloader:
            batch = self._move_batch(batch)
            sigmoids = self.model(batch["feature_bags"])
            labels = batch["targets"].float()
            uids = self._extract_user_ids(batch)
            for i in range(len(labels)):
                results.append({
                    "user_id": int(uids[i]) if uids is not None else -1,
                    "score": float(sigmoids[i]),
                    "label": int(labels[i]),
                })
        return results


    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Run inference and compute ranking metrics.

        :return: Dict with keys ``auc``, ``ap``, ``gauc``, ``map``, ``mrr``.
        """
        results = self.predict(dataloader)
        if not results:
            return {}

        scores = torch.tensor([r["score"] for r in results], dtype=torch.float32)
        labels = torch.tensor([r["label"] for r in results], dtype=torch.float32)
        user_ids = torch.tensor([r["user_id"] for r in results], dtype=torch.long)

        metrics = {
            "auc": round(auc(labels, scores), 4),
            "ap": round(average_precision(labels, scores), 4),
            "gauc": round(gauc(user_ids, labels, scores), 4),
            "map": round(map_score(user_ids, labels, scores, weighted=True), 4),
            "mrr": round(mrr_score(user_ids, labels, scores, weighted=True), 4),
        }
        return metrics


    def predict_and_eval(
        self,
        dataloader: DataLoader,
        output_path: str | Path | None = None,
    ) -> dict[str, float]:
        """Run inference, optionally save results, and return metrics.

        :param output_path: If provided, predictions are written to this file.
        :return: Evaluation metrics dict.
        """
        results = self.predict(dataloader)
        if output_path is not None:
            from .result_writer import write_results
            write_results(results, output_path)
        if not results:
            return {}

        scores = torch.tensor([r["score"] for r in results], dtype=torch.float32)
        labels = torch.tensor([r["label"] for r in results], dtype=torch.float32)
        user_ids = torch.tensor([r["user_id"] for r in results], dtype=torch.long)

        return {
            "auc": round(auc(labels, scores), 4),
            "ap": round(average_precision(labels, scores), 4),
            "gauc": round(gauc(user_ids, labels, scores), 4),
            "map": round(map_score(user_ids, labels, scores, weighted=True), 4),
            "mrr": round(mrr_score(user_ids, labels, scores, weighted=True), 4),
        }


    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        if isinstance(batch, dict):
            return {k: self._move_batch(v) for k, v in batch.items()}
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        return batch


    @staticmethod
    def _extract_user_ids(batch: dict[str, Any]) -> torch.Tensor | None:
        """Extract user IDs from a batch of feature bags."""
        uid_bag = batch.get("feature_bags", {}).get("user_id")
        if uid_bag is None:
            return None
        indices = uid_bag["indices"]
        offsets = uid_bag["offsets"]
        next_offsets = torch.cat([offsets[1:], torch.tensor([len(indices)], device=offsets.device)])
        return indices[next_offsets - 1]
