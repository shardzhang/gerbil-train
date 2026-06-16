"""Batch inspection utility for debugging training data."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = ["BatchInspector"]

_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass
class BatchInspector:
    log_every: int = 0
    log_first: int = 3
    log_first_epoch_only: bool = True
    keys: tuple[str, ...] = ()
    max_elems: int = 10
    log: bool = True

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def __call__(self, step: int, batch: dict[str, Any], epoch: int | None = None) -> None:
        if not self._should_log(step, epoch):
            return
        self._inspect(step, batch, epoch)

    def _should_log(self, step: int, epoch: int | None = None) -> bool:
        if self.log_first_epoch_only and epoch is not None and epoch > 0 and step <= self.log_first:
            return False
        return step <= self.log_first or (self.log_every > 0 and step % self.log_every == 0)

    def _inspect(self, step: int, batch: dict[str, Any], epoch: int | None = None) -> None:
        prefix = f"[Epoch {epoch + 1}] " if epoch is not None else ""
        lines = [f"{prefix}Step {step} batch inspection:"]

        keys = self.keys or list(batch.keys())
        for key in keys:
            if key not in batch:
                continue
            val = batch[key]
            if isinstance(val, dict):
                if not val:
                    lines.append(f"  {key}: {{}}")
                    continue
                lines.append(f"  {key}:")
                for k, v in val.items():
                    lines.append(f"    {k}: {self._fmt(v)}")
            else:
                lines.append(f"  {key}: {self._fmt(val)}")

        msg = "\n".join(lines)
        if self.log:
            self._logger.info(msg)
        else:
            print(msg)

    def _fmt(self, v: Any) -> str:
        if isinstance(v, torch.Tensor):
            flat = v.flatten()
            n = flat.numel()
            show = min(n, self.max_elems)
            parts = []
            for x in flat[:show]:
                if x.numel() == 1:
                    if x.dtype.is_floating_point:
                        parts.append(f"{x.item():.4f}")
                    else:
                        parts.append(str(x.item()))
                else:
                    parts.append(str(x))
            suffix = f", ... ({n - show} more)" if n > show else ""
            return f"tensor([{', '.join(parts)}]{suffix}, shape={list(v.shape)}, dtype={v.dtype})"
        return str(v)[:200]
