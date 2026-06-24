"""Neural network construction helpers."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


__all__ = ["print_model_structure", "count_parameters"]


def _format_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def _strip_compile_prefix(name: str) -> str:
    return name[len("_orig_mod."):] if name.startswith("_orig_mod.") else name


def print_model_structure(model: nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    nodes: list[tuple[str, str, int, str]] = []

    def _walk(module: nn.Module, name: str = "", depth: int = 0, is_last: bool = True) -> None:
        display = getattr(module, "field_name", name)
        prefix = "    " * depth + ("└── " if is_last else "├── ")
        label = f"{display}  {type(module).__name__}" if display else type(module).__name__
        params = sum(p.numel() for p in module.parameters(recurse=False))
        shapes = ", ".join(str(tuple(p.shape)) for p in module.parameters(recurse=False))
        nodes.append((prefix, label, params, shapes))
        children = list(module.named_children())
        for i, (child_name, child) in enumerate(children):
            _walk(child, child_name, depth + 1, i == len(children) - 1)

    for child_name, child in model.named_children():
        _walk(child, child_name)

    max_left = max(len(p) + len(l) for p, l, _, _ in nodes)
    rendered: list[str] = []
    for prefix, label, params, shapes in nodes:
        left = f"{prefix}{label}"
        padding = max_left - len(left)
        pct = f"({params / total * 100:.1f}%)" if total and params else ""
        tail = f"  {shapes:<30s} {params:>6,}  {pct:<10s}" if (shapes or params) else ""
        rendered.append(f"{left}{' ' * padding}{tail}")

    width = max(len(l) for l in rendered) + 2
    title = f"Model Structure: {type(model).__name__} ({total:,} params, {_format_size(param_bytes)})"
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")
    for line in rendered:
        print(f"  {line}")
    print()


def count_parameters(model: nn.Module) -> None:
    rows: list[tuple[str, str, str, int, float]] = []
    total_all = sum(p.numel() for p in model.parameters()) or 1
    for module_name, module in model.named_modules():
        if module_name == "":
            continue
        count = sum(p.numel() for p in module.parameters(recurse=False))
        if count == 0:
            continue
        name = getattr(module, "field_name", _strip_compile_prefix(module_name))
        shapes = ", ".join(str(tuple(p.shape)) for p in module.parameters(recurse=False))
        rows.append((name, type(module).__name__, shapes, count, count / total_all * 100))

    rows.sort(key=lambda r: r[3], reverse=True)
    fmt_name = max(len(r[0]) for r in rows)
    col_w = {"name": max(5, fmt_name), "type": 18, "shape": 30, "params": 10, "pct": 6}
    cw = col_w

    head = f"  {'Layer':<{cw['name']}s} {'Type':<{cw['type']}s} {'Shape':<{cw['shape']}s} {'Params':>{cw['params']}s}  {'%':>{cw['pct']}s}"
    lines = [f"  {n:<{cw['name']}s} {t:<{cw['type']}s} {s:<{cw['shape']}s} {c:>{cw['params']},}  {p:>{cw['pct'] - 1}.1f}%" for n, t, s, c, p in rows]
    width = max(len(l) for l in [head] + lines) + 2
    sep = "  " + "─" * (width - 2)
    print(f"\n{'=' * width}")
    print(f"  Parameter Distribution")
    print(f"{'=' * width}")
    print(head)
    print(sep)
    for line in lines:
        print(line)

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = total_all - non_trainable
    print(sep)
    pad = width - 2 - cw["params"] - 2
    footer: list[tuple[str, str | int]] = [("Trainable", trainable)]
    if non_trainable:
        footer.append(("Non-trainable", non_trainable))
    footer.extend([("Total", total_all), ("Model size", _format_size(param_bytes))])
    for label, val in footer:
        if isinstance(val, int):
            print(f"  {label:>{pad}s}  {val:>{cw['params']},}")
        else:
            print(f"  {label:>{pad}s}  {val:>{cw['params']}s}")
    print()
