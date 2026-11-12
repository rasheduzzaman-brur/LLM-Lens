"""Lazy tensor inspection: slicing, pagination, pooling and JSON-safe serialization.

Nothing here is part of the model computation. Pooling is a visualization transform and
is always flagged in the response.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import HTTPException

from app.core.config import settings
from app.tracing.schema import TensorStats


def parse_index(spec: Optional[str], shape: tuple[int, ...]) -> tuple:
    """'0,3,:,2:10' -> (0, 3, slice(None), slice(2, 10)). Ints drop a dimension."""
    if not spec:
        return tuple(slice(None) for _ in shape)
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) > len(shape):
        raise HTTPException(422, f"index '{spec}' has more dimensions than tensor shape {list(shape)}")
    out: list = []
    for p, dim in zip(parts, shape):
        try:
            if ":" in p:
                bits = [int(b) if b else None for b in p.split(":")]
                if len(bits) > 3:
                    raise ValueError
                out.append(slice(*bits))
            else:
                i = int(p)
                if not -dim <= i < dim:
                    raise HTTPException(422, f"index {i} out of range for dimension of size {dim}")
                out.append(i)
        except ValueError:
            raise HTTPException(422, f"bad index component '{p}'") from None
    out.extend(slice(None) for _ in range(len(shape) - len(parts)))
    return tuple(out)


def stats(t: torch.Tensor) -> TensorStats:
    if t.numel() == 0:
        return TensorStats()
    x = t.detach()
    if x.dtype == torch.bool or not x.dtype.is_floating_point:
        x = x.float()
    else:
        x = x.float()
    finite = torch.isfinite(x) & (x > -1e30)
    if not bool(finite.any()):
        return TensorStats()
    v = x[finite]
    return TensorStats(
        min=float(v.min()),
        max=float(v.max()),
        mean=float(v.mean()),
        std=float(v.std()) if v.numel() > 1 else 0.0,
        l2=float(torch.linalg.vector_norm(v)),
    )


def to_jsonable(t: torch.Tensor) -> list | float | None:
    """Exact float32->float64 conversion; non-finite values become null."""
    a = t.detach().to("cpu")
    if a.dtype == torch.bool:
        return a.tolist()
    if a.dtype.is_floating_point:
        arr = a.to(torch.float32).numpy().astype(np.float64)
        if np.isfinite(arr).all():
            return arr.tolist()
        obj = arr.astype(object)
        obj[~np.isfinite(arr)] = None
        return obj.tolist()
    return a.tolist()


def view(
    t: torch.Tensor,
    index: Optional[str] = None,
    max_cells: Optional[int] = None,
    pool: bool = True,
    with_stats: bool = True,
) -> dict:
    """Serialize a slice of `t`. Large 2-D results are mean-pooled (flagged); other
    oversized results are rejected so the caller narrows the slice."""
    max_cells = max_cells or settings.max_cells
    idx = parse_index(index, tuple(t.shape))
    sub = t[idx]
    out: dict = {
        "full_shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "index": index or "",
        "shape": list(sub.shape),
        "pooled": False,
    }
    if sub.dtype in (torch.float16, torch.bfloat16):
        out["serialization_note"] = f"{out['dtype']} values are upcast to float32 for transfer (exact, lossless)."
    if with_stats:
        out["stats"] = stats(sub).model_dump()
    n = sub.numel()
    if n > max_cells:
        if sub.dim() == 2 and pool and sub.dtype.is_floating_point:
            side = settings.max_pool_side
            h = min(sub.shape[0], side)
            w = min(sub.shape[1], max(1, min(side, max_cells // h)))
            x = sub.float()
            x = torch.where(torch.isfinite(x) & (x > -1e30), x, torch.zeros_like(x))
            pooled = F.adaptive_avg_pool2d(x[None, None], (h, w))[0, 0]
            out.update(
                pooled=True,
                pooled_shape=[h, w],
                pool_note=(
                    f"Visualization transform: {list(sub.shape)} mean-pooled to [{h}, {w}] for display. "
                    "Request a narrower index for exact values."
                ),
                values=to_jsonable(pooled),
            )
            return out
        raise HTTPException(
            413,
            f"slice has {n} values (> {max_cells}). Narrow the index, e.g. page through the last dimension "
            f"with 'a:b' (shape {list(sub.shape)}).",
        )
    out["values"] = to_jsonable(sub)
    return out


def nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def human_bytes(n: float) -> str:
    if n <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = min(int(math.log(n, 1024)), len(units) - 1)
    return f"{n / 1024 ** i:.1f} {units[i]}"
