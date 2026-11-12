"""Re-derivation of operations that have no module boundary (fused attention math,
normalization internals). Every function mirrors the Hugging Face implementation
line by line so results can be verified against what the model actually produced."""
from __future__ import annotations

from typing import Literal, Optional

import torch


def tolerance(dtype: torch.dtype) -> tuple[float, float]:
    """(rtol, atol) appropriate for comparing two tensors of `dtype`."""
    if dtype == torch.float64:
        return 1e-9, 1e-11
    if dtype in (torch.float16, torch.bfloat16):
        return 3e-2, 3e-3
    return 1e-4, 1e-5


def compare(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, float]:
    """-> (matches within dtype tolerance, max abs error)."""
    if a.shape != b.shape:
        return False, float("inf")
    if a.numel() == 0:
        return True, 0.0
    af, bf = a.float(), b.float()
    err = float((af - bf).abs().max())
    rtol, atol = tolerance(a.dtype if a.dtype.is_floating_point else torch.float32)
    return bool(torch.allclose(af, bf, rtol=rtol, atol=atol)), err


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """[B, T, H*dh] -> [B, H, T, dh]"""
    b, t, d = x.shape
    return x.view(b, t, n_heads, d // n_heads).transpose(1, 2)


def causal_allowed(tq: int, tk: int, device) -> torch.Tensor:
    """bool [tq, tk]; row r (absolute position tk - tq + r) may attend to columns <= its position."""
    return torch.tril(torch.ones(tk, tk, dtype=torch.bool, device=device))[tk - tq : tk, :tk]


def attention_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    mask_mode: Literal["fill", "add"],
    softmax_dtype: Optional[torch.dtype] = None,
) -> dict[str, torch.Tensor]:
    """q [B,H,Tq,dh], k/v [B,H,Tk,dh] -> every intermediate of scaled dot-product attention.

    mask_mode 'fill': masked scores are replaced by finfo.min (GPT-2).
    mask_mode 'add' : finfo.min is added to masked scores (Llama / Qwen2).
    """
    tq, tk = q.shape[-2], k.shape[-2]
    scores_raw = torch.matmul(q, k.transpose(-1, -2))
    scores = scores_raw * scale
    allowed = causal_allowed(tq, tk, q.device)
    neg = torch.finfo(scores.dtype).min
    if mask_mode == "fill":
        masked = torch.where(allowed, scores, torch.full([], neg, dtype=scores.dtype, device=scores.device))
    else:
        add = torch.zeros(tq, tk, dtype=scores.dtype, device=scores.device).masked_fill(~allowed, neg)
        masked = scores + add
    if softmax_dtype is not None:
        weights = torch.softmax(masked, dim=-1, dtype=softmax_dtype).to(q.dtype)
    else:
        weights = torch.softmax(masked, dim=-1).type(v.dtype)
    head_out = torch.matmul(weights, v)
    b, h, _, dh = head_out.shape
    merged = head_out.transpose(1, 2).reshape(b, tq, h * dh)
    return {
        "scores_raw": scores_raw,
        "scores": scores,
        "allowed": allowed,
        "scores_masked": masked,
        "weights": weights,
        "head_out": head_out,
        "merged": merged,
    }


def layernorm_steps(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], eps: float) -> dict:
    """torch.nn.LayerNorm over the last dim (biased variance), step by step."""
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    normalized = (x - mean) / torch.sqrt(var + eps)
    out = normalized * weight
    if bias is not None:
        out = out + bias
    return {"mean": mean, "var": var, "normalized": normalized, "out": out}


def rmsnorm_steps(x: torch.Tensor, weight: torch.Tensor, eps: float) -> dict:
    """LlamaRMSNorm / Qwen2RMSNorm: computed in float32, cast back, then scaled."""
    dtype = x.dtype
    xf = x.to(torch.float32)
    mean_sq = xf.pow(2).mean(-1, keepdim=True)
    rms = torch.sqrt(mean_sq + eps)
    normalized = (xf * torch.rsqrt(mean_sq + eps)).to(dtype)
    return {"mean_sq": mean_sq, "rms": rms, "normalized": normalized, "out": weight * normalized}


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B,H,T,dh], cos/sin [B,T,dh]"""
    return x * cos.unsqueeze(1) + rotate_half(x) * sin.unsqueeze(1)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, Hkv, T, dh] -> [B, Hkv*n_rep, T, dh] (grouped-query attention)."""
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)
