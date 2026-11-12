"""Read-side services over a run's captured tensors.

All positions are absolute sequence positions. Rows computed in different steps
(prefill vs. later decode steps) are stitched together; that is valid because a
position's activations never change once computed (causal attention + KV cache).
"""
from __future__ import annotations

import copy
import math
import time
from typing import Optional

import torch
from fastapi import HTTPException

from app.core.config import settings
from app.inference.sampling import SamplingConfig, distribution
from app.inference.session import Run, entropy, token_text
from app.services import tensor_view as tv
from app.tracing.engine import StepRecord, cache_kv

RELEASED = ("tensor '{name}' of step {step} is not stored (capture mode '{mode}' or released by trace retention). "
            "Re-run with full capture or raise the retention settings.")


def need(run: Run, rec: StepRecord, layer: Optional[int], name: str) -> torch.Tensor:
    t = rec.tensor(layer, name)
    if t is None:
        if rec.by_name.get((layer, name)) is None:
            raise HTTPException(404, f"no tensor '{name}' at layer {layer} in step {rec.index} for this architecture")
        raise HTTPException(409, RELEASED.format(name=name, step=rec.index, mode=run.capture.mode))
    return t


def locate(run: Run, position: int, upto: int) -> tuple[StepRecord, int]:
    """absolute position -> (step that computed it, row inside that step)."""
    last = run.step(upto)
    if not 0 <= position < last.seq_len:
        raise HTTPException(422, f"position {position} out of range (0…{last.seq_len - 1}) at step {last.index}")
    n = len(run.prompt_ids)
    return (run.steps[0], position) if position < n else (run.steps[position - n + 1], 0)


def assemble(run: Run, layer: Optional[int], name: str, upto: int) -> tuple[torch.Tensor, bool]:
    """Concatenate a per-position tensor over steps 0…upto along the sequence axis (dim -2)."""
    last = run.step(upto)
    first = need(run, run.steps[0], layer, name)
    if last.index == 0 or first.dim() < 2 or first.shape[-2] != run.steps[0].seq_len:
        return need(run, last, layer, name), False
    parts = [need(run, r, layer, name) for r in run.steps[: last.index + 1]]
    ref = list(parts[0].shape)
    for p in parts[1:]:
        s = list(p.shape)
        if len(s) != len(ref) or s[:-2] != ref[:-2] or s[-1] != ref[-1]:
            return need(run, last, layer, name), False
    return torch.cat(parts, dim=-2), True


def labels(run: Run, n: int) -> list[str]:
    return [token_text(run.tok, t) for t in run.all_ids()[:n]]


# ---------------------------------------------------------------------------- attention
def attention_matrix(run: Run, layer: int, upto: int) -> torch.Tensor:
    """[H, T, T] attention weights for all positions processed up to `upto`; future = NaN."""
    last = run.step(upto)
    t = last.seq_len
    w0 = need(run, run.steps[0], layer, "weights")[0]
    out = torch.full((w0.shape[0], t, t), float("nan"), dtype=torch.float32, device=w0.device)
    for rec in run.steps[: last.index + 1]:
        w = need(run, rec, layer, "weights")[0].float()
        out[:, rec.past_len : rec.seq_len, : rec.seq_len] = w
    future = torch.triu(torch.ones(t, t, dtype=torch.bool, device=out.device), diagonal=1)
    out[:, future] = float("nan")  # masked cells (weight exactly 0 in the model) are reported as null
    return out


def attention(run: Run, layer: int, upto: int, head: Optional[int]) -> dict:
    m = attention_matrix(run, layer, upto)
    h, t, _ = m.shape
    if head is not None:
        if not 0 <= head < h:
            raise HTTPException(422, f"head {head} out of range (0…{h - 1})")
        sel = m[head : head + 1]
    else:
        sel = m
    if sel.numel() > settings.max_cells:
        raise HTTPException(413, f"{h} heads × {t}×{t} is too large for one response; request a single head")
    p = torch.nan_to_num(m, nan=0.0)
    ent = -(p * torch.where(p > 0, p.log(), torch.zeros_like(p))).sum(-1)
    last = run.step(upto)
    checks = [run.steps[s].event(layer, "weights") for s in range(last.index + 1)]
    merged = [run.steps[s].event(layer, "attn_merged") for s in range(last.index + 1)]
    return {
        "layer": layer, "step": last.index, "n_heads": h, "seq_len": t, "head": head, "tokens": labels(run, t),
        "weights": tv.to_jsonable(sel), "entropy": tv.to_jsonable(ent if head is None else ent[head : head + 1]),
        "n_prompt_tokens": len(run.prompt_ids),
        "verified": all(e.verified for e in checks if e.verified is not None) and all(e.verified for e in merged),
        "max_abs_err": max([e.max_abs_err or 0.0 for e in checks + merged]),
        "verified_against": "the model's own attention weights and its attention output (input of the output projection)",
        "note": "Rows of generated tokens were computed in their own decode step and stitched; cells above the diagonal are masked (null).",
    }


def attention_grid(run: Run, upto: int, reduce: str, head: int = 0) -> dict:
    """One [T, T] matrix per layer (mean / max over heads, or a fixed head) for cross-layer comparison."""
    n = run.adapter.n_layers
    last = run.step(upto)
    t = last.seq_len
    if n * t * t > settings.max_cells * 4:
        raise HTTPException(413, "sequence too long for the cross-layer grid")
    mats = []
    for layer in range(n):
        m = attention_matrix(run, layer, upto)
        if reduce == "mean":
            mats.append(m.mean(0))
        elif reduce == "max":
            mats.append(m.max(0).values)
        else:
            mats.append(m[head])
    return {"step": last.index, "reduce": reduce, "head": head, "seq_len": t, "tokens": labels(run, t),
            "matrices": tv.to_jsonable(torch.stack(mats)),
            "note": None if reduce == "head" else f"Visualization transform: {reduce} over the {run.adapter.n_heads} heads of each layer."}


def attention_row(run: Run, layer: int, head: int, position: int, upto: int) -> dict:
    """Everything that produces one row of attention weights: Q·Kᵀ, scale, mask, exp, normalise."""
    rec, row = locate(run, position, upto)
    ad = run.adapter
    raw = need(run, rec, layer, "scores_raw")[0, head, row].float()
    scaled = need(run, rec, layer, "scores")[0, head, row].float()
    allowed = need(run, rec, layer, "mask")[row]
    w = need(run, rec, layer, "weights")[0, head, row].float()
    shift = scaled[allowed].max()
    ex = torch.where(allowed, torch.exp(scaled - shift), torch.zeros_like(scaled))
    denom = ex.sum()
    recomputed = ex / denom
    q = need(run, rec, layer, "q_used")[0, head, row]
    out = need(run, rec, layer, "head_out")[0, head, row]
    return {
        "layer": layer, "head": head, "position": position, "src_step": rec.index, "phase": rec.phase,
        "token": token_text(run.tok, run.all_ids()[position]), "tokens": labels(run, rec.seq_len),
        "d_k": ad.head_dim, "scale": ad.attention_scale(layer), "scores_raw": tv.to_jsonable(raw),
        "scores": tv.to_jsonable(scaled), "allowed": allowed.tolist(), "shift": float(shift),
        "exp": tv.to_jsonable(ex), "denominator": float(denom), "weights": tv.to_jsonable(w),
        "weights_sum": float(w.sum()), "recompute_max_err": float((recomputed - w).abs().max()),
        "query": tv.to_jsonable(q), "head_output": tv.to_jsonable(out),
        "exp_note": "exp is evaluated as exp(score − max_allowed_score) for numerical stability; the ratio exp/Σexp is unchanged.",
    }


# ---------------------------------------------------------------------------- per-layer detail
def _norm_params(module) -> dict:
    eps = getattr(module, "eps", None)
    if eps is None:
        eps = getattr(module, "variance_epsilon", None)
    out = {"type": type(module).__name__, "eps": float(eps) if eps is not None else None}
    if getattr(module, "weight", None) is not None:
        out["gamma"] = tv.to_jsonable(module.weight)
    if getattr(module, "bias", None) is not None:
        out["beta"] = tv.to_jsonable(module.bias)
    return out


def _collect(run: Run, rec: StepRecord, row: int, layer: Optional[int], stages: tuple[str, ...]) -> tuple[dict, dict, dict, list]:
    vectors, scalars, checks, events = {}, {}, {}, []
    new_len = rec.seq_len - rec.past_len
    for ev in rec.events:
        if ev.layer != layer or ev.stage not in stages:
            continue
        events.append(ev.model_dump(exclude={"values"}))
        if ev.verified is not None:
            checks[ev.tensor_name] = {"verified": ev.verified, "max_abs_err": ev.max_abs_err, "against": ev.verified_against}
        t = rec.tensors.get(ev.event_id)
        if t is None or t.dim() != 3 or t.shape[1] != new_len:
            continue
        v = t[0, row]
        if v.numel() == 1:
            scalars[ev.tensor_name] = float(v)
        else:
            vectors[ev.tensor_name] = tv.to_jsonable(v)
    return vectors, scalars, checks, events


def layer_detail(run: Run, layer: int, position: int, upto: int) -> dict:
    ad = run.adapter
    if not 0 <= layer < ad.n_layers:
        raise HTTPException(422, f"layer {layer} out of range (0…{ad.n_layers - 1})")
    rec, row = locate(run, position, upto)
    vectors, scalars, checks, events = _collect(run, rec, row, layer, ("residual", "norm", "attention", "mlp"))
    if not vectors:
        raise HTTPException(409, RELEASED.format(name="layer tensors", step=rec.index, mode=run.capture.mode))
    n1, n2 = ad.norm_modules(layer)
    return {
        "layer": layer, "position": position, "src_step": rec.index, "phase": rec.phase,
        "token": token_text(run.tok, run.all_ids()[position]), "vectors": vectors, "scalars": scalars, "checks": checks,
        "norm": {"ln1": _norm_params(n1), "ln2": _norm_params(n2)}, "capabilities": ad.capabilities(),
        "events": events,
    }


def global_detail(run: Run, position: int, upto: int) -> dict:
    ad = run.adapter
    rec, row = locate(run, position, upto)
    vectors, scalars, checks, events = _collect(run, rec, row, None, ("embedding", "final_norm"))
    return {
        "position": position, "src_step": rec.index, "phase": rec.phase,
        "token": token_text(run.tok, run.all_ids()[position]), "token_id": run.all_ids()[position],
        "vectors": vectors, "scalars": scalars, "checks": checks,
        "norm": {"final_norm": _norm_params(ad.final_norm)}, "capabilities": ad.capabilities(), "events": events,
    }


def layers_summary(run: Run, upto: int) -> dict:
    rec = run.step(upto)
    ad = run.adapter
    out = []
    for i in range(ad.n_layers):
        evs = [e for e in rec.events if e.layer == i]
        chk = [e for e in evs if e.verified is not None]
        row = {"layer": i, "n_events": len(evs), "checks": len(chk), "checks_failed": sum(1 for e in chk if not e.verified),
               "max_abs_err": max([e.max_abs_err or 0.0 for e in chk], default=0.0),
               "time_ms": (rec.event(i, "resid_post").execution_time_ms if rec.event(i, "resid_post") else None), "norms": {}}
        for name in ("resid_pre", "attn_out", "resid_mid", "mlp_out", "resid_post"):
            t = rec.tensor(i, name)
            if t is not None:
                row["norms"][name] = float(torch.linalg.vector_norm(t[0, -1].float()))
        out.append(row)
    return {"step": rec.index, "phase": rec.phase, "position": rec.seq_len - 1, "layers": out,
            "note": "Norms are L2 norms of the last position's vector at each point in the layer."}


# ---------------------------------------------------------------------------- hidden states
HIDDEN_STAGES = {
    "resid_post": "Layer output (residual stream)",
    "resid_mid": "After attention residual add",
    "attn_out": "Attention output",
    "mlp_out": "MLP output",
}


def hidden_stack(run: Run, stage: str, upto: int) -> tuple[torch.Tensor, list[str]]:
    """[N, T, D] stack over depth plus a label per row."""
    if stage not in HIDDEN_STAGES:
        raise HTTPException(422, f"stage must be one of {list(HIDDEN_STAGES)}")
    ad = run.adapter
    rows, names = [], []
    if stage == "resid_post":
        rows.append(assemble(run, None, "emb_out", upto)[0][0])
        names.append("embedding")
    for i in range(ad.n_layers):
        rows.append(assemble(run, i, stage, upto)[0][0])
        names.append(f"L{i}")
    if stage == "resid_post":
        rows.append(assemble(run, None, "final_norm_out", upto)[0][0])
        names.append("final norm")
    return torch.stack(rows).float(), names


def cosine_matrix(x: torch.Tensor) -> torch.Tensor:
    n = torch.nn.functional.normalize(x, dim=-1, eps=1e-12)
    return n @ n.T


def pca2(x: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
    """Visualization-only PCA to 2-D via SVD of the centred data. x [N, D] -> [N, 2]."""
    x = x.float().cpu()
    if x.shape[0] < 2:
        return torch.zeros(x.shape[0], 2), [0.0, 0.0]
    xc = x - x.mean(0, keepdim=True)
    u, s, _ = torch.linalg.svd(xc, full_matrices=False)
    k = min(2, s.numel())
    pts = u[:, :k] * s[:k]
    if k < 2:
        pts = torch.cat([pts, torch.zeros(pts.shape[0], 2 - k)], dim=1)
    var = (s ** 2)
    ratio = (var[:2] / var.sum().clamp_min(1e-30)).tolist()
    return pts, ratio + [0.0] * (2 - len(ratio))


def hidden_states(run: Run, stage: str, upto: int, position: int, depth: int) -> dict:
    stack, names = hidden_stack(run, stage, upto)
    n, t, d = stack.shape
    position = max(0, min(position if position >= 0 else t - 1, t - 1))
    depth = max(0, min(depth if depth >= 0 else n - 1, n - 1))
    pts, ratio = pca2(stack.reshape(n * t, d))
    pts = pts.reshape(n, t, 2)
    seq = stack[:, position]
    consecutive = torch.nn.functional.cosine_similarity(seq[1:], seq[:-1], dim=-1) if n > 1 else torch.zeros(0)
    return {
        "stage": stage, "stages": HIDDEN_STAGES, "step": run.step(upto).index, "depth_labels": names,
        "tokens": labels(run, t), "hidden_size": d, "position": position, "depth": depth,
        "norms": tv.to_jsonable(torch.linalg.vector_norm(stack, dim=-1)),
        "pca": {"points": tv.to_jsonable(pts), "explained_variance_ratio": ratio,
                "note": "Visualization transform: PCA fitted on every (depth, position) vector of this stage; not a model computation."},
        "token_similarity": tv.to_jsonable(cosine_matrix(stack[depth])),
        "depth_similarity": tv.to_jsonable(cosine_matrix(seq)),
        "consecutive_similarity": tv.to_jsonable(consecutive),
        "similarity_note": "Cosine similarity is a visualization transform computed in float32.",
    }


# ---------------------------------------------------------------------------- logits
def logits_view(run: Run, upto: int, cfg: Optional[SamplingConfig], top_n: int) -> dict:
    rec = run.step(upto)
    logits = need(run, rec, None, "next_token_logits")
    used = cfg or SamplingConfig(**rec.sampling["config"])
    dist = distribution(logits, used)
    temp_only = torch.softmax(dist.scaled, dim=-1)
    order = torch.argsort(dist.logits, descending=True)[: max(1, min(top_n, 500))]
    rows = [{"rank": r, "token_id": int(t), "token": token_text(run.tok, int(t)), "logit": float(dist.logits[t]),
             "prob_raw": float(dist.probs_raw[t]), "prob_temperature": float(temp_only[t]),
             "prob_final": float(dist.probs[t]), "kept": bool(dist.kept[t])} for r, t in enumerate(order.tolist())]
    sel = rec.sampling
    sid = sel["token_id"]
    return {
        "step": rec.index, "phase": rec.phase, "position": rec.seq_len - 1,
        "context_token": token_text(run.tok, run.all_ids()[rec.seq_len - 1]),
        "vocab_size": int(logits.numel()), "config": used.model_dump(), "what_if": cfg is not None,
        "top": rows, "kept": int(dist.kept.sum()), "entropy_raw": entropy(dist.probs_raw),
        "entropy_final": entropy(dist.probs), "greedy_token_id": int(order[0]),
        "logit_stats": tv.stats(dist.logits).model_dump(),
        "selected": {**sel, "prob_under_view": float(dist.probs[sid]), "kept_under_view": bool(dist.kept[sid])},
        "dtype": str(logits.dtype).replace("torch.", ""),
        "note": "Probabilities are computed in float32 from the model's logits. With a what-if configuration the model is not "
                "re-run and the token actually selected in this step does not change.",
    }


# ---------------------------------------------------------------------------- KV cache
def kv_cache(run: Run, upto: int) -> dict:
    rec = run.step(upto)
    ad = run.adapter
    history = [{"step": r.index, "phase": r.phase, "seq_len": r.seq_len, "new_len": r.seq_len - r.past_len,
                "cache_bytes": r.cache_bytes, "forward_ms": r.forward_ms,
                "token": token_text(run.tok, r.input_ids[-1]),
                "positions_without_cache": r.seq_len, "positions_with_cache": r.seq_len - r.past_len}
               for r in run.steps]
    return {
        "step": rec.index, "phase": rec.phase, "seq_len": rec.seq_len, "past_len": rec.past_len,
        "new_len": rec.seq_len - rec.past_len, "n_layers": ad.n_layers, "n_kv_heads": ad.n_kv_heads,
        "n_heads": ad.n_heads, "head_dim": ad.head_dim, "layers": rec.cache_layers, "total_bytes": rec.cache_bytes,
        "cache_class": type(run.cache).__name__, "history": history, "tokens": labels(run, rec.seq_len),
        "key_note": ("Cached keys already include the rotary position rotation." if ad.capabilities()["rope"]
                     else "Cached keys are the K projections split into heads."),
    }


def kv_cache_tensor(run: Run, upto: int, layer: int, which: str, head: int, index: Optional[str]) -> dict:
    rec = run.step(upto)
    if which not in ("key", "value"):
        raise HTTPException(422, "which must be 'key' or 'value'")
    k, v = cache_kv(run.cache, layer)
    t = (k if which == "key" else v)[0, :, : rec.seq_len]
    if not 0 <= head < t.shape[0]:
        raise HTTPException(422, f"head {head} out of range (0…{t.shape[0] - 1})")
    out = tv.view(t[head], index)
    out.update(layer=layer, which=which, head=head, step=rec.index, new_from=rec.past_len, tokens=labels(run, rec.seq_len),
               note="Read from the live cache object, truncated to this step's sequence length (earlier rows never change).")
    return out


@torch.inference_mode()
def kv_cache_compare(run: Run, upto: int) -> dict:
    """Really run both paths (without hooks) for this step and compare timing and logits."""
    rec = run.step(upto)
    ad = run.adapter
    ids = torch.tensor([run.all_ids()[: rec.seq_len]], dtype=torch.long, device=ad.device)
    with run.lock, run.lm.lock:
        past = copy.deepcopy(run.cache)
        past.crop(rec.past_len)
        t0 = time.perf_counter()
        full = ad.model(input_ids=ids, use_cache=False).logits[0, -1]
        t_full = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        if rec.past_len:
            cached = ad.model(input_ids=ids[:, rec.past_len :], past_key_values=past, use_cache=True).logits[0, -1]
        else:
            cached = ad.model(input_ids=ids, past_key_values=past, use_cache=True).logits[0, -1]
        t_cached = (time.perf_counter() - t0) * 1e3
    traced = rec.tensor(None, "next_token_logits")
    return {
        "step": rec.index, "phase": rec.phase, "seq_len": rec.seq_len,
        "without_cache": {"positions_processed": rec.seq_len, "forward_ms": t_full},
        "with_cache": {"positions_processed": rec.seq_len - rec.past_len, "reused_positions": rec.past_len, "forward_ms": t_cached},
        "speedup": (t_full / t_cached) if t_cached > 0 else None,
        "max_abs_logit_diff": float((full.float() - cached.float()).abs().max()),
        "max_abs_diff_vs_trace": float((cached.float() - traced.float()).abs().max()) if traced is not None else None,
        "same_argmax": bool(full.argmax() == cached.argmax()),
        "note": "Both passes were executed just now without instrumentation hooks. Timings are single measurements and noisy, "
                "especially for short sequences.",
    }
