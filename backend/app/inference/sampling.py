"""Token selection. Written out explicitly (instead of calling generate()) so every
intermediate distribution can be inspected."""
from __future__ import annotations

from typing import Literal, Optional

import torch
from pydantic import BaseModel, Field


class SamplingConfig(BaseModel):
    strategy: Literal["greedy", "sample"] = "greedy"
    temperature: float = Field(1.0, gt=0.0, le=100.0)
    top_k: int = Field(0, ge=0, description="0 disables top-k")
    top_p: float = Field(1.0, gt=0.0, le=1.0, description="1.0 disables nucleus filtering")
    seed: Optional[int] = 0


class Distribution(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    logits: torch.Tensor          # raw logits [V] (float32)
    probs_raw: torch.Tensor       # softmax(logits)
    scaled: torch.Tensor          # logits / temperature
    kept: torch.Tensor            # bool [V], survives top-k / top-p
    probs: torch.Tensor           # final sampling distribution


def distribution(logits: torch.Tensor, cfg: SamplingConfig) -> Distribution:
    """logits [V] -> every stage of the sampling distribution.

    Order mirrors Hugging Face logits processors: temperature, then top-k, then top-p.
    """
    logits = logits.detach().float()
    probs_raw = torch.softmax(logits, dim=-1)
    scaled = logits / cfg.temperature
    kept = torch.ones_like(scaled, dtype=torch.bool)
    if cfg.top_k and cfg.top_k < scaled.numel():
        kth = torch.topk(scaled, cfg.top_k).values[-1]
        kept &= scaled >= kth
    if cfg.top_p < 1.0:
        masked = scaled.masked_fill(~kept, float("-inf"))
        sorted_logits, sorted_idx = torch.sort(masked, descending=True)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # drop tokens once the mass *before* them already reaches top_p (always keeps the top-1)
        remove = (cum - torch.softmax(sorted_logits, dim=-1)) >= cfg.top_p
        drop = torch.zeros_like(kept)
        drop[sorted_idx[remove]] = True
        kept &= ~drop
    probs = torch.softmax(scaled.masked_fill(~kept, float("-inf")), dim=-1)
    return Distribution(logits=logits, probs_raw=probs_raw, scaled=scaled, kept=kept, probs=probs)


def select(dist: Distribution, cfg: SamplingConfig, generator: Optional[torch.Generator]) -> int:
    if cfg.strategy == "greedy":
        return int(torch.argmax(dist.logits).item())
    return int(torch.multinomial(dist.probs.cpu(), 1, generator=generator).item())
