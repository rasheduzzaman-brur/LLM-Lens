from __future__ import annotations

import asyncio
import json
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from app.inference.sampling import SamplingConfig
from app.inference.session import create_run, runs
from app.services import analysis
from app.services import tensor_view as tv
from app.services.model_manager import models
from app.tracing.schema import CaptureConfig

router = APIRouter(prefix="/api/inference", tags=["inference"])


class RunRequest(BaseModel):
    model_id: str
    prompt: str
    capture: CaptureConfig = CaptureConfig()
    sampling: SamplingConfig = SamplingConfig()
    max_new_tokens: int = Field(32, ge=1, le=1024)
    stop_on_eos: bool = True


class StepRequest(BaseModel):
    kind: Literal["next_token", "next_op", "prev_op", "goto"] = "next_token"
    step: Optional[int] = None
    order: Optional[int] = None
    sampling: Optional[SamplingConfig] = Field(None, description="Replace the sampling configuration for the following tokens")


class ResumeRequest(BaseModel):
    delay_ms: int = Field(0, ge=0, le=5000)
    sampling: Optional[SamplingConfig] = None


@router.post("/run")
def run_inference(req: RunRequest) -> dict:
    """Tokenize the prompt and run the traced prefill pass (step 0), selecting the first new token."""
    lm = models.get(req.model_id)
    return create_run(lm, req.prompt, req.capture, req.sampling, req.max_new_tokens, req.stop_on_eos).summary()


@router.get("")
def list_runs() -> dict:
    return {"runs": [{"run_id": r.run_id, "model_id": r.lm.model_id, "prompt": r.prompt, "state": r.state,
                      "steps": len(r.steps), "stored_bytes": r.stored_bytes(), "created": r.created}
                     for r in runs.runs.values()], "usage": runs.usage()}


@router.get("/{run_id}")
def get_run(run_id: str) -> dict:
    return runs.get(run_id).summary()


@router.delete("/{run_id}")
def delete_run(run_id: str) -> dict:
    runs.delete(run_id)
    return {"deleted": run_id}


# ------------------------------------------------------------------ control
@router.post("/{run_id}/step")
def step(run_id: str, req: StepRequest = StepRequest()) -> dict:
    run = runs.get(run_id)
    if run.state == "running":
        raise HTTPException(409, "run is generating; pause it before stepping manually")
    if req.sampling is not None:
        run.sampling = req.sampling
    if req.kind == "next_token":
        run.next_token()
    else:
        run.move_cursor(req.kind, req.step, req.order)
    return run.summary()


@router.post("/{run_id}/pause")
def pause(run_id: str) -> dict:
    run = runs.get(run_id)
    run.pause()
    return run.summary()


@router.post("/{run_id}/resume")
def resume(run_id: str, req: ResumeRequest = ResumeRequest()) -> dict:
    """Start (or continue) automatic token-by-token generation in the background."""
    run = runs.get(run_id)
    if req.sampling is not None:
        run.sampling = req.sampling
    run.delay_ms = req.delay_ms
    run.resume()
    return run.summary()


@router.post("/{run_id}/reset")
def reset(run_id: str) -> dict:
    """Discard generated tokens and re-run the prefill pass."""
    run = runs.get(run_id)
    run.reset()
    return run.summary()


@router.get("/{run_id}/events")
async def events(run_id: str, request: Request) -> StreamingResponse:
    """Server-Sent Events: a `run` event with the run summary whenever it changes."""
    run = runs.get(run_id)

    async def gen():
        seen = -1
        idle = 0
        while True:
            if await request.is_disconnected():
                break
            if run.version != seen:
                seen = run.version
                idle = 0
                yield f"event: run\ndata: {json.dumps(run.summary())}\n\n"
            else:
                idle += 1
                if idle % 150 == 0:
                    yield ": keep-alive\n\n"
            if run_id not in runs.runs:
                yield "event: gone\ndata: {}\n\n"
                break
            await asyncio.sleep(0.1)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ------------------------------------------------------------------ trace
@router.get("/{run_id}/trace")
def trace(run_id: str, step: int = -1, layer: Optional[int] = None, stage: Optional[str] = None,
          global_only: bool = False, offset: int = Query(0, ge=0), limit: int = Query(200, ge=1, le=2000)) -> dict:
    """Ordered trace events of one step (metadata only; fetch values through the tensor endpoints)."""
    run = runs.get(run_id)
    rec = run.step(step)
    evs = [e for e in rec.events
           if (layer is None or e.layer == layer) and (stage is None or e.stage == stage)
           and (not global_only or e.layer is None)]
    return {"run_id": run_id, "step": rec.index, "phase": rec.phase, "total": len(evs), "offset": offset, "limit": limit,
            "forward_ms": rec.forward_ms, "stages": sorted({e.stage for e in rec.events}),
            "events": [e.model_dump(exclude={"values"}) for e in evs[offset : offset + limit]]}


@router.get("/{run_id}/trace-export")
def trace_export(run_id: str, step: int = -1) -> Response:
    """Trace events of one step as CSV (metadata and statistics, no tensor values)."""
    import pandas as pd

    rec = runs.get(run_id).step(step)
    rows = []
    for e in rec.events:
        d = e.model_dump(exclude={"values", "stats", "inputs"})
        d["shape"] = "x".join(map(str, e.shape))
        d["inputs"] = " ".join(e.inputs)
        d.update({f"stat_{k}": v for k, v in (e.stats.model_dump() if e.stats else {}).items()})
        rows.append(d)
    csv = pd.DataFrame(rows).to_csv(index=False)
    return Response(csv, media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="trace_{run_id}_step{rec.index}.csv"'})


@router.get("/{run_id}/trace/{event_id}")
def trace_event(run_id: str, event_id: int, index: Optional[str] = None, values: bool = True) -> dict:
    """One event; with `values` a slice of its tensor (numpy-style `index`, e.g. `0,3,:,0:64`)."""
    run = runs.get(run_id)
    rec, ev = run.find_event(event_id)
    out = {"event": ev.model_dump(exclude={"values"})}
    if values:
        t = rec.tensors.get(event_id)
        if t is None:
            out["tensor"] = None
            out["unavailable_reason"] = ev.note or f"Not stored (capture mode '{run.capture.mode}')."
        else:
            out["tensor"] = tv.view(t, index)
    return out


@router.get("/{run_id}/tensor")
def tensor(run_id: str, name: str, step: int = -1, layer: Optional[int] = None, index: Optional[str] = None,
           assembled: bool = False) -> dict:
    """Tensor by (step, layer, name). `assembled` stitches per-position tensors over steps 0…step."""
    run = runs.get(run_id)
    rec = run.step(step)
    if assembled:
        t, did = analysis.assemble(run, layer, name, rec.index)
    else:
        t, did = analysis.need(run, rec, layer, name), False
    ev = rec.event(layer, name)
    out = tv.view(t, index)
    out.update(name=name, layer=layer, step=rec.index, assembled=did, event=ev.model_dump(exclude={"values"}) if ev else None,
               tokens=analysis.labels(run, rec.seq_len))
    return out


# ------------------------------------------------------------------ analysis views
@router.get("/{run_id}/layers")
def layers(run_id: str, step: int = -1) -> dict:
    return analysis.layers_summary(runs.get(run_id), step)


@router.get("/{run_id}/layers/{layer}")
def layer(run_id: str, layer: int, position: int = -1, step: int = -1) -> dict:
    """Every per-token vector of one layer (norm steps, Q/K/V rows, residuals, MLP) for one position."""
    run = runs.get(run_id)
    rec = run.step(step)
    return analysis.layer_detail(run, layer, position if position >= 0 else rec.seq_len - 1, rec.index)


@router.get("/{run_id}/global")
def global_detail(run_id: str, position: int = -1, step: int = -1) -> dict:
    """Embedding and final-normalization vectors for one position."""
    run = runs.get(run_id)
    rec = run.step(step)
    return analysis.global_detail(run, position if position >= 0 else rec.seq_len - 1, rec.index)


@router.get("/{run_id}/attention")
def attention(run_id: str, layer: int, step: int = -1, head: Optional[int] = None) -> dict:
    run = runs.get(run_id)
    if not 0 <= layer < run.adapter.n_layers:
        raise HTTPException(422, "layer out of range")
    return analysis.attention(run, layer, step, head)


@router.get("/{run_id}/attention/grid")
def attention_grid(run_id: str, step: int = -1, reduce: Literal["mean", "max", "head"] = "mean", head: int = 0) -> dict:
    return analysis.attention_grid(runs.get(run_id), step, reduce, head)


@router.get("/{run_id}/attention/row")
def attention_row(run_id: str, layer: int, head: int, position: int, step: int = -1) -> dict:
    run = runs.get(run_id)
    if not 0 <= layer < run.adapter.n_layers or not 0 <= head < run.adapter.n_heads:
        raise HTTPException(422, "layer or head out of range")
    return analysis.attention_row(run, layer, head, position, step)


@router.get("/{run_id}/hidden-states")
def hidden_states(run_id: str, stage: str = "resid_post", step: int = -1, position: int = -1, depth: int = -1) -> dict:
    return analysis.hidden_states(runs.get(run_id), stage, step, position, depth)


@router.get("/{run_id}/logits")
def logits(run_id: str, step: int = -1, top_n: int = Query(50, ge=1, le=500),
           temperature: Optional[float] = Query(None, gt=0, le=100), top_k: Optional[int] = Query(None, ge=0),
           top_p: Optional[float] = Query(None, gt=0, le=1), strategy: Optional[Literal["greedy", "sample"]] = None) -> dict:
    """Next-token distribution of a step. Passing any sampling parameter evaluates a what-if distribution."""
    run = runs.get(run_id)
    cfg = None
    if any(v is not None for v in (temperature, top_k, top_p, strategy)):
        base = run.step(step).sampling["config"]
        cfg = SamplingConfig(strategy=strategy or base["strategy"],
                             temperature=temperature if temperature is not None else base["temperature"],
                             top_k=top_k if top_k is not None else base["top_k"],
                             top_p=top_p if top_p is not None else base["top_p"], seed=base["seed"])
    return analysis.logits_view(run, step, cfg, top_n)


@router.get("/{run_id}/kv-cache")
def kv_cache(run_id: str, step: int = -1) -> dict:
    return analysis.kv_cache(runs.get(run_id), step)


@router.get("/{run_id}/kv-cache/tensor")
def kv_cache_tensor(run_id: str, layer: int, which: Literal["key", "value"] = "key", head: int = 0, step: int = -1,
                    index: Optional[str] = None) -> dict:
    run = runs.get(run_id)
    if not 0 <= layer < run.adapter.n_layers:
        raise HTTPException(422, "layer out of range")
    return analysis.kv_cache_tensor(run, step, layer, which, head, index)


@router.get("/{run_id}/kv-cache/compare")
def kv_cache_compare(run_id: str, step: int = -1) -> dict:
    run = runs.get(run_id)
    if run.state == "running":
        raise HTTPException(409, "pause generation before running the cache comparison")
    return analysis.kv_cache_compare(run, step)
