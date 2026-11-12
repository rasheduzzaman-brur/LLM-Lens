"""Generation sessions: prefill + step-by-step autoregressive decoding with a KV cache.

Every step is a real, separately traced forward pass. Token selection is done here
(not by `generate()`) so the full sampling distribution stays inspectable.
"""
from __future__ import annotations

import math
import threading
import time
import uuid
from collections import OrderedDict
from typing import Optional

import torch
from fastapi import HTTPException

from app.core.config import settings
from app.inference.sampling import SamplingConfig, distribution, select
from app.services.model_manager import LoadedModel
from app.tracing.engine import Emitter, StepRecord, new_cache, trace_forward
from app.tracing.schema import CaptureConfig, TraceEvent
from app.instrumentation.hooks import Capture


def token_text(tok, tid: int) -> str:
    return tok.decode([tid], clean_up_tokenization_spaces=False)


def entropy(p: torch.Tensor) -> float:
    q = p[p > 0]
    return float(-(q * q.log()).sum())


class Run:
    def __init__(self, lm: LoadedModel, prompt: str, prompt_ids: list[int], capture: CaptureConfig,
                 sampling: SamplingConfig, max_new_tokens: int, stop_on_eos: bool = True) -> None:
        self.run_id = uuid.uuid4().hex[:12]
        self.lm = lm
        self.prompt = prompt
        self.prompt_ids = prompt_ids
        self.capture = capture
        self.sampling = sampling
        self.max_new_tokens = max_new_tokens
        self.stop_on_eos = stop_on_eos
        self.created = time.time()
        self.lock = threading.RLock()
        self.steps: list[StepRecord] = []
        self.generated: list[int] = []
        self.cache = None
        self.state = "ready"            # ready | running | paused | finished
        self.stop_reason: Optional[str] = None
        self.version = 0
        self.next_event_id = 0
        self.cursor = {"step": 0, "order": 0}
        self.delay_ms = 0
        self._thread: Optional[threading.Thread] = None
        self._gen: Optional[torch.Generator] = None
        self._seed_generator()

    # ------------------------------------------------------------------ helpers
    def _seed_generator(self) -> None:
        self._gen = torch.Generator(device="cpu")
        if self.sampling.seed is not None:
            self._gen.manual_seed(int(self.sampling.seed))
        else:
            self._gen.seed()

    @property
    def adapter(self):
        return self.lm.adapter

    @property
    def tok(self):
        return self.lm.tokenizer

    def all_ids(self) -> list[int]:
        return self.prompt_ids + self.generated

    def step(self, index: Optional[int]) -> StepRecord:
        if not self.steps:
            raise HTTPException(409, "run has no steps yet")
        if index is None or index == -1:
            return self.steps[-1]
        if not 0 <= index < len(self.steps):
            raise HTTPException(404, f"step {index} does not exist (run has {len(self.steps)} steps)")
        return self.steps[index]

    def find_event(self, event_id: int) -> tuple[StepRecord, TraceEvent]:
        for rec in self.steps:
            if rec.events and rec.events[0].event_id <= event_id <= rec.events[-1].event_id:
                return rec, rec.events[event_id - rec.events[0].event_id]
        raise HTTPException(404, f"event {event_id} not found")

    def stored_bytes(self) -> int:
        return sum(r.stored_bytes() for r in self.steps)

    # ------------------------------------------------------------------ execution
    def _forward(self, ids: list[int], phase: str) -> StepRecord:
        past = self.steps[-1].seq_len if self.steps else 0
        x = torch.tensor([ids], dtype=torch.long, device=self.adapter.device)
        with self.lm.lock:
            rec = trace_forward(self.adapter, x, self.cache, step=len(self.steps), phase=phase,  # type: ignore[arg-type]
                                past_len=past, capture=self.capture, start_event_id=self.next_event_id)
        self._sample(rec)
        self.next_event_id = rec.events[-1].event_id + 1
        self.steps.append(rec)
        self._retain()
        self.version += 1
        return rec

    def _sample(self, rec: StepRecord) -> None:
        logits = rec.tensor(None, "next_token_logits")
        dist = distribution(logits, self.sampling)
        tid = select(dist, self.sampling, self._gen)
        em = Emitter(rec, Capture(), self.capture, rec.events[-1].event_id + 1)
        em.emit(stage="sampling", layer=None, op="softmax", name="next_token_probs", tensor=dist.probs_raw,
                provenance="derived", formula="p = softmax(logits)", inputs=("next_token_logits",), force_store=True,
                note="Computed in float32.")
        c = self.sampling
        desc = f"temperature={c.temperature:g}" + (f", top_k={c.top_k}" if c.top_k else "") + (f", top_p={c.top_p:g}" if c.top_p < 1 else "")
        em.emit(stage="sampling", layer=None, op="sampling_distribution", name="sampling_probs", tensor=dist.probs,
                provenance="derived", formula=f"softmax(filter(logits / T))   [{desc}]", inputs=("next_token_logits",),
                force_store=True, note=f"Strategy: {c.strategy}. Greedy decoding takes argmax(logits) and ignores this distribution.")
        order = torch.argsort(dist.logits, descending=True)
        rank = int((order == tid).nonzero()[0, 0])
        top = order[:10].tolist()
        rec.selected_token_id = tid
        rec.sampling = {
            "config": c.model_dump(),
            "token_id": tid,
            "token": token_text(self.tok, tid),
            "logit": float(dist.logits[tid]),
            "prob_raw": float(dist.probs_raw[tid]),
            "prob_final": float(dist.probs[tid]),
            "rank": rank,
            "entropy_raw": entropy(dist.probs_raw),
            "entropy_final": entropy(dist.probs),
            "kept": int(dist.kept.sum()),
            "top": [{"token_id": t, "token": token_text(self.tok, t), "logit": float(dist.logits[t]),
                     "prob_raw": float(dist.probs_raw[t]), "prob_final": float(dist.probs[t])} for t in top],
        }
        self.generated.append(tid)
        eos = self.tok.eos_token_id
        if self.stop_on_eos and eos is not None and tid == eos:
            self.state, self.stop_reason = "finished", "eos"
        elif len(self.generated) >= self.max_new_tokens:
            self.state, self.stop_reason = "finished", "max_new_tokens"

    def _retain(self) -> None:
        keep_from = len(self.steps) - settings.retain_steps
        for rec in self.steps[1:max(1, keep_from)]:
            if rec.tensors and len(rec.tensors) > 1:
                rec.drop_tensors()
        runs.enforce_budget(self)

    def prefill(self) -> StepRecord:
        with self.lock:
            self.cache = new_cache()
            return self._forward(self.prompt_ids, "prefill")

    def next_token(self) -> StepRecord:
        with self.lock:
            if self.state == "finished":
                raise HTTPException(409, f"generation finished ({self.stop_reason}); reset the run to generate again")
            rec = self._forward([self.generated[-1]], "decode")
            self.cursor = {"step": rec.index, "order": 0}
            return rec

    def reset(self) -> None:
        self.pause()
        with self.lock:
            self.steps.clear()
            self.generated.clear()
            self.next_event_id = 0
            self.state, self.stop_reason = "ready", None
            self.cursor = {"step": 0, "order": 0}
            self._seed_generator()
            self.prefill()

    # ------------------------------------------------------------------ operation cursor
    def move_cursor(self, kind: str, step: Optional[int] = None, order: Optional[int] = None) -> None:
        with self.lock:
            s, o = self.cursor["step"], self.cursor["order"]
            if kind == "goto":
                rec = self.step(step if step is not None else s)
                s, o = rec.index, max(0, min(order or 0, len(rec.events) - 1))
            elif kind == "next_op":
                if o + 1 < len(self.steps[s].events):
                    o += 1
                elif s + 1 < len(self.steps):
                    s, o = s + 1, 0
                elif self.state != "finished":
                    self.next_token()
                    return
            elif kind == "prev_op":
                if o > 0:
                    o -= 1
                elif s > 0:
                    s, o = s - 1, len(self.steps[s - 1].events) - 1
            self.cursor = {"step": s, "order": o}
            self.version += 1

    # ------------------------------------------------------------------ auto run
    def resume(self) -> None:
        with self.lock:
            if self.state == "finished":
                raise HTTPException(409, f"generation finished ({self.stop_reason})")
            if self.state == "running":
                return
            self.state = "running"
            self.version += 1
            self._thread = threading.Thread(target=self._loop, daemon=True, name=f"run-{self.run_id}")
            self._thread.start()

    def _loop(self) -> None:
        while True:
            with self.lock:
                if self.state != "running":
                    break
                try:
                    self.next_token()
                except Exception as e:  # surface instead of dying silently
                    self.state, self.stop_reason = "paused", f"error: {e}"
                    self.version += 1
                    break
            if self.delay_ms:
                time.sleep(self.delay_ms / 1e3)
            else:
                time.sleep(0)

    def pause(self) -> None:
        with self.lock:
            if self.state == "running":
                self.state = "paused"
                self.version += 1
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=60)

    # ------------------------------------------------------------------ description
    def token_list(self) -> list[dict]:
        n = len(self.prompt_ids)
        out = []
        for pos, tid in enumerate(self.all_ids()):
            gen = pos >= n
            out.append({"position": pos, "token_id": tid, "token": token_text(self.tok, tid), "generated": gen,
                        # step whose forward pass processed this position (None = selected but not yet fed back)
                        "step": 0 if not gen else (pos - n + 1 if pos - n + 1 < len(self.steps) else None)})
        return out

    def summary(self) -> dict:
        total = failed = 0
        steps = []
        for r in self.steps:
            chk = [e for e in r.events if e.verified is not None]
            bad = sum(1 for e in chk if not e.verified)
            total += len(chk)
            failed += bad
            steps.append({
                "index": r.index, "phase": r.phase, "past_len": r.past_len, "new_len": r.seq_len - r.past_len,
                "seq_len": r.seq_len, "forward_ms": r.forward_ms, "n_events": len(r.events),
                "first_event_id": r.events[0].event_id, "stored_bytes": r.stored_bytes(),
                "has_tensors": len(r.tensors) > 3, "checks": len(chk), "checks_failed": bad,
                "cache_bytes": r.cache_bytes, "sampling": r.sampling,
            })
        return {
            "run_id": self.run_id, "model_id": self.lm.model_id, "state": self.state, "stop_reason": self.stop_reason,
            "version": self.version, "prompt": self.prompt, "tokens": self.token_list(),
            "n_prompt_tokens": len(self.prompt_ids), "generated_text": self.tok.decode(self.generated),
            "text": self.tok.decode(self.all_ids()), "steps": steps, "capture": self.capture.model_dump(),
            "sampling": self.sampling.model_dump(), "max_new_tokens": self.max_new_tokens, "cursor": self.cursor,
            "checks": {"total": total, "failed": failed}, "stored_bytes": self.stored_bytes(),
            "created": self.created,
        }


class RunManager:
    def __init__(self) -> None:
        self.runs: "OrderedDict[str, Run]" = OrderedDict()
        self._lock = threading.Lock()

    def add(self, run: Run) -> None:
        with self._lock:
            self.runs[run.run_id] = run
            while len(self.runs) > settings.max_runs:
                _, old = self.runs.popitem(last=False)
                old.state = "finished"

    def get(self, run_id: str) -> Run:
        run = self.runs.get(run_id)
        if run is None:
            raise HTTPException(404, f"run '{run_id}' not found (it may have been evicted by trace retention)")
        return run

    def delete(self, run_id: str) -> None:
        run = self.runs.pop(run_id, None)
        if run is not None:
            run.pause()

    def drop_model(self, model_id: str) -> None:
        for rid in [r for r, run in self.runs.items() if run.lm.model_id == model_id]:
            self.delete(rid)

    def total_bytes(self) -> int:
        return sum(r.stored_bytes() for r in list(self.runs.values()))

    def enforce_budget(self, current: Run) -> None:
        """Byte budget: release tensors of the oldest runs first, then the oldest decode steps of this run."""
        if self.total_bytes() <= settings.max_store_bytes:
            return
        for run in list(self.runs.values()):
            if run is current:
                continue
            for rec in run.steps:
                rec.drop_tensors()
            if self.total_bytes() <= settings.max_store_bytes:
                return
        for rec in current.steps[1:-1]:
            rec.drop_tensors()
            if self.total_bytes() <= settings.max_store_bytes:
                return

    def usage(self) -> dict:
        return {"runs": len(self.runs), "stored_bytes": self.total_bytes(), "budget_bytes": settings.max_store_bytes,
                "retain_steps": settings.retain_steps, "max_runs": settings.max_runs}


runs = RunManager()


def create_run(lm: LoadedModel, prompt: str, capture: CaptureConfig, sampling: SamplingConfig,
               max_new_tokens: int, stop_on_eos: bool = True) -> Run:
    ids = lm.tokenizer(prompt)["input_ids"]
    if not ids:
        bos = lm.tokenizer.bos_token_id if lm.tokenizer.bos_token_id is not None else lm.tokenizer.eos_token_id
        if bos is None:
            raise HTTPException(422, "prompt produced no tokens")
        ids = [bos]
    limit = min(settings.max_prompt_tokens, (lm.adapter.info()["max_positions"] or 10**9) - max_new_tokens)
    if len(ids) > limit:
        raise HTTPException(422, f"prompt has {len(ids)} tokens; the limit is {limit} (see Settings → max prompt tokens)")
    run = Run(lm, prompt, list(ids), capture, sampling, min(max_new_tokens, settings.max_new_tokens), stop_on_eos)
    run.prefill()
    runs.add(run)
    return run
