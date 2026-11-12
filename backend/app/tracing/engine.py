"""Computation trace engine: runs one real forward pass under hooks and turns the
captured tensors into an ordered list of standardized trace events."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch

from app.instrumentation.hooks import Capture, HookManager
from app.services.tensor_view import nbytes, stats
from app.tracing.schema import CaptureConfig, Phase, TraceEvent

if TYPE_CHECKING:  # pragma: no cover
    from app.adapters.base import BaseModelAdapter

# logits [1, T, V] can be very large; above this only the last position is kept
MAX_FULL_LOGITS_BYTES = 48_000_000


@dataclass
class StepContext:
    step: int
    phase: Phase
    past_len: int
    new_len: int
    cache: object

    @property
    def total_len(self) -> int:
        return self.past_len + self.new_len


@dataclass
class StepRecord:
    index: int
    phase: Phase
    input_ids: list[int]
    past_len: int
    seq_len: int
    forward_ms: float = 0.0
    events: list[TraceEvent] = field(default_factory=list)
    tensors: dict[int, torch.Tensor] = field(default_factory=dict)
    by_name: dict[tuple[Optional[int], str], int] = field(default_factory=dict)
    sampling: Optional[dict] = None
    selected_token_id: Optional[int] = None
    cache_layers: list[dict] = field(default_factory=list)
    cache_bytes: int = 0

    def event(self, layer: Optional[int], name: str) -> Optional[TraceEvent]:
        eid = self.by_name.get((layer, name))
        if eid is None:
            return None
        return self.events[eid - self.events[0].event_id]

    def tensor(self, layer: Optional[int], name: str) -> Optional[torch.Tensor]:
        eid = self.by_name.get((layer, name))
        return None if eid is None else self.tensors.get(eid)

    def stored_bytes(self) -> int:
        return sum(nbytes(t) for t in self.tensors.values())

    def drop_tensors(self, keep: tuple[str, ...] = ("next_token_logits",)) -> None:
        """Retention: turn this step into metadata-only (cheap vocabulary logits stay)."""
        for ev in self.events:
            if ev.stored and ev.tensor_name not in keep:
                self.tensors.pop(ev.event_id, None)
                ev.stored = False
                ev.note = ((ev.note + " ") if ev.note else "") + "Tensor released by trace retention policy."


class Emitter:
    """Appends events in execution order and stores tensors according to the capture policy."""

    def __init__(self, rec: StepRecord, cap: Capture, capture: CaptureConfig, start_id: int) -> None:
        self.rec = rec
        self.cap = cap
        self.capture = capture
        self.next_id = start_id

    def emit(
        self,
        *,
        stage: str,
        layer: Optional[int],
        op: str,
        name: str,
        tensor: Optional[torch.Tensor],
        provenance: str = "hooked",
        formula: Optional[str] = None,
        inputs: tuple[str, ...] = (),
        check: Optional[tuple[bool, float]] = None,
        against: Optional[str] = None,
        note: Optional[str] = None,
        raw_key: Optional[str] = None,
        force_store: bool = False,
        never_store: bool = False,
    ) -> TraceEvent:
        key = (layer, raw_key or name)
        ev = TraceEvent(
            event_id=self.next_id,
            step=self.rec.index,
            phase=self.rec.phase,
            order=len(self.rec.events),
            stage=stage,
            layer=layer,
            operation=op,
            tensor_name=name,
            shape=list(tensor.shape) if tensor is not None else [],
            dtype=str(tensor.dtype).replace("torch.", "") if tensor is not None else "n/a",
            provenance=provenance if tensor is not None else "unavailable",  # type: ignore[arg-type]
            formula=formula,
            inputs=list(inputs),
            module_path=self.cap.module_path.get(key) if provenance == "hooked" else None,
            execution_time_ms=self.cap.time_ms.get(key) if provenance == "hooked" else None,
            note=note,
        )
        if check is not None:
            ev.verified, ev.max_abs_err = bool(check[0]), float(check[1])
            ev.verified_against = against
        if tensor is not None:
            ev.stats = stats(tensor)
            ev.nbytes = nbytes(tensor)
            if not never_store and (force_store or self.capture.wants(layer, name)):
                self.rec.tensors[ev.event_id] = tensor
                ev.stored = True
        self.rec.events.append(ev)
        self.rec.by_name[(layer, name)] = ev.event_id
        self.next_id += 1
        return ev


def cache_kv(cache, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Key / value tensors of one layer from a transformers Cache object."""
    if hasattr(cache, "layers"):
        lay = cache.layers[layer]
        return lay.keys, lay.values
    if hasattr(cache, "key_cache"):  # transformers < 4.54
        return cache.key_cache[layer], cache.value_cache[layer]
    k, v = cache[layer][:2]
    return k, v


def new_cache():
    from transformers import DynamicCache

    return DynamicCache()


@torch.inference_mode()
def trace_forward(
    adapter: "BaseModelAdapter",
    input_ids: torch.Tensor,
    cache,
    *,
    step: int,
    phase: Phase,
    past_len: int,
    capture: CaptureConfig,
    start_event_id: int,
) -> StepRecord:
    """One instrumented forward pass. `cache` is updated in place by the model."""
    cap = Capture()
    hm = HookManager(cap, adapter.module_names)
    rec = StepRecord(
        index=step,
        phase=phase,
        input_ids=input_ids[0].tolist(),
        past_len=past_len,
        seq_len=past_len + input_ids.shape[1],
    )
    adapter.register_hooks(hm)
    try:
        t0 = time.perf_counter()
        adapter.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        rec.forward_ms = (time.perf_counter() - t0) * 1e3
    finally:
        hm.remove()
    ctx = StepContext(step=step, phase=phase, past_len=past_len, new_len=input_ids.shape[1], cache=cache)
    em = Emitter(rec, cap, capture, start_event_id)
    adapter.emit_step(cap, em, ctx)
    # KV cache bookkeeping (metadata only; tensors stay inside the live cache object)
    total = 0
    for i in range(adapter.n_layers):
        k, v = cache_kv(cache, i)
        b = nbytes(k) + nbytes(v)
        total += b
        rec.cache_layers.append(
            {"layer": i, "key_shape": list(k.shape), "value_shape": list(v.shape), "dtype": str(k.dtype).replace("torch.", ""), "bytes": b}
        )
    rec.cache_bytes = total
    return rec
