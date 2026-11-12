"use client";
import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { X } from "lucide-react";
import { api } from "@/services/api";
import { useStore } from "@/hooks/use-store";
import { useModel, useRun, useStepIndex } from "@/hooks/use-lens";
import { ErrorNotice, Loading, ProvenanceBadge, VerifiedBadge, useTrace } from "@/components/common";
import { Heatmap, VectorHeat } from "@/components/tensors/heatmap";
import { Badge, Button, Formula, Input, Notice } from "@/components/ui/primitives";
import { bytes, fmt, ms, shapeStr } from "@/lib/utils";
import type { Matrix, TensorView, TraceEvent } from "@/types/api";

function defaultIndex(shape: number[], head: number, seqDim: boolean): string {
  if (shape.length <= 2) return "";
  if (shape.length === 3) return "0";
  if (shape.length === 4) return `0,${Math.min(head, shape[1] - 1)}`;
  return shape.slice(0, -2).map(() => 0).join(",") + (seqDim ? "" : "");
}

export function TensorValues({ view, tokens }: { view: TensorView; tokens?: string[] }) {
  const v = view.values;
  const shape = view.pooled ? view.pooled_shape! : view.shape;
  if (shape.length === 0) return <div className="num text-lg font-semibold">{typeof v === "boolean" ? String(v) : fmt(v as number, 6)}</div>;
  if (shape.length === 1) {
    const arr = (v as (number | boolean | null)[]).map((x) => (typeof x === "boolean" ? Number(x) : (x ?? NaN)));
    return (
      <div className="space-y-2">
        <VectorHeat values={arr} wrapAt={arr.length > 512 ? 96 : arr.length > 96 ? 48 : arr.length} />
        <ValueTable values={arr} />
      </div>
    );
  }
  if (shape.length === 2) {
    const m = (v as (number | boolean | null)[][]).map((r) => r.map((x) => (typeof x === "boolean" ? Number(x) : x))) as Matrix;
    const labels = !view.pooled && tokens && tokens.length >= shape[0] ? tokens.slice(0, shape[0]) : undefined;
    const colLabels = !view.pooled && tokens && shape[1] <= tokens.length && shape[1] === shape[0] ? tokens.slice(0, shape[1]) : undefined;
    return <Heatmap data={m} rowLabels={labels} colLabels={colLabels} maxHeight={340} maxCell={20} />;
  }
  return <Notice>The selected slice is {shape.length}-dimensional. Add index components (e.g. <span className="num">0,0</span>) to reduce it to a matrix or vector.</Notice>;
}

export function ValueTable({ values, page = 48 }: { values: number[]; page?: number }) {
  const [off, setOff] = useState(0);
  useEffect(() => setOff(0), [values]);
  const slice = values.slice(off, off + page);
  return (
    <div>
      <div className="num grid grid-cols-4 gap-x-2 gap-y-px text-[11px]">
        {slice.map((v, i) => <div key={i} className="flex justify-between gap-1 border-b border-line/60"><span className="text-muted">{off + i}</span><span className="text-ink">{fmt(v, 5)}</span></div>)}
      </div>
      {values.length > page && (
        <div className="mt-1.5 flex items-center justify-between text-[11px] text-muted">
          <Button size="sm" variant="ghost" disabled={off === 0} onClick={() => setOff(Math.max(0, off - page))}>previous</Button>
          <span className="num">{off}–{Math.min(values.length, off + page) - 1} of {values.length}</span>
          <Button size="sm" variant="ghost" disabled={off + page >= values.length} onClick={() => setOff(off + page)}>next</Button>
        </div>
      )}
    </div>
  );
}

function EventMeta({ ev }: { ev: TraceEvent }) {
  const select = useStore((s) => s.select);
  const rows: [string, React.ReactNode][] = [
    ["operation", <span key="o" className="num">{ev.operation}</span>],
    ["stage", <>{ev.stage}{ev.layer !== null ? ` · layer ${ev.layer}` : ""} · step {ev.step} ({ev.phase})</>],
    ["shape", <span key="s" className="num">{shapeStr(ev.shape)} · {ev.dtype}</span>],
    ["size", <span key="b" className="num">{bytes(ev.nbytes)}{ev.stored ? "" : " (not stored)"}</span>],
  ];
  if (ev.module_path) rows.push(["module", <span key="m" className="num break-all">{ev.module_path}</span>]);
  if (ev.execution_time_ms != null) rows.push(["module time", <span key="t" className="num">{ms(ev.execution_time_ms)}</span>]);
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1"><ProvenanceBadge p={ev.provenance} /><VerifiedBadge check={ev} /></div>
      {ev.formula && <Formula>{ev.formula}</Formula>}
      <dl className="grid grid-cols-[84px_1fr] gap-x-2 gap-y-1 text-xs">
        {rows.map(([k, v]) => <div key={k} className="contents"><dt className="text-muted">{k}</dt><dd className="min-w-0 text-ink">{v}</dd></div>)}
      </dl>
      {ev.inputs.length > 0 && (
        <div className="flex flex-wrap items-center gap-1 text-xs text-muted">inputs
          {ev.inputs.map((n) => <button key={n} onClick={() => select({ kind: "tensor", name: n, layer: n.startsWith("rope_") || ["tok_emb", "pos_emb", "final_norm_in", "final_norm_out", "logits", "next_token_logits"].includes(n) ? null : ev.layer, step: ev.step })}
            className="num rounded border border-line-strong px-1.5 py-px text-[11px] text-ink-2 hover:border-accent cursor-pointer">{n}</button>)}
        </div>
      )}
      {ev.verified_against && <p className="text-[11px] leading-4 text-muted">Checked against: {ev.verified_against}</p>}
      {ev.note && <Notice>{ev.note}</Notice>}
      {ev.stats && ev.stats.min !== null && (
        <div className="num grid grid-cols-5 gap-1 text-[11px]">
          {(["min", "max", "mean", "std", "l2"] as const).map((k) => <div key={k} className="rounded border border-line px-1.5 py-1"><div className="text-muted">{k}</div><div className="text-ink">{fmt(ev.stats![k], 4)}</div></div>)}
        </div>
      )}
    </div>
  );
}

export function Inspector() {
  const sel = useStore((s) => s.selection);
  const head = useStore((s) => s.head);
  const set = useStore((s) => s.set);
  const model = useModel();
  const run = useRun();
  const stepIdx = useStepIndex(run);
  const selStep = sel?.kind === "tensor" ? Math.min(sel.step < 0 ? stepIdx : sel.step, (run?.steps.length ?? 1) - 1) : stepIdx;
  const trace = useTrace(run?.run_id ?? null, selStep, 0);
  const [slice, setSlice] = useState<{ key: string; index: string; draft: string }>({ key: "", index: "", draft: "" });

  const ev = sel?.kind === "tensor" ? trace.find(sel.layer, sel.name) : sel?.kind === "event" ? trace.data?.events.find((e) => e.event_id === sel.eventId) : undefined;
  const auto = sel?.kind === "tensor" ? (sel.index ?? (ev ? defaultIndex(ev.shape, head, true) : null))
    : sel?.kind === "event" ? (ev ? defaultIndex(ev.shape, head, true) : "") : sel?.kind === "weight" ? "" : null;
  // the manual slice only applies to the selection it was typed for; otherwise the default slice is used
  const key = sel ? JSON.stringify(sel) + (auto ?? "") : "";
  const index = slice.key === key ? slice.index : auto ?? "";
  const draft = slice.key === key ? slice.draft : auto ?? "";
  const setDraft = (d: string) => setSlice({ key, index, draft: d });
  const setIndex = (i: string) => setSlice({ key, index: i, draft: i });

  const tq = useQuery({
    queryKey: ["inspect", run?.run_id, model?.model_id, sel, selStep, index],
    enabled: !!sel && auto !== null && ((sel.kind === "tensor" && !!run && !!ev && ev.stored) || (sel.kind === "weight" && !!model) || (sel.kind === "event" && !!run && (!ev || ev.stored))),
    staleTime: Infinity, retry: false,
    queryFn: async (): Promise<{ view: TensorView | null; event?: TraceEvent; reason?: string }> => {
      if (sel!.kind === "tensor") return { view: await api.tensor(run!.run_id, { name: sel!.name, layer: sel!.layer, step: selStep, index }) };
      if (sel!.kind === "weight") return { view: await api.weightTensor(model!.model_id, sel!.name, sel!.layer, index) };
      const r = await api.traceEvent(run!.run_id, (sel as { eventId: number }).eventId, index || undefined);
      return { view: r.tensor, event: r.event, reason: r.unavailable_reason };
    },
  });

  const event = ev ?? tq.data?.event;
  const title = !sel ? "Inspector" : sel.kind === "tensor" ? sel.name : sel.kind === "weight" ? sel.name : event?.tensor_name ?? `event ${sel.eventId}`;
  const tokens = run?.tokens.map((t) => t.token);

  return (
    <aside className="flex h-full w-full flex-col border-l border-line bg-surface">
      <div className="flex items-center justify-between border-b border-line px-3 py-2">
        <div className="min-w-0"><div className="text-[11px] text-muted">Inspector</div><div className="num truncate text-sm font-semibold">{title}</div></div>
        <Button size="icon" variant="ghost" onClick={() => set({ inspectorOpen: false })} aria-label="Close inspector"><X /></Button>
      </div>
      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
        {!sel && <p className="text-xs leading-relaxed text-ink-2">Select a component in the architecture diagram, a tensor name, or a trace event. Its formula, shape, provenance, verification status and real values appear here.</p>}
        {sel?.kind === "weight" && (
          <div className="flex flex-wrap gap-1"><Badge tone="accent">model parameter</Badge>{sel.layer !== null && <Badge>layer {sel.layer}</Badge>}<Badge>convention y = x·W + b</Badge></div>
        )}
        {sel?.kind === "tensor" && trace.isLoading && <Loading />}
        {sel?.kind === "tensor" && trace.data && !ev && <Notice tone="warn">This architecture has no tensor “{sel.name}”{sel.layer !== null ? ` in layer ${sel.layer}` : ""} at step {selStep}.</Notice>}
        {event && <EventMeta ev={event} />}
        {event && !event.stored && sel?.kind === "tensor" && <Notice tone="warn">Values are not stored for this event ({run?.capture.mode} capture or released by retention). Shape and statistics above are still from the real run.</Notice>}
        {tq.data?.reason && !tq.data.view && <Notice tone="warn">{tq.data.reason}</Notice>}
        {sel && (
          <form className="flex items-center gap-1.5" onSubmit={(e) => { e.preventDefault(); setIndex(draft.trim()); }}>
            <span className="text-xs text-muted">index</span>
            <Input className="num h-7" value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="e.g. 0,3,:,0:64" aria-label="Tensor index" />
            <Button size="sm" type="submit">slice</Button>
          </form>
        )}
        {tq.isFetching && <Loading what="Fetching values" />}
        <ErrorNotice error={tq.error} />
        {tq.data?.view && !tq.isFetching && (
          <div className="space-y-2">
            <div className="num text-[11px] text-muted">
              {shapeStr(tq.data.view.full_shape)}{tq.data.view.index ? `[${tq.data.view.index}]` : ""} → {shapeStr(tq.data.view.shape)} · {tq.data.view.dtype}
            </div>
            {tq.data.view.pooled && <Notice tone="warn">{tq.data.view.pool_note}</Notice>}
            {tq.data.view.serialization_note && <Notice>{tq.data.view.serialization_note}</Notice>}
            <TensorValues view={tq.data.view} tokens={sel?.kind === "weight" ? undefined : tokens} />
            {tq.data.view.stats && tq.data.view.stats.min !== null && (
              <div className="num text-[11px] text-muted">slice: min {fmt(tq.data.view.stats.min)} · max {fmt(tq.data.view.stats.max)} · mean {fmt(tq.data.view.stats.mean)} · std {fmt(tq.data.view.stats.std)}</div>
            )}
          </div>
        )}
      </div>
    </aside>
  );
}
