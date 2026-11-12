"use client";
import { useMemo, useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useStore } from "@/hooks/use-store";
import { useRun, useStepIndex } from "@/hooks/use-lens";
import { ProvenanceBadge, VerifiedBadge, useTrace } from "@/components/common";
import { Badge, Button, Select, Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/primitives";
import { cn, fmt, ms, pct, shapeStr, showToken } from "@/lib/utils";
import type { RunSummary } from "@/types/api";

const STAGE_COLOR: Record<string, string> = {
  embedding: "bg-[#199e70]", norm: "bg-[#c98500]", attention: "bg-[#3987e5]", residual: "bg-[#898781]", mlp: "bg-[#d95926]",
  final_norm: "bg-[#c98500]", lm_head: "bg-[#9085e9]", sampling: "bg-[#d55181]",
};

function Timeline({ run, stepIdx }: { run: RunSummary; stepIdx: number }) {
  const set = useStore((s) => s.set);
  const select = useStore((s) => s.select);
  const sel = useStore((s) => s.selection);
  const trace = useTrace(run.run_id, stepIdx, 0);
  const maxMs = Math.max(...run.steps.map((s) => s.forward_ms));
  return (
    <div className="space-y-2 p-2">
      <div className="flex items-end gap-1 overflow-x-auto pb-1">
        {run.steps.map((s) => (
          <button key={s.index} onClick={() => set({ step: s.index, position: -1 })} title={`step ${s.index} · ${s.phase} · ${s.new_len} new position(s) · forward ${ms(s.forward_ms)}`}
            className={cn("flex shrink-0 flex-col items-center gap-0.5 rounded border px-1.5 pt-1 pb-0.5 cursor-pointer", s.index === stepIdx ? "border-accent bg-accent-soft" : "border-line hover:border-line-strong")}
            style={{ minWidth: s.phase === "prefill" ? 96 : 44 }}>
            <span className="w-full rounded-sm bg-accent/70" style={{ height: 3 + (18 * s.forward_ms) / maxMs }} />
            <span className="num max-w-24 truncate text-[11px] text-ink">{s.phase === "prefill" ? `prefill ×${s.new_len}` : showToken(run.tokens[s.seq_len - 1]?.token ?? "")}</span>
            <span className="num text-[10px] text-muted">{ms(s.forward_ms)}</span>
          </button>
        ))}
        {run.state !== "finished" && <span className="px-2 pb-2 text-[11px] text-muted">next token not yet computed</span>}
      </div>
      <div>
        <div className="mb-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted">
          <span>Operations of step {stepIdx}, in execution order ({trace.data?.total ?? "…"}). Recorded trace of a completed forward pass.</span>
          {Object.entries(STAGE_COLOR).filter(([k]) => k !== "final_norm").map(([k, c]) => <span key={k} className="flex items-center gap-1"><i className={cn("inline-block size-2 rounded-sm", c)} />{k}</span>)}
        </div>
        <div className="flex gap-px overflow-x-auto pb-1">
          {trace.data?.events.map((e) => {
            const active = (sel?.kind === "event" && sel.eventId === e.event_id) || (sel?.kind === "tensor" && sel.name === e.tensor_name && sel.layer === e.layer);
            return <button key={e.event_id} onClick={() => select({ kind: "event", eventId: e.event_id })}
              title={`#${e.order} ${e.layer !== null ? `L${e.layer} ` : ""}${e.operation} → ${e.tensor_name} ${shapeStr(e.shape)}`}
              className={cn("h-5 w-[5px] shrink-0 rounded-[1px] cursor-pointer", STAGE_COLOR[e.stage] ?? "bg-muted", active ? "outline-2 outline-ink" : "opacity-70 hover:opacity-100")} />;
          })}
        </div>
      </div>
    </div>
  );
}

export function TraceTable({ run, stepIdx }: { run: RunSummary; stepIdx: number }) {
  const select = useStore((s) => s.select);
  const sel = useStore((s) => s.selection);
  const trace = useTrace(run.run_id, stepIdx, 0);
  const [layer, setLayer] = useState("all");
  const [stage, setStage] = useState("all");
  const layers = useMemo(() => [...new Set(trace.data?.events.map((e) => e.layer).filter((l) => l !== null))] as number[], [trace.data]);
  const rows = (trace.data?.events ?? []).filter((e) => (layer === "all" || (layer === "global" ? e.layer === null : e.layer === Number(layer))) && (stage === "all" || e.stage === stage));
  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-wrap items-center gap-2 border-b border-line px-2 py-1 text-xs text-ink-2">
        <Select value={layer} onChange={(e) => setLayer(e.target.value)} aria-label="Layer filter"><option value="all">all layers</option><option value="global">global (no layer)</option>{layers.map((l) => <option key={l} value={l}>layer {l}</option>)}</Select>
        <Select value={stage} onChange={(e) => setStage(e.target.value)} aria-label="Stage filter"><option value="all">all stages</option>{trace.data?.stages.map((s) => <option key={s}>{s}</option>)}</Select>
        <span className="text-muted">{rows.length} of {trace.data?.total ?? 0} events · step {stepIdx}</span>
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        <table className="w-full text-left text-xs">
          <thead className="sticky top-0 bg-surface text-[11px] text-muted"><tr>{["#", "stage", "layer", "operation", "tensor", "shape", "dtype", "source", "check", "time"].map((h) => <th key={h} className="border-b border-line px-2 py-1 font-medium">{h}</th>)}</tr></thead>
          <tbody>
            {rows.slice(0, 600).map((e) => (
              <tr key={e.event_id} onClick={() => select({ kind: "event", eventId: e.event_id })}
                className={cn("cursor-pointer border-b border-line/60 hover:bg-surface-2", sel?.kind === "event" && sel.eventId === e.event_id && "bg-accent-soft")}>
                <td className="num px-2 py-0.5 text-muted">{e.event_id}</td><td className="px-2">{e.stage}</td><td className="num px-2">{e.layer ?? "—"}</td>
                <td className="num px-2">{e.operation}</td><td className="num px-2 text-ink">{e.tensor_name}</td><td className="num px-2">{shapeStr(e.shape)}</td>
                <td className="num px-2">{e.dtype}</td><td className="px-2"><ProvenanceBadge p={e.provenance} /></td><td className="px-2"><VerifiedBadge check={e} /></td>
                <td className="num px-2">{ms(e.execution_time_ms)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length > 600 && <p className="p-2 text-[11px] text-muted">Showing the first 600 rows; filter by layer or stage to see the rest.</p>}
      </div>
    </div>
  );
}

function History({ run, stepIdx }: { run: RunSummary; stepIdx: number }) {
  const set = useStore((s) => s.set);
  return (
    <div className="h-full overflow-auto">
      <table className="w-full text-left text-xs">
        <thead className="sticky top-0 bg-surface text-[11px] text-muted"><tr>{["step", "phase", "context length", "selected token", "id", "probability", "rank", "entropy (nats)", "strategy", "forward"].map((h) => <th key={h} className="border-b border-line px-2 py-1 font-medium">{h}</th>)}</tr></thead>
        <tbody>
          {run.steps.map((s) => (
            <tr key={s.index} onClick={() => set({ step: s.index, position: -1 })} className={cn("cursor-pointer border-b border-line/60 hover:bg-surface-2", s.index === stepIdx && "bg-accent-soft")}>
              <td className="num px-2 py-0.5">{s.index}</td><td className="px-2">{s.phase}</td><td className="num px-2">{s.seq_len}</td>
              <td className="num px-2 text-ink">“{showToken(s.sampling.token)}”</td><td className="num px-2">{s.sampling.token_id}</td>
              <td className="num px-2">{pct(s.sampling.prob_final, 2)}</td><td className="num px-2">{s.sampling.rank + 1}</td><td className="num px-2">{fmt(s.sampling.entropy_final, 3)}</td>
              <td className="px-2">{s.sampling.config.strategy}{s.sampling.config.strategy === "sample" ? ` · T=${s.sampling.config.temperature}` : ""}</td><td className="num px-2">{ms(s.forward_ms)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function BottomPanel() {
  const run = useRun();
  const stepIdx = useStepIndex(run);
  const open = useStore((s) => s.bottomOpen);
  const set = useStore((s) => s.set);
  const [tab, setTab] = useState("timeline");
  if (!run) return null;
  return (
    <Tabs value={tab} onValueChange={(v) => { setTab(v); set({ bottomOpen: true }); }} className={cn("flex shrink-0 flex-col border-t border-line bg-surface", open ? "h-52 max-sm:h-40" : "h-9")}>
      <div className="flex h-9 shrink-0 items-center justify-between gap-2 border-b border-line px-2">
        <TabsList className="min-w-0 flex-nowrap overflow-x-auto border-0 bg-transparent p-0 [&>button]:whitespace-nowrap"><TabsTrigger value="timeline">Execution timeline</TabsTrigger><TabsTrigger value="trace">Trace events</TabsTrigger><TabsTrigger value="history">Token generation history</TabsTrigger></TabsList>
        <div className="flex shrink-0 items-center gap-2">
          <Badge className="max-sm:hidden" tone={run.checks.failed ? "bad" : "good"}>{run.checks.total - run.checks.failed}/{run.checks.total} checks verified</Badge>
          <Button size="icon" variant="ghost" onClick={() => set({ bottomOpen: !open })} aria-label="Toggle bottom panel">{open ? <ChevronDown /> : <ChevronUp />}</Button>
        </div>
      </div>
      {open && (
        <div className="min-h-0 flex-1 overflow-auto">
          <TabsContent value="timeline"><Timeline run={run} stepIdx={stepIdx} /></TabsContent>
          <TabsContent value="trace" className="h-full"><TraceTable run={run} stepIdx={stepIdx} /></TabsContent>
          <TabsContent value="history" className="h-full"><History run={run} stepIdx={stepIdx} /></TabsContent>
        </div>
      )}
    </Tabs>
  );
}
