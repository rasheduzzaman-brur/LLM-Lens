"use client";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { CheckCircle2, CircleSlash, FlaskConical, Radio, TriangleAlert } from "lucide-react";
import { api, ApiError } from "@/services/api";
import { useStore } from "@/hooks/use-store";
import { Badge, Button, Card, Notice, Select } from "@/components/ui/primitives";
import { cn, fmt, showToken } from "@/lib/utils";
import type { Check, ModelInfo, Provenance, RunSummary, TraceEvent } from "@/types/api";

export function ProvenanceBadge({ p }: { p: Provenance }) {
  if (p === "hooked") return <Badge tone="accent" title="Captured directly from a model module by a forward hook"><Radio className="size-3" />hooked</Badge>;
  if (p === "derived") return <Badge tone="derived" title="Recomputed from hooked tensors with the model's own parameters"><FlaskConical className="size-3" />derived</Badge>;
  if (p === "viz") return <Badge tone="warn" title="Visualization transform, not a model computation">viz transform</Badge>;
  return <Badge tone="bad"><CircleSlash className="size-3" />unavailable</Badge>;
}

export function VerifiedBadge({ check }: { check?: Pick<Check, "verified" | "max_abs_err"> | { verified: boolean | null; max_abs_err: number | null } | null }) {
  if (!check || check.verified === null || check.verified === undefined) return null;
  return check.verified
    ? <Badge tone="good" title="Matches the tensor the model itself produced"><CheckCircle2 className="size-3" />verified · max err {fmt(check.max_abs_err, 2)}</Badge>
    : <Badge tone="bad"><TriangleAlert className="size-3" />mismatch · max err {fmt(check.max_abs_err, 2)}</Badge>;
}

export function VizNote({ children }: { children: React.ReactNode }) {
  return <p className="mt-1.5 flex gap-1.5 text-[11px] leading-4 text-muted"><Badge tone="warn" className="h-fit shrink-0">viz</Badge><span>{children}</span></p>;
}

export function ErrorNotice({ error }: { error: unknown }) {
  if (!error) return null;
  const e = error as ApiError;
  return <Notice tone="bad">{e.status === 409 ? "Not available: " : ""}{e.message}</Notice>;
}

export function Loading({ what = "Loading" }: { what?: string }) { return <div className="py-6 text-center text-xs text-muted">{what}…</div>; }

export function NeedModel() {
  return (
    <Card className="mx-auto mt-10 max-w-md p-5 text-center">
      <h2 className="text-sm font-semibold">No model loaded</h2>
      <p className="mt-1 text-xs text-ink-2">Load a model from a local directory to begin. Nothing is downloaded without your confirmation.</p>
      <Button asChild variant="default" className="mt-3"><Link href="/models">Open Model Explorer</Link></Button>
    </Card>
  );
}
export function NeedRun({ model }: { model: ModelInfo }) {
  return (
    <Card className="mx-auto mt-10 max-w-md p-5 text-center">
      <h2 className="text-sm font-semibold">No inference run yet</h2>
      <p className="mt-1 text-xs text-ink-2">Enter a prompt in the bar above and press <b>Run</b>. {model.name} will execute a real forward pass and every intermediate tensor becomes inspectable here.</p>
    </Card>
  );
}

/** All trace events of one step, with a (layer, name) lookup. */
export function useTrace(runId: string | null, step: number, version: number) {
  const q = useQuery({ queryKey: ["trace", runId, step, version > 0 ? "x" : "x"], queryFn: () => api.trace(runId!, { step, limit: 2000 }), enabled: !!runId, staleTime: Infinity });
  const find = (layer: number | null, name: string): TraceEvent | undefined => q.data?.events.find((e) => e.layer === layer && e.tensor_name === name);
  return { ...q, find };
}

export function TokenStrip({ run, stepIdx, position }: { run: RunSummary; stepIdx: number; position: number }) {
  const set = useStore((s) => s.set);
  const seq = run.steps[stepIdx].seq_len;
  return (
    <div className="flex flex-wrap items-center gap-1">
      {run.tokens.map((t) => {
        const processed = t.position < seq;
        return (
          <button key={t.position} title={`position ${t.position} · id ${t.token_id}${t.generated ? " · generated" : ""}${processed ? "" : " · not yet processed at this step"}`}
            onClick={() => processed && set({ position: t.position })}
            className={cn("num rounded border px-1.5 py-0.5 text-xs transition-colors",
              t.generated ? "border-dashed" : "border-solid",
              t.position === position ? "border-accent bg-accent text-accent-ink" : processed ? "border-line-strong bg-surface-2 text-ink hover:border-accent cursor-pointer" : "border-line text-muted cursor-default")}>
            {showToken(t.token)}
          </button>
        );
      })}
    </div>
  );
}

export function StepSelect({ run, stepIdx }: { run: RunSummary; stepIdx: number }) {
  const set = useStore((s) => s.set);
  return (
    <Select value={stepIdx} onChange={(e) => set({ step: Number(e.target.value), position: -1 })} aria-label="Step">
      {run.steps.map((s) => (
        <option key={s.index} value={s.index}>
          {s.index === 0 ? `step 0 · prefill (${s.seq_len} tokens)` : `step ${s.index} · decode “${showToken(run.tokens[s.seq_len - 1]?.token ?? "")}”`}
        </option>
      ))}
    </Select>
  );
}

export function LayerSelect({ n, value, onChange }: { n: number; value: number; onChange: (v: number) => void }) {
  return <Select value={value} onChange={(e) => onChange(Number(e.target.value))} aria-label="Layer">{Array.from({ length: n }, (_, i) => <option key={i} value={i}>layer {i}</option>)}</Select>;
}
export function HeadSelect({ n, value, onChange }: { n: number; value: number; onChange: (v: number) => void }) {
  return <Select value={value} onChange={(e) => onChange(Number(e.target.value))} aria-label="Head">{Array.from({ length: n }, (_, i) => <option key={i} value={i}>head {i}</option>)}</Select>;
}

/** Clickable tensor name that opens the tensor in the inspector. */
export function TensorChip({ name, layer, step, label }: { name: string; layer: number | null; step: number; label?: string }) {
  const select = useStore((s) => s.select);
  const sel = useStore((s) => s.selection);
  const active = sel?.kind === "tensor" && sel.name === name && sel.layer === layer;
  return (
    <button onClick={() => select({ kind: "tensor", name, layer, step })} title="Inspect in the right panel"
      className={cn("num rounded border px-1.5 py-px text-[11px] cursor-pointer", active ? "border-accent bg-accent-soft text-accent" : "border-line-strong text-ink-2 hover:border-accent hover:text-ink")}>
      {label ?? name}
    </button>
  );
}

export function PageHeader({ title, sub, right }: { title: string; sub?: React.ReactNode; right?: React.ReactNode }) {
  return (
    <div className="mb-3 flex flex-wrap items-end justify-between gap-2">
      <div><h1 className="text-base font-semibold tracking-tight">{title}</h1>{sub && <p className="mt-0.5 max-w-3xl text-xs text-ink-2">{sub}</p>}</div>
      {right && <div className="flex flex-wrap items-center gap-2">{right}</div>}
    </div>
  );
}

/** Label / value rows with a labeled numeric vector preview. */
export function VectorRow({ label, values, onInspect, note }: { label: React.ReactNode; values?: number[]; onInspect?: () => void; note?: React.ReactNode }) {
  if (!values) return null;
  return (
    <div className="grid grid-cols-[minmax(110px,150px)_minmax(0,1fr)] items-center gap-2 border-b border-line py-1 last:border-0">
      <div className="min-w-0 text-xs text-ink-2">{onInspect ? <button onClick={onInspect} className="num text-left text-accent hover:underline cursor-pointer">{label}</button> : label}{note && <div className="text-[11px] text-muted">{note}</div>}</div>
      <div className="num truncate text-[11px] text-ink">[{values.slice(0, 8).map((v) => fmt(v, 4)).join(", ")}{values.length > 8 ? `, … ${values.length} values` : ""}]</div>
    </div>
  );
}
