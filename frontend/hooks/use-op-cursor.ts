"use client";
import { useStore } from "@/hooks/use-store";
import { useTrace } from "@/components/common";
import type { RunSummary, TraceEvent } from "@/types/api";

/** Walk the recorded operations of the selected step. The current operation is the inspector selection. */
export function useOpCursor(run: RunSummary | null, stepIdx: number) {
  const sel = useStore((s) => s.selection);
  const set = useStore((s) => s.set);
  const trace = useTrace(run?.run_id ?? null, stepIdx, 0);
  const events = trace.data?.events ?? [];
  let idx = -1;
  if (sel?.kind === "event") idx = events.findIndex((e) => e.event_id === sel.eventId);
  else if (sel?.kind === "tensor") idx = events.findIndex((e) => e.layer === sel.layer && e.tensor_name === sel.name);
  const current: TraceEvent | undefined = idx >= 0 ? events[idx] : undefined;
  const go = (i: number) => {
    const e = events[Math.max(0, Math.min(events.length - 1, i))];
    if (!e) return;
    set({ selection: { kind: "event", eventId: e.event_id }, ...(e.layer !== null ? { layer: e.layer } : {}) });
  };
  return { events, current, index: idx, total: events.length, next: () => go(idx + 1), prev: () => go(idx <= 0 ? 0 : idx - 1), first: () => go(0), atEnd: idx >= events.length - 1 };
}
