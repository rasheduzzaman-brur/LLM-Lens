"use client";
import { useEffect } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "@/services/api";
import { useStore } from "@/hooks/use-store";
import { DARK, LIGHT } from "@/lib/colors";
import type { RunSummary } from "@/types/api";

export function usePalette() { return useStore((s) => s.theme) === "dark" ? DARK : LIGHT; }

export function useModels() { return useQuery({ queryKey: ["models"], queryFn: api.models, refetchInterval: 15000 }); }

/** Currently selected model, if it is still loaded in the backend. */
export function useModel() {
  const modelId = useStore((s) => s.modelId);
  const set = useStore((s) => s.set);
  const { data } = useModels();
  // adopt a model the backend already has loaded (new browser, or the active one was unloaded)
  const fallback = data && !data.loaded.some((m) => m.model_id === modelId) ? data.loaded[0]?.model_id : undefined;
  useEffect(() => { if (fallback) set({ modelId: fallback, runId: null, selection: null }); }, [fallback, set]);
  return data?.loaded.find((m) => m.model_id === modelId) ?? null;
}

export function useRun() {
  const runId = useStore((s) => s.runId);
  const set = useStore((s) => s.set);
  const q = useQuery({
    queryKey: ["run", runId], queryFn: () => api.run(runId!), enabled: !!runId, retry: false, staleTime: Infinity,
  });
  useEffect(() => { if (q.error instanceof ApiError && q.error.status === 404) set({ runId: null, selection: null }); }, [q.error, set]);
  return q.data && q.data.run_id === runId ? q.data : null;
}

/** Selected step resolved against the run (-1 = latest). */
export function useStepIndex(run: RunSummary | null): number {
  const step = useStore((s) => s.step);
  if (!run) return 0;
  return step < 0 || step >= run.steps.length ? run.steps.length - 1 : step;
}
export function usePosition(run: RunSummary | null, stepIdx: number): number {
  const position = useStore((s) => s.position);
  if (!run) return 0;
  const max = run.steps[stepIdx].seq_len - 1;
  return position < 0 || position > max ? max : position;
}

export function useSetRun() {
  const qc = useQueryClient();
  return (r: RunSummary) => qc.setQueryData(["run", r.run_id], r);
}

export function useStartRun() {
  const set = useStore((s) => s.set);
  const setRun = useSetRun();
  return useMutation({
    mutationFn: () => {
      const s = useStore.getState();
      if (!s.modelId) throw new ApiError(0, "Load a model first (Model Explorer).");
      return api.runInference({ model_id: s.modelId, prompt: s.prompt, capture: s.capture, sampling: s.sampling, max_new_tokens: s.maxNewTokens });
    },
    onSuccess: (r) => { setRun(r); set({ runId: r.run_id, step: -1, position: -1, selection: null }); },
  });
}

/** Live run updates over Server-Sent Events while a run exists. */
export function useRunStream() {
  const runId = useStore((s) => s.runId);
  const qc = useQueryClient();
  useEffect(() => {
    if (!runId) return;
    const es = new EventSource(api.eventsUrl(runId));
    es.addEventListener("run", (e) => {
      const r = JSON.parse((e as MessageEvent).data) as RunSummary;
      const prev = qc.getQueryData<RunSummary>(["run", runId]);
      if (!prev || prev.version <= r.version) qc.setQueryData(["run", runId], r);
    });
    es.addEventListener("gone", () => es.close());
    es.onerror = () => { /* EventSource reconnects by itself */ };
    return () => es.close();
  }, [runId, qc]);
}
