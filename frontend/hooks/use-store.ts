"use client";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { CaptureConfig, SamplingConfig } from "@/types/api";

export type Selection =
  | { kind: "tensor"; name: string; layer: number | null; step: number; index?: string }
  | { kind: "weight"; name: string; layer: number | null }
  | { kind: "event"; eventId: number }
  | null;

interface State {
  modelId: string | null; runId: string | null; prompt: string; step: number; layer: number; head: number; position: number;
  sampling: SamplingConfig; capture: CaptureConfig; maxNewTokens: number; selection: Selection; theme: "dark" | "light";
  inspectorOpen: boolean; bottomOpen: boolean;
  set: (p: Partial<State>) => void; select: (s: Selection) => void;
}

export const useStore = create<State>()(
  persist(
    (set) => ({
      modelId: null, runId: null, prompt: "The capital of France is", step: -1, layer: 0, head: 0, position: -1,
      sampling: { strategy: "greedy", temperature: 1, top_k: 0, top_p: 1, seed: 0 },
      capture: { mode: "full", tensors: [], layers: [] }, maxNewTokens: 24, selection: null, theme: "dark",
      inspectorOpen: true, bottomOpen: true,
      set: (p) => set(p),
      select: (selection) => set({ selection, inspectorOpen: true }),
    }),
    { name: "llm-lens", partialize: (s) => ({ modelId: s.modelId, runId: s.runId, prompt: s.prompt, sampling: s.sampling, capture: s.capture, maxNewTokens: s.maxNewTokens, theme: s.theme, inspectorOpen: s.inspectorOpen, bottomOpen: s.bottomOpen }) },
  ),
);
