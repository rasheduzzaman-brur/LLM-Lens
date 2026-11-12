import type * as T from "@/types/api";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(public status: number, message: string, public detail?: unknown) { super(message); }
}

type Params = Record<string, string | number | boolean | null | undefined>;

function qs(params?: Params): string {
  if (!params) return "";
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") u.set(k, String(v));
  const s = u.toString();
  return s ? `?${s}` : "";
}

async function request<R>(method: string, path: string, params?: Params, body?: unknown): Promise<R> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}${qs(params)}`, {
      method, headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined, cache: "no-store",
    });
  } catch {
    throw new ApiError(0, `Cannot reach the LLM Lens backend at ${API_URL}. Is it running?`);
  }
  if (!res.ok) {
    let detail: unknown;
    try { detail = (await res.json()).detail; } catch { /* non-JSON error body */ }
    const msg = typeof detail === "string" ? detail
      : detail && typeof detail === "object" && "message" in detail ? String((detail as { message: unknown }).message)
      : Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : `${res.status} ${res.statusText}`;
    throw new ApiError(res.status, msg, detail);
  }
  return res.json() as Promise<R>;
}

const get = <R,>(p: string, q?: Params) => request<R>("GET", p, q);
const post = <R,>(p: string, b?: unknown) => request<R>("POST", p, undefined, b ?? {});
const run = (id: string) => `/api/inference/${id}`;

export const api = {
  models: () => get<T.ModelsResponse>("/api/models"),
  loadModel: (b: { source: string; device?: string; dtype?: string; allow_download?: boolean }) => post<T.ModelInfo>("/api/models/load", b),
  unloadModel: (id: string) => request<unknown>("DELETE", `/api/models/${id}`),
  architecture: (id: string) => get<T.Architecture>(`/api/models/${id}/architecture`),
  weights: (id: string, layer: number | null) => get<T.WeightList>(`/api/models/${id}/weights`, { layer }),
  weightTensor: (id: string, name: string, layer: number | null, index?: string) => get<T.TensorView>(`/api/models/${id}/weights/tensor`, { name, layer, index }),
  embedding: (id: string, token_id: number) => get<T.EmbeddingResponse>(`/api/models/${id}/embedding`, { token_id }),
  embeddingsPca: (id: string, token_ids: number[]) => post<{ points: number[][]; explained_variance_ratio: number[]; note: string }>(`/api/models/${id}/embeddings/pca`, { token_ids }),
  activation: (id: string) => get<T.ActivationCurve>(`/api/models/${id}/activation`),
  tokenize: (model_id: string, text: string) => post<T.TokenizeResponse>("/api/tokenize", { model_id, text }),
  system: () => get<T.SystemInfo>("/api/system"),
  settings: () => get<T.AppSettings>("/api/settings"),
  patchSettings: (b: Partial<T.AppSettings>) => request<T.AppSettings>("PATCH", "/api/settings", undefined, b),

  runInference: (b: { model_id: string; prompt: string; capture: T.CaptureConfig; sampling: T.SamplingConfig; max_new_tokens: number }) => post<T.RunSummary>("/api/inference/run", b),
  run: (id: string) => get<T.RunSummary>(run(id)),
  step: (id: string, b: { kind: "next_token" | "next_op" | "prev_op" | "goto"; step?: number; order?: number; sampling?: T.SamplingConfig }) => post<T.RunSummary>(`${run(id)}/step`, b),
  pause: (id: string) => post<T.RunSummary>(`${run(id)}/pause`),
  resume: (id: string, b: { delay_ms?: number; sampling?: T.SamplingConfig }) => post<T.RunSummary>(`${run(id)}/resume`, b),
  reset: (id: string) => post<T.RunSummary>(`${run(id)}/reset`),
  trace: (id: string, q: Params) => get<T.TraceList>(`${run(id)}/trace`, q),
  traceEvent: (id: string, eventId: number, index?: string) => get<{ event: T.TraceEvent; tensor: T.TensorView | null; unavailable_reason?: string }>(`${run(id)}/trace/${eventId}`, { index }),
  tensor: (id: string, q: { name: string; step: number; layer?: number | null; index?: string; assembled?: boolean }) => get<T.TensorView>(`${run(id)}/tensor`, q),
  layers: (id: string, step: number) => get<T.LayersSummary>(`${run(id)}/layers`, { step }),
  layer: (id: string, layer: number, position: number, step: number) => get<T.LayerDetail>(`${run(id)}/layers/${layer}`, { position, step }),
  global: (id: string, position: number, step: number) => get<T.LayerDetail>(`${run(id)}/global`, { position, step }),
  attention: (id: string, layer: number, step: number, head?: number) => get<T.AttentionResponse>(`${run(id)}/attention`, { layer, step, head }),
  attentionGrid: (id: string, step: number, reduce: string, head: number) => get<T.AttentionGrid>(`${run(id)}/attention/grid`, { step, reduce, head }),
  attentionRow: (id: string, layer: number, head: number, position: number, step: number) => get<T.AttentionRow>(`${run(id)}/attention/row`, { layer, head, position, step }),
  hiddenStates: (id: string, q: { stage: string; step: number; position: number; depth: number }) => get<T.HiddenStates>(`${run(id)}/hidden-states`, q),
  logits: (id: string, q: Params) => get<T.LogitsResponse>(`${run(id)}/logits`, q),
  kvCache: (id: string, step: number) => get<T.KvCache>(`${run(id)}/kv-cache`, { step }),
  kvTensor: (id: string, q: { layer: number; which: string; head: number; step: number }) => get<T.TensorView>(`${run(id)}/kv-cache/tensor`, q),
  kvCompare: (id: string, step: number) => get<T.KvCompare>(`${run(id)}/kv-cache/compare`, { step }),
  eventsUrl: (id: string) => `${API_URL}${run(id)}/events`,
};
