export type Provenance = "hooked" | "derived" | "viz" | "unavailable";
export type Matrix = (number | null)[][];

export interface TensorStats { min: number | null; max: number | null; mean: number | null; std: number | null; l2: number | null }

export interface TraceEvent {
  event_id: number; step: number; phase: "prefill" | "decode"; order: number; stage: string; layer: number | null;
  operation: string; tensor_name: string; shape: number[]; dtype: string; provenance: Provenance;
  verified: boolean | null; max_abs_err: number | null; verified_against: string | null; formula: string | null;
  inputs: string[]; module_path: string | null; execution_time_ms: number | null; stats: TensorStats | null;
  stored: boolean; nbytes: number; note: string | null;
}

export interface TensorView {
  full_shape: number[]; dtype: string; index: string; shape: number[]; pooled: boolean; pooled_shape?: number[];
  pool_note?: string; serialization_note?: string; stats?: TensorStats; values: unknown;
  name?: string; layer?: number | null; step?: number; assembled?: boolean; event?: TraceEvent | null; tokens?: string[];
  new_from?: number; note?: string;
}

export interface Capabilities {
  norm: "layernorm" | "rmsnorm"; norm_placement: string; positional: string; mlp: "standard" | "swiglu" | "gated";
  activation: string; fused_qkv: boolean; qkv_bias: boolean; grouped_query_attention: boolean; rope: boolean; mask_mode: string;
}

export interface ModelInfo {
  model_id: string; name: string; source: string; path: string; family: string; model_type: string; model_class: string;
  n_layers: number; n_heads: number; n_kv_heads: number; head_dim: number; hidden_size: number; intermediate_size: number;
  vocab_size: number; max_positions: number; parameter_count: number; dtype: string; device: string;
  attn_implementation: string | null; capabilities: Capabilities; precision_note: string | null;
  tokenizer: { class: string; vocab_size: number; bos_token: string | null; eos_token: string | null; pad_token: string | null; is_fast: boolean };
}

export interface LocalModel { name: string; path: string; model_type: string; supported: boolean; size_bytes: number; loaded: boolean; has_weights: boolean }
export interface ModelsResponse { loaded: ModelInfo[]; local: LocalModel[]; supported_model_types: string[]; suggested: { repo_id: string; note: string }[] }

export interface ArchNode {
  id: string; kind: "io" | "embedding" | "layer" | "norm" | "head" | "op"; label: string; shape: string; tensors: string[];
  layer?: number; stage?: string; formula?: string; module: string | null; params: Record<string, number[]>; order: number; note?: string | null;
}
export interface LayerOp { key: string; label: string; stage: string; module: string | null; in: string[]; out: string[]; detail: string[]; shape: string; formula: string }
export interface Architecture { nodes: ArchNode[]; layer_template: LayerOp[]; info: ModelInfo }

export interface SamplingConfig { strategy: "greedy" | "sample"; temperature: number; top_k: number; top_p: number; seed: number | null }
export interface CaptureConfig { mode: "metadata" | "selected" | "full"; tensors: string[]; layers: number[] }

export interface TopToken { token_id: number; token: string; logit: number; prob_raw: number; prob_final: number }
export interface StepSampling {
  config: SamplingConfig; token_id: number; token: string; logit: number; prob_raw: number; prob_final: number; rank: number;
  entropy_raw: number; entropy_final: number; kept: number; top: TopToken[];
}
export interface StepSummary {
  index: number; phase: "prefill" | "decode"; past_len: number; new_len: number; seq_len: number; forward_ms: number;
  n_events: number; first_event_id: number; stored_bytes: number; has_tensors: boolean; checks: number; checks_failed: number;
  cache_bytes: number; sampling: StepSampling;
}
export interface RunToken { position: number; token_id: number; token: string; generated: boolean; step: number | null }
export interface RunSummary {
  run_id: string; model_id: string; state: "ready" | "running" | "paused" | "finished"; stop_reason: string | null; version: number;
  prompt: string; tokens: RunToken[]; n_prompt_tokens: number; generated_text: string; text: string; steps: StepSummary[];
  capture: CaptureConfig; sampling: SamplingConfig; max_new_tokens: number; cursor: { step: number; order: number };
  checks: { total: number; failed: number }; stored_bytes: number; created: number;
}

export interface TraceList { run_id: string; step: number; phase: string; total: number; offset: number; limit: number; forward_ms: number; stages: string[]; events: TraceEvent[] }

export interface AttentionResponse {
  layer: number; step: number; n_heads: number; seq_len: number; head: number | null; tokens: string[]; weights: Matrix[];
  entropy: number[][]; n_prompt_tokens: number; verified: boolean; max_abs_err: number; verified_against: string; note: string;
}
export interface AttentionGrid { step: number; reduce: string; head: number; seq_len: number; tokens: string[]; matrices: Matrix[]; note: string | null }
export interface AttentionRow {
  layer: number; head: number; position: number; src_step: number; phase: string; token: string; tokens: string[]; d_k: number;
  scale: number; scores_raw: number[]; scores: number[]; allowed: boolean[]; shift: number; exp: number[]; denominator: number;
  weights: number[]; weights_sum: number; recompute_max_err: number; query: number[]; head_output: number[]; exp_note: string;
}

export interface Check { verified: boolean; max_abs_err: number; against: string | null }
export interface NormParams { type: string; eps: number | null; gamma?: number[]; beta?: number[] }
export interface LayerDetail {
  layer?: number; position: number; src_step: number; phase: string; token: string; token_id?: number;
  vectors: Record<string, number[]>; scalars: Record<string, number>; checks: Record<string, Check>;
  norm: Record<string, NormParams>; capabilities: Capabilities; events: TraceEvent[];
}
export interface LayersSummary {
  step: number; phase: string; position: number; note: string;
  layers: { layer: number; n_events: number; checks: number; checks_failed: number; max_abs_err: number; time_ms: number | null; norms: Record<string, number> }[];
}

export interface HiddenStates {
  stage: string; stages: Record<string, string>; step: number; depth_labels: string[]; tokens: string[]; hidden_size: number;
  position: number; depth: number; norms: number[][]; pca: { points: number[][][]; explained_variance_ratio: number[]; note: string };
  token_similarity: number[][]; depth_similarity: number[][]; consecutive_similarity: number[]; similarity_note: string;
}

export interface LogitRow { rank: number; token_id: number; token: string; logit: number; prob_raw: number; prob_temperature: number; prob_final: number; kept: boolean }
export interface LogitsResponse {
  step: number; phase: string; position: number; context_token: string; vocab_size: number; config: SamplingConfig; what_if: boolean;
  top: LogitRow[]; kept: number; entropy_raw: number; entropy_final: number; greedy_token_id: number; logit_stats: TensorStats;
  selected: StepSampling & { prob_under_view: number; kept_under_view: boolean }; dtype: string; note: string;
}

export interface KvLayer { layer: number; key_shape: number[]; value_shape: number[]; dtype: string; bytes: number }
export interface KvHistory { step: number; phase: string; seq_len: number; new_len: number; cache_bytes: number; forward_ms: number; token: string; positions_without_cache: number; positions_with_cache: number }
export interface KvCache {
  step: number; phase: string; seq_len: number; past_len: number; new_len: number; n_layers: number; n_kv_heads: number; n_heads: number;
  head_dim: number; layers: KvLayer[]; total_bytes: number; cache_class: string; history: KvHistory[]; tokens: string[]; key_note: string;
}
export interface KvCompare {
  step: number; phase: string; seq_len: number; without_cache: { positions_processed: number; forward_ms: number };
  with_cache: { positions_processed: number; reused_positions: number; forward_ms: number }; speedup: number | null;
  max_abs_logit_diff: number; max_abs_diff_vs_trace: number | null; same_argmax: boolean; note: string;
}

export interface TokenInfo { index: number; token_id: number; text: string; raw: string; start: number | null; end: number | null; bytes: number; special: boolean; leading_space: boolean }
export interface TokenizeResponse {
  text: string; tokens: TokenInfo[]; count: number; characters: number; decoded: string; round_trip: boolean;
  vocabulary: { tokenizer_class: string; vocab_size: number; base_vocab_size: number; model_max_length: number | null; special_tokens: Record<string, string>; bos_token_id: number | null; eos_token_id: number | null; is_fast: boolean; algorithm: string | null };
}
export interface EmbeddingResponse {
  token_id: number; token: string; shape: number[]; dtype: string; values: number[]; stats: TensorStats; matrix_shape: number[];
  neighbors?: { token_id: number; token: string; cosine: number }[]; neighbors_note?: string;
}
export interface ActivationCurve { name: string; module: string; mlp: string; x: number[]; y: number[]; references: Record<string, number[]>; note: string }
export interface WeightList { layer: number | null; convention: string; weights: { name: string; shape: number[]; dtype: string }[] }

export interface SystemInfo {
  platform: string; python: string; torch: string;
  cpu: { cores: number; torch_threads: number; ram_total: number; ram_available: number; ram_used_percent: number; process_rss: number };
  cuda: { available: boolean; note?: string; devices: { index: number; name: string; total: number; free: number; allocated: number; reserved: number }[] };
  trace_store: { runs: number; stored_bytes: number; budget_bytes: number; retain_steps: number; max_runs: number }; loaded_models: string[];
}
export interface AppSettings { max_runs: number; max_store_bytes: number; retain_steps: number; max_prompt_tokens: number; max_new_tokens: number; max_cells: number; models_dir: string }
