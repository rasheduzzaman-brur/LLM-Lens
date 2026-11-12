"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/services/api";
import { useStore } from "@/hooks/use-store";
import { usePalette } from "@/hooks/use-lens";
import { ErrorNotice, Loading, TensorChip, VerifiedBadge, useTrace } from "@/components/common";
import { Plot } from "@/components/charts/plot";
import { Heatmap, VectorHeat } from "@/components/tensors/heatmap";
import { Badge, Formula, Notice } from "@/components/ui/primitives";
import { fmt, pct, shapeStr, showToken } from "@/lib/utils";
import type { Matrix, ModelInfo, RunSummary } from "@/types/api";

interface Ctx { run: RunSummary; model: ModelInfo; layer: number; head: number; position: number; stepIdx: number }

/** Step whose forward pass computed `position`, and the row index inside that step. */
export function sourceOf(run: RunSummary, position: number) {
  const n = run.n_prompt_tokens;
  return position < n ? { step: 0, row: position } : { step: position - n + 1, row: 0 };
}

function useMat(run: RunSummary, name: string, layer: number | null, step: number, index: string, assembled = false) {
  return useQuery({ queryKey: ["tensor", run.run_id, name, layer, step, index, assembled], queryFn: () => api.tensor(run.run_id, { name, layer, step, index, assembled }), staleTime: Infinity, retry: false });
}

function Mat({ title, sub, q, tokens, rowTokens, sel, kind, chip }: { title: string; sub?: string; q: ReturnType<typeof useMat>; tokens?: string[]; rowTokens?: string[];
  sel?: { row?: number; col?: number; onSelect?: (r: number, c: number) => void }; kind?: "sequential" | "diverging"; chip?: React.ReactNode }) {
  const v = q.data;
  const data = v && (v.pooled ? v.pooled_shape! : v.shape).length === 2 ? (v.values as (number | boolean | null)[][]).map((r) => r.map((x) => (typeof x === "boolean" ? Number(x) : x))) as Matrix : null;
  return (
    <div className="min-w-0 rounded-md border border-line p-2">
      <div className="mb-1 flex flex-wrap items-center justify-between gap-1">
        <div className="text-xs font-medium">{title}</div>
        <div className="flex items-center gap-1">{v && <span className="num text-[11px] text-muted">{shapeStr(v.shape)}</span>}{chip}</div>
      </div>
      {sub && <div className="num mb-1.5 text-[11px] text-ink-2">{sub}</div>}
      {q.isLoading && <Loading />}
      <ErrorNotice error={q.error} />
      {data && <Heatmap data={data} kind={kind} rowLabels={v!.pooled ? undefined : rowTokens} colLabels={v!.pooled ? undefined : tokens} maxCell={22} maxHeight={300}
        selectedRow={sel?.row} selectedCol={sel?.col} onSelect={sel?.onSelect} rowName="query" colName="key" />}
      {v?.pooled && <p className="mt-1 text-[11px] text-muted">{v.pool_note}</p>}
    </div>
  );
}

export function QkvTab({ run, model, layer, head, position, stepIdx }: Ctx) {
  const select = useStore((s) => s.select);
  const trace = useTrace(run.run_id, stepIdx, 0);
  const idx = `0,${head}`;
  const x = useMat(run, "ln1_out", layer, stepIdx, "0", true);
  const q = useMat(run, "q_used", layer, stepIdx, idx, true);
  const k = useMat(run, "k_used", layer, stepIdx, idx);
  const v = useMat(run, "v_used", layer, stepIdx, idx);
  const seq = run.steps[stepIdx].seq_len;
  const tokens = run.tokens.slice(0, seq).map((t) => t.token);
  const c = model.capabilities;
  const ev = (n: string) => trace.find(layer, n);
  return (
    <div className="space-y-3">
      <Formula>{c.fused_qkv
        ? "[Q | K | V] = X · W_attn + b_attn        one fused projection (GPT-2 Conv1D), then split in three\nQ = X·Wq + bq     K = X·Wk + bk     V = X·Wv + bv"
        : `Q = X · Wq${c.qkv_bias ? " + bq" : ""}     K = X · Wk${c.qkv_bias ? " + bk" : ""}     V = X · Wv${c.qkv_bias ? " + bv" : ""}\nQ, K are then rotated by the position (RoPE):  x ⊙ cos + rotate_half(x) ⊙ sin`}</Formula>
      <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-muted">
        <span>weight matrices:</span>
        {["Wq", "Wk", "Wv", ...(c.qkv_bias ? ["bq", "bk", "bv"] : [])].map((w) => <button key={w} onClick={() => select({ kind: "weight", name: w, layer })} className="num cursor-pointer rounded border border-line-strong px-1.5 py-px text-ink-2 hover:border-accent">{w}</button>)}
        <span className="ml-2">tensors:</span>
        {["ln1_out", ...(c.fused_qkv ? ["qkv"] : []), "q", "k", "v", "q_used", ...(c.rope ? ["k_rot"] : []), "k_cache", "v_cache", "k_used", "v_used"].map((n) => <TensorChip key={n} name={n} layer={layer} step={stepIdx} />)}
      </div>
      <div className="num grid gap-x-4 gap-y-0.5 text-[11px] text-ink-2 sm:grid-cols-2">
        {["ln1_out", "q", "k", "v", "q_used", "k_used", "v_used"].map((n) => ev(n) && <div key={n}><span className="text-ink">{n}</span> {shapeStr(ev(n)!.shape)} {ev(n)!.dtype} · {ev(n)!.operation}</div>)}
      </div>
      <Mat title="X: input to attention (after normalization 1)" sub={`rows = tokens, columns = ${model.hidden_size} hidden dimensions`} q={x} rowTokens={tokens} sel={{ row: position }} chip={<TensorChip name="ln1_out" layer={layer} step={stepIdx} />} />
      <div className="grid gap-3 xl:grid-cols-3">
        <Mat title={`Q, head ${head}`} sub={`rows = tokens, columns = ${model.head_dim} head dimensions${c.rope ? " (after RoPE)" : ""}`} q={q} rowTokens={tokens} sel={{ row: position }} />
        <Mat title={`K, head ${head}`} sub={c.grouped_query_attention ? `K/V head ${Math.floor(head / (model.n_heads / model.n_kv_heads))} shared by ${model.n_heads / model.n_kv_heads} query heads` : "as stored in the KV cache"} q={k} rowTokens={tokens} sel={{ row: position }} />
        <Mat title={`V, head ${head}`} sub="as stored in the KV cache" q={v} rowTokens={tokens} sel={{ row: position }} />
      </div>
      <p className="text-[11px] leading-4 text-muted">Each projection output of width {model.hidden_size} is reshaped into {model.n_heads} heads of {model.head_dim} dimensions (a reshape, no arithmetic). Signed values use the diverging scale centred on 0.</p>
    </div>
  );
}

export function ScoresTab({ run, model, layer, head, position }: Ctx) {
  const set = useStore((s) => s.set);
  const src = sourceOf(run, position);
  const [key, setKey] = useState<number | null>(null);
  const idx = `0,${head}`;
  const raw = useMat(run, "scores_raw", layer, src.step, idx);
  const scaled = useMat(run, "scores", layer, src.step, idx);
  const mask = useMat(run, "mask", layer, src.step, "");
  const masked = useMat(run, "scores_masked", layer, src.step, idx);
  const seq = run.steps[src.step].seq_len;
  const tokens = run.tokens.slice(0, seq).map((t) => t.token);
  const rowTokens = src.step === 0 ? tokens : [run.tokens[position].token];
  const j = Math.min(key ?? position, seq - 1);
  const row = useQuery({ queryKey: ["arow", run.run_id, layer, head, position], queryFn: () => api.attentionRow(run.run_id, layer, head, position, run.steps.length - 1), staleTime: Infinity, retry: false });
  const kvec = useMat(run, "k_used", layer, src.step, `0,${head},${j}`);
  const onSelect = (r: number, c: number) => { setKey(c); if (src.step === 0) set({ position: r }); };
  const sel = { row: src.row, col: j, onSelect };
  const r = row.data;
  return (
    <div className="space-y-3">
      <Formula>{`scores = Q · Kᵀ / √d_k        d_k = ${model.head_dim},  1/√d_k = ${r ? fmt(r.scale, 6) : "…"}\nmasked = ${model.capabilities.mask_mode === "fill" ? "where(j ≤ i, scores, finfo.min)" : "scores + (0 if j ≤ i else finfo.min)"}`}</Formula>
      {src.step > 0 && <Notice>Token “{showToken(run.tokens[position].token)}” was generated, so its attention was computed incrementally in decode step {src.step}: Q has a single row (this token) while K holds all {seq} positions from the KV cache. Select a prompt token to see the full prefill matrix.</Notice>}
      <div className="grid gap-3 xl:grid-cols-2">
        <Mat title="1 · Q · Kᵀ" sub="one dot product per (query, key) pair" q={raw} tokens={tokens} rowTokens={rowTokens} sel={sel} chip={<TensorChip name="scores_raw" layer={layer} step={src.step} />} />
        <Mat title="2 · Scaled by 1/√d_k" sub="keeps the softmax from saturating" q={scaled} tokens={tokens} rowTokens={rowTokens} sel={sel} chip={<TensorChip name="scores" layer={layer} step={src.step} />} />
        <Mat title="3 · Causal mask" sub="1 = may attend, 0 = future position" q={mask} tokens={tokens} rowTokens={rowTokens} sel={sel} kind="sequential" chip={<TensorChip name="mask" layer={layer} step={src.step} />} />
        <Mat title="4 · Masked scores" sub="future positions hold finfo.min (drawn hatched)" q={masked} tokens={tokens} rowTokens={rowTokens} sel={sel} chip={<TensorChip name="scores_masked" layer={layer} step={src.step} />} />
      </div>
      <Notice>
        <b className="text-ink">Why the mask?</b> During generation, token <i>i</i> is produced before tokens <i>i+1, i+2, …</i> exist. Training must match that, so position <i>i</i> may only read positions ≤ <i>i</i>. Masked scores are set to the most negative finite number, which softmax turns into a weight of exactly 0. This is also what makes the KV cache valid: earlier positions never depend on later ones, so their keys and values never need recomputing.
      </Notice>
      <div className="space-y-2 rounded-md border border-line p-2.5">
        <div className="text-xs font-medium">One cell: query “{showToken(run.tokens[position].token)}” (position {position}) · key “{showToken(tokens[j] ?? "")}” (position {j}) <span className="font-normal text-muted">click any cell above to change</span></div>
        {r && kvec.data && (kvec.data.shape.length === 1) ? (
          <>
            <div className="grid gap-2 md:grid-cols-2">
              <div><div className="mb-0.5 text-[11px] text-muted">query vector q<sub>{position}</sub> ({r.query.length} values)</div><VectorHeat values={r.query} wrapAt={r.query.length} maxCell={10} legend={false} /></div>
              <div><div className="mb-0.5 text-[11px] text-muted">key vector k<sub>{j}</sub></div><VectorHeat values={kvec.data.values as number[]} wrapAt={r.query.length} maxCell={10} legend={false} /></div>
            </div>
            <div className="num overflow-x-auto text-xs leading-6 whitespace-nowrap">
              <div>q · k = Σ<sub>d</sub> q[d]·k[d] = <b>{fmt(r.scores_raw[j], 6)}</b> <span className="text-muted">(scores_raw[{src.row}, {j}])</span></div>
              <div>scaled = {fmt(r.scores_raw[j], 6)} × {fmt(r.scale, 6)} = <b>{fmt(r.scores[j], 6)}</b></div>
              <div>mask: {r.allowed[j] ? <>key {j} ≤ query {position} → <b>kept</b></> : <>key {j} &gt; query {position} → <b>masked</b> (finfo.min)</>}</div>
            </div>
          </>
        ) : row.error ? <ErrorNotice error={row.error} /> : <Loading />}
      </div>
    </div>
  );
}

export function SoftmaxTab({ run, layer, head, position }: Ctx) {
  const p = usePalette();
  const row = useQuery({ queryKey: ["arow", run.run_id, layer, head, position], queryFn: () => api.attentionRow(run.run_id, layer, head, position, run.steps.length - 1), staleTime: Infinity, retry: false });
  const r = row.data;
  if (row.error) return <ErrorNotice error={row.error} />;
  if (!r) return <Loading />;
  const x = r.tokens.map((t, i) => `${i} ${showToken(t)}`);
  const bar = (y: (number | null)[], color: string, name: string) => ({ type: "bar" as const, x, y, marker: { color }, name, hovertemplate: `%{x}<br>${name} %{y:.5f}<extra></extra>` });
  const allowed = (v: number[]) => v.map((s, i) => (r.allowed[i] ? s : null));
  const lay = (t: string) => ({ margin: { l: 48, r: 8, t: 8, b: 64 }, xaxis: { tickangle: -50, tickfont: { size: 9, family: "ui-monospace, monospace" } }, yaxis: { title: { text: t } } });
  return (
    <div className="space-y-3">
      <Formula>{"weights[i, j] = exp(score[i, j]) / Σₖ exp(score[i, k])        (sum over the allowed keys k ≤ i)"}</Formula>
      <div className="flex flex-wrap items-center gap-1.5 text-xs text-ink-2">
        Row for query “<span className="num text-ink">{showToken(r.token)}</span>” (position {r.position}), layer {r.layer}, head {r.head}, computed in step {r.src_step} ({r.phase}).
        <Badge tone="good">Σ weights = {fmt(r.weights_sum, 7)}</Badge><Badge>exp/Σexp vs model weights: max diff {fmt(r.recompute_max_err, 2)}</Badge>
      </div>
      <div className="grid gap-3 xl:grid-cols-3">
        <div className="rounded-md border border-line p-2"><div className="text-xs font-medium">1 · Input scores</div><Plot height={220} data={[bar(allowed(r.scores), p.series[0], "score")]} layout={lay("scaled score")} /></div>
        <div className="rounded-md border border-line p-2"><div className="text-xs font-medium">2 · Exponentiation</div><Plot height={220} data={[bar(allowed(r.exp), p.series[0], "exp")]} layout={lay("exp(score − max)")} /></div>
        <div className="rounded-md border border-line p-2"><div className="text-xs font-medium">3 · Normalized probabilities</div><Plot height={220} data={[bar(allowed(r.weights), p.series[0], "weight")]} layout={{ ...lay("attention weight"), yaxis: { title: { text: "attention weight" }, range: [0, 1] } }} /></div>
      </div>
      <div className="num rounded-md border border-line bg-surface-2 px-2.5 py-1.5 text-xs">normalization denominator Σ exp = <b>{fmt(r.denominator, 6)}</b> · max allowed score = {fmt(r.shift, 6)}</div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="text-[11px] text-muted"><tr>{["key position", "key token", "Q·Kᵀ", "scaled score", "mask", "exp(score − max)", "÷ Σexp = weight", ""].map((h) => <th key={h} className="border-b border-line px-2 py-1 font-medium">{h}</th>)}</tr></thead>
          <tbody>
            {r.tokens.map((t, i) => (
              <tr key={i} className="border-b border-line/60">
                <td className="num px-2 py-0.5 text-muted">{i}</td><td className="num px-2 text-ink">“{showToken(t)}”</td><td className="num px-2">{fmt(r.scores_raw[i], 5)}</td><td className="num px-2">{fmt(r.scores[i], 5)}</td>
                <td className="px-2">{r.allowed[i] ? "allowed" : <span className="text-muted">masked (future)</span>}</td><td className="num px-2">{r.allowed[i] ? fmt(r.exp[i], 5) : "0"}</td>
                <td className="num px-2 font-semibold text-ink">{r.allowed[i] ? fmt(r.weights[i], 5) : "0"}</td>
                <td className="w-28 px-2"><span className="block h-1.5 rounded-sm bg-surface-3"><span className="block h-full rounded-sm bg-accent" style={{ width: pct(r.weights[i] ?? 0) }} /></span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[11px] leading-4 text-muted">{r.exp_note} Scores, exponentials and weights come from the backend, computed from the captured tensors of this run.</p>
    </div>
  );
}

export function OutputTab({ run, model, layer, head, position, stepIdx }: Ctx) {
  const src = sourceOf(run, position);
  const trace = useTrace(run.run_id, src.step, 0);
  const row = useQuery({ queryKey: ["arow", run.run_id, layer, head, position], queryFn: () => api.attentionRow(run.run_id, layer, head, position, run.steps.length - 1), staleTime: Infinity, retry: false });
  const v = useMat(run, "v_used", layer, src.step, `0,${head}`);
  const merged = useMat(run, "attn_merged", layer, src.step, `0,${src.row}`);
  const out = useMat(run, "attn_out", layer, src.step, `0,${src.row}`);
  const tokens = run.tokens.slice(0, run.steps[src.step].seq_len).map((t) => t.token);
  const r = row.data;
  const check = trace.find(layer, "attn_merged");
  const top = r ? r.weights.map((w, i) => ({ w, i })).filter((o) => r.allowed[o.i]).sort((a, b) => b.w - a.w).slice(0, 5) : [];
  return (
    <div className="space-y-3">
      <Formula>{`head_out[i] = Σⱼ weights[i, j] · V[j]              (weighted average of value vectors)\nattn_merged  = concat(head_0 … head_${model.n_heads - 1})         [${model.n_heads} × ${model.head_dim} = ${model.n_heads * model.head_dim}]\nattn_out     = attn_merged · Wo${model.capabilities.fused_qkv ? " + bo" : ""}`}</Formula>
      <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-muted">
        <VerifiedBadge check={check} /><span>{check?.verified_against}</span>
      </div>
      <div className="grid gap-3 xl:grid-cols-[minmax(220px,300px)_minmax(0,1fr)]">
        <div className="rounded-md border border-line p-2">
          <div className="mb-1 text-xs font-medium">Attention weights of “{showToken(run.tokens[position].token)}”</div>
          {r ? (
            <table className="w-full text-xs">
              <tbody>{r.tokens.map((t, i) => r.allowed[i] && (
                <tr key={i} className="border-b border-line/60"><td className="num py-0.5 text-muted">{i}</td><td className="num text-ink">“{showToken(t)}”</td><td className="num text-right">{fmt(r.weights[i], 4)}</td>
                  <td className="w-20 pl-2"><span className="block h-1.5 rounded-sm bg-surface-3"><span className="block h-full rounded-sm bg-accent" style={{ width: pct(r.weights[i]) }} /></span></td></tr>
              ))}</tbody>
            </table>
          ) : <Loading />}
        </div>
        <Mat title={`V, head ${head}: one value vector per key token`} sub={`rows = key tokens, columns = ${model.head_dim} head dimensions`} q={v} rowTokens={tokens} chip={<TensorChip name="v_used" layer={layer} step={src.step} />} />
      </div>
      {r && (
        <div className="space-y-2 rounded-md border border-line p-2.5">
          <div className="text-xs font-medium">Multiplication result for head {head}</div>
          <div className="num overflow-x-auto text-xs whitespace-nowrap text-ink-2">head_out = {top.map((o) => `${fmt(o.w, 3)}·V[“${showToken(r.tokens[o.i])}”]`).join(" + ")}{r.weights.filter((_, i) => r.allowed[i]).length > 5 ? " + …" : ""}</div>
          <VectorHeat values={r.head_output} wrapAt={r.head_output.length} maxCell={12} name="head dim" />
          <div className="flex items-center gap-1.5 text-[11px] text-muted"><TensorChip name="head_out" layer={layer} step={src.step} /> row [{head}, {src.row}] of the per-head output tensor</div>
        </div>
      )}
      <div className="grid gap-3 xl:grid-cols-2">
        <div className="rounded-md border border-line p-2">
          <div className="mb-1 flex items-center justify-between text-xs font-medium">All heads concatenated <TensorChip name="attn_merged" layer={layer} step={src.step} /></div>
          {merged.data && merged.data.shape.length === 1 ? <VectorHeat values={merged.data.values as number[]} wrapAt={model.head_dim} maxCell={9} name="element" /> : merged.error ? <ErrorNotice error={merged.error} /> : <Loading />}
          <p className="mt-1 text-[11px] text-muted">One row per head ({model.n_heads} rows × {model.head_dim}). Row {head} is the vector above. This tensor is hooked from the model (input of the output projection).</p>
        </div>
        <div className="rounded-md border border-line p-2">
          <div className="mb-1 flex items-center justify-between text-xs font-medium">Final attention output (after Wo) <TensorChip name="attn_out" layer={layer} step={src.step} /></div>
          {out.data && out.data.shape.length === 1 ? <VectorHeat values={out.data.values as number[]} wrapAt={out.data.shape[0] > 512 ? 64 : 48} maxCell={9} name="element" /> : out.error ? <ErrorNotice error={out.error} /> : <Loading />}
          <p className="mt-1 text-[11px] text-muted">This vector is added to the residual stream of this token (see Transformer Layers → Residual connections).</p>
        </div>
      </div>
      <span className="hidden">{stepIdx}</span>
    </div>
  );
}
