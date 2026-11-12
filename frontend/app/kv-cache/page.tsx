"use client";
import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Loader2, Timer } from "lucide-react";
import { api } from "@/services/api";
import { useStore } from "@/hooks/use-store";
import { useModel, usePalette, useRun, useStepIndex } from "@/hooks/use-lens";
import { ErrorNotice, HeadSelect, LayerSelect, Loading, NeedModel, NeedRun, PageHeader } from "@/components/common";
import { Plot } from "@/components/charts/plot";
import { Heatmap } from "@/components/tensors/heatmap";
import { Badge, Button, Card, CardBody, CardHeader, Field, Formula, Notice, Select, Stat } from "@/components/ui/primitives";
import { bytes, cn, fmt, ms, shapeStr, showToken } from "@/lib/utils";
import type { KvCache, Matrix } from "@/types/api";

/** Which positions each step has to compute, with and without a cache. */
function ReuseGrid({ kv, cached, stepIdx }: { kv: KvCache; cached: boolean; stepIdx: number }) {
  const total = kv.history[kv.history.length - 1].seq_len;
  const size = total > 40 ? 8 : 13;
  let computed = 0;
  return (
    <div>
      <div className="space-y-px overflow-x-auto">
        {kv.history.map((h) => {
          computed += cached ? h.new_len : h.seq_len;
          return (
            <div key={h.step} className={cn("flex items-center gap-1.5", h.step === stepIdx && "bg-accent-soft")}>
              <span className="num w-12 shrink-0 text-[10px] text-muted">step {h.step}</span>
              <div className="flex gap-px">
                {Array.from({ length: total }, (_, pos) => {
                  const exists = pos < h.seq_len, isNew = pos >= h.seq_len - h.new_len && exists;
                  const cls = !exists ? "bg-transparent" : !cached || isNew ? "bg-accent" : "border border-line-strong bg-surface-3";
                  return <i key={pos} className={cn("block shrink-0 rounded-[2px]", cls)} style={{ width: size, height: size }}
                    title={exists ? `step ${h.step}, position ${pos} “${showToken(kv.tokens[pos] ?? "")}”: ${!cached || isNew ? "K and V computed in this step" : "K and V read from the cache"}` : undefined} />;
                })}
              </div>
            </div>
          );
        })}
      </div>
      <p className="num mt-1.5 text-[11px] text-ink-2">{computed} position-passes through the network in total</p>
    </div>
  );
}

export default function KvCachePage() {
  const model = useModel();
  const run = useRun();
  const p = usePalette();
  const stepIdx = useStepIndex(run);
  const layerSel = useStore((s) => s.layer);
  const set = useStore((s) => s.set);
  const [which, setWhich] = useState("key");
  const [headSel, setHead] = useState(0);
  const layer = model ? Math.min(layerSel, model.n_layers - 1) : 0;
  const kvq = useQuery({ queryKey: ["kv", run?.run_id, stepIdx, run?.steps.length], queryFn: () => api.kvCache(run!.run_id, stepIdx), enabled: !!run, placeholderData: (prev) => prev });
  const kv = kvq.data;
  const head = kv ? Math.min(headSel, kv.n_kv_heads - 1) : 0;
  const ten = useQuery({ queryKey: ["kvt", run?.run_id, stepIdx, layer, which, head], queryFn: () => api.kvTensor(run!.run_id, { layer, which, head, step: stepIdx }), enabled: !!run, staleTime: Infinity, retry: false });
  const cmp = useMutation({ mutationFn: () => api.kvCompare(run!.run_id, stepIdx) });
  if (!model) return <><PageHeader title="KV Cache" /><NeedModel /></>;
  if (!run) return <><PageHeader title="KV Cache" /><NeedRun model={model} /></>;
  const c = cmp.data && cmp.data.step === stepIdx ? cmp.data : null;

  return (
    <div className="space-y-3">
      <PageHeader title="KV Cache Explorer" sub="During generation, the Key and Value tensors of earlier tokens are stored and reused, so each new token needs only one position through the network instead of the whole sequence." />
      <ErrorNotice error={kvq.error} />
      {!kv && !kvq.error && <Loading />}
      {kv && (
        <>
          <Card>
            <CardHeader title={`Cache after step ${kv.step} (${kv.phase})`} sub={<span className="num">{kv.cache_class}</span>}
              right={<Badge tone="accent">{kv.phase === "prefill" ? `filled with ${kv.new_len} prompt positions` : `${kv.past_len} reused + ${kv.new_len} appended`}</Badge>} />
            <CardBody className="space-y-3">
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 xl:grid-cols-6">
                <Stat label="Sequence length" value={kv.seq_len} /><Stat label="Layers" value={kv.n_layers} /><Stat label="K/V heads" value={kv.n_kv_heads} sub={kv.n_kv_heads !== kv.n_heads ? `${kv.n_heads} query heads` : undefined} />
                <Stat label="Head dimension" value={kv.head_dim} /><Stat label="Cache size" value={bytes(kv.total_bytes)} /><Stat label="Per-layer K shape" value={shapeStr(kv.layers[0].key_shape)} />
              </div>
              <Formula>{`cache bytes = layers × 2 (K and V) × heads × sequence × head_dim × bytes per value\n            = ${kv.n_layers} × 2 × ${kv.n_kv_heads} × ${kv.seq_len} × ${kv.head_dim} × ${kv.layers[0].bytes / (2 * kv.n_kv_heads * kv.seq_len * kv.head_dim)} = ${kv.total_bytes.toLocaleString()} bytes`}</Formula>
              <Notice>{kv.key_note} Because attention is causal, the keys and values of a position depend only on earlier positions and never change afterwards. A decode step therefore computes Q, K and V for the single new token, appends its K and V to the cache, and attends over all cached rows.</Notice>
            </CardBody>
          </Card>

          <div className="grid gap-3 xl:grid-cols-2">
            <Card>
              <CardHeader title="Without KV caching" sub="Every step re-runs the whole sequence: all keys and values are recomputed." />
              <CardBody><ReuseGrid kv={kv} cached={false} stepIdx={stepIdx} /></CardBody>
            </Card>
            <Card>
              <CardHeader title="With KV caching" sub="After the prefill, each step computes one new position and reuses the rest." />
              <CardBody>
                <ReuseGrid kv={kv} cached stepIdx={stepIdx} />
                <div className="mt-1.5 flex flex-wrap gap-3 text-[11px] text-muted"><span className="flex items-center gap-1"><i className="inline-block size-2.5 rounded-[2px] bg-accent" />computed in this step</span>
                  <span className="flex items-center gap-1"><i className="inline-block size-2.5 rounded-[2px] border border-line-strong bg-surface-3" />reused from the cache</span></div>
              </CardBody>
            </Card>
          </div>

          <Card>
            <CardHeader title="Measured comparison for this step" sub="Runs both paths now, without instrumentation hooks, and compares their next-token logits."
              right={<Button size="sm" variant="default" disabled={cmp.isPending || run.state === "running"} onClick={() => cmp.mutate()}>{cmp.isPending ? <Loader2 className="animate-spin" /> : <Timer />}Run comparison</Button>} />
            <CardBody className="space-y-2">
              <ErrorNotice error={cmp.error} />
              {c ? (
                <>
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                    <Stat label="Without cache" value={ms(c.without_cache.forward_ms)} sub={`${c.without_cache.positions_processed} positions processed`} />
                    <Stat label="With cache" value={ms(c.with_cache.forward_ms)} sub={`${c.with_cache.positions_processed} processed, ${c.with_cache.reused_positions} reused`} />
                    <Stat label="Speed-up" value={c.speedup ? `${c.speedup.toFixed(2)}×` : "—"} />
                    <Stat label="Max logit difference" value={fmt(c.max_abs_logit_diff, 3)} sub={c.same_argmax ? "same top token" : "different top token"} />
                  </div>
                  <p className="text-[11px] leading-4 text-muted">{c.note}{kv.phase === "prefill" && " At the prefill step nothing is cached yet, so both paths process the full prompt."}</p>
                </>
              ) : <p className="text-xs text-ink-2">Press “Run comparison” to execute step {stepIdx} both ways.</p>}
            </CardBody>
          </Card>

          <div className="grid gap-3 xl:grid-cols-2">
            <Card>
              <CardHeader title="Cache growth" sub="Cache size after each step." />
              <CardBody>
                <Plot height={230} data={[{ type: "scatter", mode: "lines+markers", x: kv.history.map((h) => h.step), y: kv.history.map((h) => h.cache_bytes / 1024), text: kv.history.map((h) => `${h.seq_len} positions`),
                  line: { color: p.series[0], width: 2 }, marker: { size: 7, line: { color: p.surface, width: 2 } }, hovertemplate: "step %{x}<br>%{y:.1f} KB<br>%{text}<extra></extra>" }]}
                  layout={{ xaxis: { title: { text: "step" }, dtick: Math.max(1, Math.ceil(kv.history.length / 12)) }, yaxis: { title: { text: "cache size (KB)" }, rangemode: "tozero" } }} onClick={(e) => set({ step: kv.history[e.pointIndex].step })} />
              </CardBody>
            </Card>
            <Card>
              <CardHeader title="Positions processed per step" sub="What each step must push through the network." />
              <CardBody>
                <Plot height={230} data={[
                  { type: "bar", name: "without cache", x: kv.history.map((h) => h.step), y: kv.history.map((h) => h.positions_without_cache), marker: { color: p.series[1] }, hovertemplate: "step %{x}<br>%{y} positions<extra>without cache</extra>" },
                  { type: "bar", name: "with cache", x: kv.history.map((h) => h.step), y: kv.history.map((h) => h.positions_with_cache), marker: { color: p.series[0] }, hovertemplate: "step %{x}<br>%{y} positions<extra>with cache</extra>" },
                ]} layout={{ barmode: "group", xaxis: { title: { text: "step" }, dtick: Math.max(1, Math.ceil(kv.history.length / 12)) }, yaxis: { title: { text: "positions" } } }} />
              </CardBody>
            </Card>
          </div>

          <div className="grid gap-3 xl:grid-cols-[minmax(300px,420px)_1fr]">
            <Card>
              <CardHeader title="Layer-wise cache" sub="Click a layer to view its tensors." />
              <CardBody className="max-h-[420px] overflow-auto p-0">
                <table className="w-full text-left text-xs">
                  <thead className="sticky top-0 bg-surface text-[11px] text-muted"><tr>{["layer", "key shape", "value shape", "dtype", "size"].map((h) => <th key={h} className="border-b border-line px-2 py-1 font-medium">{h}</th>)}</tr></thead>
                  <tbody>{kv.layers.map((l) => (
                    <tr key={l.layer} onClick={() => set({ layer: l.layer })} className={cn("cursor-pointer border-b border-line/60 hover:bg-surface-2", l.layer === layer && "bg-accent-soft")}>
                      <td className="num px-2 py-0.5">{l.layer}</td><td className="num px-2">{shapeStr(l.key_shape)}</td><td className="num px-2">{shapeStr(l.value_shape)}</td><td className="num px-2">{l.dtype}</td><td className="num px-2">{bytes(l.bytes)}</td>
                    </tr>))}</tbody>
                </table>
              </CardBody>
            </Card>
            <Card>
              <CardHeader title={`${which === "key" ? "Key" : "Value"} cache · layer ${layer} · head ${head}`} sub={`One row per cached token (${kv.seq_len}), one column per head dimension (${kv.head_dim}).`}
                right={<><Field label="tensor"><Select value={which} onChange={(e) => setWhich(e.target.value)}><option value="key">keys</option><option value="value">values</option></Select></Field>
                  <LayerSelect n={kv.n_layers} value={layer} onChange={(v) => set({ layer: v })} /><HeadSelect n={kv.n_kv_heads} value={head} onChange={setHead} /></>} />
              <CardBody>
                {ten.data ? (
                  <>
                    <Heatmap data={ten.data.values as Matrix} rowLabels={ten.data.pooled ? undefined : ten.data.tokens} maxCell={16} minCell={2} maxHeight={400}
                      selectedRow={!ten.data.pooled && kv.phase === "decode" ? ten.data.new_from : undefined} rowName="token" colName="dimension" />
                    <p className="mt-1.5 text-[11px] leading-4 text-muted">{kv.phase === "decode" ? `The outlined row (position ${ten.data.new_from}) was appended in this step; the ${ten.data.new_from} rows above it are unchanged from earlier steps.` : "Prefill: every row was computed in this step."} {ten.data.note}</p>
                  </>
                ) : ten.error ? <ErrorNotice error={ten.error} /> : <Loading />}
              </CardBody>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
