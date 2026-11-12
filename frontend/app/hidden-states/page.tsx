"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/services/api";
import { useStore } from "@/hooks/use-store";
import { useModel, usePalette, usePosition, useRun, useStepIndex } from "@/hooks/use-lens";
import { ErrorNotice, Loading, NeedModel, NeedRun, PageHeader, TensorChip, VizNote } from "@/components/common";
import { ValueTable } from "@/components/inspector/inspector";
import { Plot } from "@/components/charts/plot";
import { Heatmap, VectorHeat } from "@/components/tensors/heatmap";
import { Card, CardBody, CardHeader, Field, Notice, Select, Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/primitives";
import { fmt, showToken } from "@/lib/utils";
import type { Matrix } from "@/types/api";

/** depth label -> (layer, tensor name) of the trace */
function tensorOf(label: string, stage: string): { layer: number | null; name: string } {
  if (label === "embedding") return { layer: null, name: "emb_out" };
  if (label === "final norm") return { layer: null, name: "final_norm_out" };
  return { layer: Number(label.slice(1)), name: stage };
}

export default function HiddenStatesPage() {
  const model = useModel();
  const run = useRun();
  const p = usePalette();
  const stepIdx = useStepIndex(run);
  const position = usePosition(run, stepIdx);
  const set = useStore((s) => s.set);
  const [stage, setStage] = useState("resid_post");
  const [depthSel, setDepth] = useState(-1);
  const hs = useQuery({ queryKey: ["hidden", run?.run_id, stage, stepIdx, position, depthSel], queryFn: () => api.hiddenStates(run!.run_id, { stage, step: stepIdx, position, depth: depthSel }),
    enabled: !!run, staleTime: Infinity, retry: false, placeholderData: (prev) => prev });
  const h = hs.data;
  const t = h ? tensorOf(h.depth_labels[h.depth], h.stage) : null;
  const mat = useQuery({ queryKey: ["hmat", run?.run_id, t?.layer, t?.name, stepIdx], queryFn: () => api.tensor(run!.run_id, { name: t!.name, layer: t!.layer, step: stepIdx, index: "0", assembled: true }), enabled: !!run && !!t, staleTime: Infinity, retry: false });
  const vec = useQuery({ queryKey: ["hvec", run?.run_id, t?.layer, t?.name, stepIdx, position], queryFn: () => api.tensor(run!.run_id, { name: t!.name, layer: t!.layer, step: stepIdx, index: `0,${position}`, assembled: true }), enabled: !!run && !!t, staleTime: Infinity, retry: false });
  if (!model) return <><PageHeader title="Hidden States" /><NeedModel /></>;
  if (!run) return <><PageHeader title="Hidden States" /><NeedRun model={model} /></>;
  const tok = run.tokens[position];

  return (
    <div className="space-y-3">
      <PageHeader title="Hidden State Explorer" sub="The vector that represents each token, at every depth of the network. Select a token in the strip above and a depth below."
        right={<>
          <Field label="stage"><Select value={stage} onChange={(e) => { setStage(e.target.value); setDepth(-1); }}>{Object.entries(h?.stages ?? { resid_post: "Layer output (residual stream)" }).map(([k, v]) => <option key={k} value={k}>{v}</option>)}</Select></Field>
          {h && <Field label="depth"><Select value={h.depth} onChange={(e) => setDepth(Number(e.target.value))}>{h.depth_labels.map((l, i) => <option key={l} value={i}>{l}</option>)}</Select></Field>}
        </>} />
      <ErrorNotice error={hs.error} />
      {!h && !hs.error && <Loading />}
      {h && (
        <>
          <div className="grid gap-3 xl:grid-cols-2">
            <Card>
              <CardHeader title="Vector norm by depth and token" sub="L2 norm of each hidden state. Rows = depth, columns = tokens. Click a cell to select that depth and token." />
              <CardBody><Heatmap data={h.norms} rowLabels={h.depth_labels} colLabels={h.tokens} kind="sequential" maxCell={30} maxHeight={420} selectedRow={h.depth} selectedCol={h.position}
                onSelect={(r, c) => { setDepth(r); set({ position: c }); }} rowName="depth" colName="token" valueName="L2 norm" /></CardBody>
            </Card>
            <Card>
              <CardHeader title="2-D projection (PCA)" sub={`Each point is one token at one depth. The line follows “${showToken(tok.token)}” from the embedding to the last layer.`} />
              <CardBody>
                <Plot height={400} data={[
                  { type: "scattergl", mode: "markers", name: "all tokens, all depths", x: h.pca.points.flatMap((r) => r.map((q) => q[0])), y: h.pca.points.flatMap((r) => r.map((q) => q[1])),
                    text: h.pca.points.flatMap((r, d) => r.map((_, i) => `${showToken(h.tokens[i])} · ${h.depth_labels[d]}`)), marker: { color: p.muted, size: 5, opacity: 0.55 }, hovertemplate: "%{text}<extra></extra>" },
                  { type: "scatter", mode: "lines+markers", name: `“${showToken(tok.token)}” through depth`, x: h.pca.points.map((r) => r[h.position][0]), y: h.pca.points.map((r) => r[h.position][1]),
                    text: h.depth_labels, line: { color: p.series[0], width: 2 }, marker: { color: p.series[0], size: 8, line: { color: p.surface, width: 2 } }, hovertemplate: "%{text}<extra></extra>" },
                  { type: "scatter", mode: "markers", name: `depth ${h.depth_labels[h.depth]}, all tokens`, x: h.pca.points[h.depth].map((q) => q[0]), y: h.pca.points[h.depth].map((q) => q[1]),
                    text: h.tokens.map(showToken), marker: { color: p.series[1], size: 8, symbol: "diamond", line: { color: p.surface, width: 2 } }, hovertemplate: "%{text}<extra></extra>" },
                ]} layout={{ xaxis: { title: { text: `PC1 (${(h.pca.explained_variance_ratio[0] * 100).toFixed(1)}% of variance)` } }, yaxis: { title: { text: `PC2 (${(h.pca.explained_variance_ratio[1] * 100).toFixed(1)}%)` } } }} />
                <VizNote>{h.pca.note}</VizNote>
              </CardBody>
            </Card>
          </div>

          <div className="grid gap-3 xl:grid-cols-2">
            <Card>
              <CardHeader title={`Token-to-token cosine similarity at depth ${h.depth_labels[h.depth]}`} sub="How similar the hidden states of two tokens are at this depth. Click a cell to select a token." />
              <CardBody>
                <Heatmap data={h.token_similarity} rowLabels={h.tokens} colLabels={h.tokens} kind="diverging" domain={[-1, 1]} maxCell={28} maxHeight={400} selectedRow={h.position}
                  onSelect={(r) => set({ position: r })} rowName="token" colName="token" valueName="cosine" />
                <VizNote>{h.similarity_note}</VizNote>
              </CardBody>
            </Card>
            <Card>
              <CardHeader title={`Layer-to-layer comparison for “${showToken(tok.token)}”`} sub="Cosine similarity between this token's hidden states at different depths." />
              <CardBody>
                <Tabs defaultValue="consecutive">
                  <TabsList><TabsTrigger value="consecutive">Change per layer</TabsTrigger><TabsTrigger value="matrix">All pairs</TabsTrigger></TabsList>
                  <TabsContent value="consecutive" className="mt-2">
                    <Plot height={300} data={[{ type: "scatter", mode: "lines+markers", x: h.depth_labels.slice(1), y: h.consecutive_similarity, line: { color: p.series[0], width: 2 }, marker: { size: 7, line: { color: p.surface, width: 2 } },
                      hovertemplate: "%{x} vs previous<br>cosine %{y:.4f}<extra></extra>" }]}
                      layout={{ xaxis: { title: { text: "depth" }, tickangle: -45 }, yaxis: { title: { text: "cosine similarity with the previous depth" }, range: [Math.min(0, ...h.consecutive_similarity) - 0.05, 1.02] } }} />
                    <p className="text-[11px] text-muted">A value near 1 means the layer changed the direction of the vector very little.</p>
                  </TabsContent>
                  <TabsContent value="matrix" className="mt-2">
                    <Heatmap data={h.depth_similarity} plainLabels rowLabels={h.depth_labels} colLabels={h.depth_labels} kind="diverging" domain={[-1, 1]} maxCell={24} maxHeight={330} selectedRow={h.depth}
                      onSelect={(r) => setDepth(r)} rowName="depth" colName="depth" valueName="cosine" />
                  </TabsContent>
                </Tabs>
              </CardBody>
            </Card>
          </div>

          <Card>
            <CardHeader title={`Tensor at depth ${h.depth_labels[h.depth]}`} sub={`${h.tokens.length} tokens × ${h.hidden_size} dimensions, read from the run.`} right={t && <TensorChip name={t.name} layer={t.layer} step={stepIdx} label={`${t.layer !== null ? `L${t.layer}.` : ""}${t.name}`} />} />
            <CardBody>
              <Tabs defaultValue="heat">
                <TabsList><TabsTrigger value="heat">Heatmap (all tokens)</TabsTrigger><TabsTrigger value="vector">Selected token</TabsTrigger><TabsTrigger value="num">Numerical values</TabsTrigger></TabsList>
                <TabsContent value="heat" className="mt-3">
                  {mat.data ? <Heatmap data={mat.data.values as Matrix} rowLabels={mat.data.pooled ? undefined : h.tokens} maxCell={8} minCell={1} maxHeight={360} selectedRow={mat.data.pooled ? undefined : h.position}
                    onSelect={mat.data.pooled ? undefined : (r) => set({ position: r })} rowName="token" colName="dimension" /> : mat.error ? <ErrorNotice error={mat.error} /> : <Loading />}
                  {mat.data?.pooled && <Notice tone="warn" className="mt-2">{mat.data.pool_note}</Notice>}
                </TabsContent>
                <TabsContent value="vector" className="mt-3">
                  {vec.data ? <><VectorHeat values={vec.data.values as number[]} wrapAt={h.hidden_size > 512 ? 64 : 48} maxCell={14} name="dimension" />
                    <p className="num mt-1 text-[11px] text-muted">“{showToken(tok.token)}” · min {fmt(vec.data.stats?.min)} · max {fmt(vec.data.stats?.max)} · mean {fmt(vec.data.stats?.mean)} · L2 {fmt(vec.data.stats?.l2)}</p></> : vec.error ? <ErrorNotice error={vec.error} /> : <Loading />}
                </TabsContent>
                <TabsContent value="num" className="mt-3">{vec.data ? <ValueTable values={vec.data.values as number[]} page={96} /> : <Loading />}</TabsContent>
              </Tabs>
            </CardBody>
          </Card>
        </>
      )}
    </div>
  );
}
