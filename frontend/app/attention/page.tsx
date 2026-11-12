"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/services/api";
import { useStore } from "@/hooks/use-store";
import { useModel, usePalette, usePosition, useRun, useStepIndex } from "@/hooks/use-lens";
import { ErrorNotice, HeadSelect, LayerSelect, Loading, NeedModel, NeedRun, PageHeader, VerifiedBadge, VizNote } from "@/components/common";
import { OutputTab, QkvTab, ScoresTab, SoftmaxTab } from "@/components/attention/math";
import { Plot } from "@/components/charts/plot";
import { Heatmap } from "@/components/tensors/heatmap";
import { Card, CardBody, CardHeader, Field, Select, Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/primitives";
import { cn, fmt, showToken } from "@/lib/utils";

export default function AttentionPage() {
  const model = useModel();
  const run = useRun();
  const p = usePalette();
  const stepIdx = useStepIndex(run);
  const position = usePosition(run, stepIdx);
  const set = useStore((s) => s.set);
  const layerSel = useStore((s) => s.layer);
  const headSel = useStore((s) => s.head);
  const [reduce, setReduce] = useState("mean");
  const layer = model ? Math.min(layerSel, model.n_layers - 1) : 0;
  const head = model ? Math.min(headSel, model.n_heads - 1) : 0;
  const att = useQuery({ queryKey: ["attention", run?.run_id, layer, stepIdx], queryFn: () => api.attention(run!.run_id, layer, stepIdx), enabled: !!run, staleTime: Infinity, retry: false, placeholderData: (prev) => prev });
  const grid = useQuery({ queryKey: ["agrid", run?.run_id, stepIdx, reduce, head], queryFn: () => api.attentionGrid(run!.run_id, stepIdx, reduce, head), enabled: !!run, staleTime: Infinity, retry: false, placeholderData: (prev) => prev });
  if (!model) return <><PageHeader title="Attention Explorer" /><NeedModel /></>;
  if (!run) return <><PageHeader title="Attention Explorer" /><NeedRun model={model} /></>;
  const a = att.data;
  const ctx = { run, model, layer, head, position, stepIdx };
  const tokens = a?.tokens ?? [];
  const row = a && position < a.seq_len ? a.weights[head]?.[position] : undefined;

  return (
    <div className="space-y-3">
      <PageHeader title="Attention Explorer" sub="Self-attention of the selected layer and head, from the Q/K/V projections to the output written back into the residual stream. Every matrix is a tensor of this run."
        right={<><LayerSelect n={model.n_layers} value={layer} onChange={(v) => set({ layer: v })} /><HeadSelect n={model.n_heads} value={head} onChange={(v) => set({ head: v })} /></>} />
      <ErrorNotice error={att.error} />
      <div className="grid gap-3 xl:grid-cols-[1fr_minmax(260px,340px)]">
        <Card>
          <CardHeader title={`Attention weights · layer ${layer} · head ${head}`} sub="Row = the token that is attending (query). Column = the token being attended to (key). Click a cell to select that query token."
            right={a && <VerifiedBadge check={a} />} />
          <CardBody>
            {a ? <Heatmap data={a.weights[head]} rowLabels={tokens} colLabels={tokens} kind="sequential" domain={[0, 1]} maxCell={34} maxHeight={560} selectedRow={position}
              onSelect={(r) => set({ position: r })} rowName="query" colName="key" valueName="weight" /> : <Loading />}
            {a && <p className="mt-1.5 text-[11px] leading-4 text-muted">{a.note} Verified against {a.verified_against}.</p>}
          </CardBody>
        </Card>
        <Card>
          <CardHeader title={<>“{showToken(run.tokens[position].token)}” attends to</>} sub={`Row ${position} of the matrix: weights over earlier tokens (they sum to 1).`} />
          <CardBody>
            {row ? (
              <>
                <Plot height={Math.max(160, 22 * (position + 1) + 50)} data={[{ type: "bar", orientation: "h", y: tokens.slice(0, position + 1).map((t, i) => `${i} ${showToken(t)}`), x: row.slice(0, position + 1) as number[],
                  marker: { color: p.series[0] }, hovertemplate: "%{y}<br>weight %{x:.4f}<extra></extra>" }]}
                  layout={{ margin: { l: 96, r: 10, t: 6, b: 30 }, xaxis: { range: [0, 1], title: { text: "attention weight" } }, yaxis: { autorange: "reversed", tickfont: { size: 10, family: "ui-monospace, monospace" } } }} />
                <p className="num text-[11px] text-muted">entropy {fmt(a!.entropy[head][position], 3)} nats · {position + 1} visible position(s)</p>
              </>
            ) : <Loading />}
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardBody>
          <Tabs defaultValue="heads">
            <TabsList>
              <TabsTrigger value="heads">Compare heads</TabsTrigger><TabsTrigger value="layers">Compare layers</TabsTrigger><TabsTrigger value="qkv">1 · Query, Key, Value</TabsTrigger>
              <TabsTrigger value="scores">2 · Scores &amp; causal mask</TabsTrigger><TabsTrigger value="softmax">3 · Softmax</TabsTrigger><TabsTrigger value="output">4 · Attention output</TabsTrigger>
            </TabsList>
            <TabsContent value="heads" className="mt-3">
              {a ? (
                <>
                  <p className="mb-2 text-xs text-ink-2">All {a.n_heads} heads of layer {layer} on one shared scale (0 to 1). Click a head to select it.</p>
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4 2xl:grid-cols-6">
                    {a.weights.map((w, h) => (
                      <button key={h} onClick={() => set({ head: h })} className={cn("cursor-pointer rounded-md border p-1.5 text-left", h === head ? "border-accent bg-accent-soft" : "border-line hover:border-line-strong")}>
                        <div className="mb-1 flex justify-between text-[11px]"><span className="font-medium text-ink">head {h}</span><span className="num text-muted" title="Mean entropy of the rows, in nats">H̄ {fmt(a.entropy[h].reduce((s, x) => s + x, 0) / a.entropy[h].length, 2)}</span></div>
                        <Heatmap data={w} kind="sequential" domain={[0, 1]} maxCell={12} maxHeight={150} legend={false} selectedRow={position} rowName="query" colName="key" valueName="weight" />
                      </button>
                    ))}
                  </div>
                </>
              ) : <Loading />}
            </TabsContent>
            <TabsContent value="layers" className="mt-3">
              <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-ink-2">
                <span>One matrix per layer. Click a layer to select it.</span>
                <Field label="show"><Select value={reduce} onChange={(e) => setReduce(e.target.value)}><option value="mean">mean over heads</option><option value="max">max over heads</option><option value="head">head {head} only</option></Select></Field>
              </div>
              <ErrorNotice error={grid.error} />
              {grid.data ? (
                <>
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4 2xl:grid-cols-6">
                    {grid.data.matrices.map((m, l) => (
                      <button key={l} onClick={() => set({ layer: l })} className={cn("cursor-pointer rounded-md border p-1.5 text-left", l === layer ? "border-accent bg-accent-soft" : "border-line hover:border-line-strong")}>
                        <div className="mb-1 text-[11px] font-medium text-ink">layer {l}</div>
                        <Heatmap data={m} kind="sequential" domain={[0, 1]} maxCell={12} maxHeight={150} legend={false} selectedRow={position} rowName="query" colName="key" valueName="weight" />
                      </button>
                    ))}
                  </div>
                  {grid.data.note && <VizNote>{grid.data.note}</VizNote>}
                </>
              ) : <Loading />}
            </TabsContent>
            <TabsContent value="qkv" className="mt-3"><QkvTab {...ctx} /></TabsContent>
            <TabsContent value="scores" className="mt-3"><ScoresTab {...ctx} /></TabsContent>
            <TabsContent value="softmax" className="mt-3"><SoftmaxTab {...ctx} /></TabsContent>
            <TabsContent value="output" className="mt-3"><OutputTab {...ctx} /></TabsContent>
          </Tabs>
        </CardBody>
      </Card>
    </div>
  );
}
