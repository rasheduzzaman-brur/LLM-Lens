"use client";
import { useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, SkipBack } from "lucide-react";
import { api } from "@/services/api";
import { useStore } from "@/hooks/use-store";
import { useModel, usePalette, usePosition, useRun, useStepIndex } from "@/hooks/use-lens";
import { useOpCursor } from "@/hooks/use-op-cursor";
import { ErrorNotice, LayerSelect, Loading, NeedModel, PageHeader, TensorChip, VerifiedBadge, useTrace } from "@/components/common";
import { ArchitectureFlow } from "@/components/architecture/flow";
import { MlpPanel, NormPanel, ResidualPanel } from "@/components/layers/panels";
import { Plot } from "@/components/charts/plot";
import { Badge, Button, Card, CardBody, CardHeader, Notice, Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/primitives";
import { ms, shapeStr, showToken } from "@/lib/utils";

export default function LayersPage() {
  const model = useModel();
  const run = useRun();
  const p = usePalette();
  const stepIdx = useStepIndex(run);
  const position = usePosition(run, stepIdx);
  const layerSel = useStore((s) => s.layer);
  const set = useStore((s) => s.set);
  const layer = model ? Math.min(layerSel, model.n_layers - 1) : 0;
  const arch = useQuery({ queryKey: ["arch", model?.model_id], queryFn: () => api.architecture(model!.model_id), enabled: !!model, staleTime: Infinity });
  const trace = useTrace(run?.run_id ?? null, stepIdx, 0);
  const cursor = useOpCursor(run, stepIdx);
  const detail = useQuery({ queryKey: ["layer", run?.run_id, layer, position, stepIdx], queryFn: () => api.layer(run!.run_id, layer, position, stepIdx), enabled: !!run, staleTime: Infinity, retry: false });
  const glob = useQuery({ queryKey: ["global", run?.run_id, position, stepIdx], queryFn: () => api.global(run!.run_id, position, stepIdx), enabled: !!run, staleTime: Infinity, retry: false });
  const summary = useQuery({ queryKey: ["layers", run?.run_id, stepIdx], queryFn: () => api.layers(run!.run_id, stepIdx), enabled: !!run, staleTime: Infinity });
  if (!model) return <><PageHeader title="Transformer Layers" /><NeedModel /></>;
  const d = detail.data;
  const ops = arch.data?.layer_template ?? [];

  return (
    <div className="space-y-3">
      <PageHeader title="Transformer Layers" sub={`The ${model.n_layers} layers of ${model.name}, drawn from the loaded model. Click a layer to expand its operations; click an operation to open its real tensor in the inspector.`}
        right={<LayerSelect n={model.n_layers} value={layer} onChange={(v) => set({ layer: v })} />} />
      <div className="grid gap-3 2xl:grid-cols-[minmax(380px,460px)_minmax(0,1fr)]">
        <Card>
          <CardHeader title="Architecture" sub={run ? `Shapes are the actual tensor shapes of step ${stepIdx} (${run.steps[stepIdx].phase}).` : "Symbolic shapes (T = sequence length). Run a prompt to see actual shapes."}
            right={run && (
              <>
                <Button size="icon" title="First operation" onClick={cursor.first}><SkipBack /></Button>
                <Button size="sm" title="Previous operation" onClick={cursor.prev} disabled={cursor.index <= 0}><ChevronLeft />Prev op</Button>
                <Button size="sm" title="Next operation" onClick={cursor.next} disabled={cursor.atEnd}>Next op<ChevronRight /></Button>
              </>
            )} />
          <CardBody className="space-y-2">
            {run && (
              <div className="num rounded-md border border-line bg-surface-2 px-2 py-1 text-[11px] text-ink-2">
                {cursor.current
                  ? <>operation {cursor.index + 1}/{cursor.total} · {cursor.current.layer !== null ? `layer ${cursor.current.layer} · ` : ""}<span className="text-ink">{cursor.current.operation}</span> → {cursor.current.tensor_name} {shapeStr(cursor.current.shape)}</>
                  : <>Use Prev/Next op to walk the {cursor.total} recorded operations of this step in execution order.</>}
              </div>
            )}
            {arch.data ? <ArchitectureFlow arch={arch.data} layer={layer} find={run ? trace.find : undefined} current={cursor.current} hasRun={!!run} onLayer={(i) => set({ layer: i })} /> : <Loading />}
            <p className="text-[11px] leading-4 text-muted">Dashed edges are residual (skip) connections. An orange outline marks the operation at the cursor. Stepping replays the recorded trace of a completed forward pass; it does not pause the model mid-computation.</p>
          </CardBody>
        </Card>

        <div className="min-w-0 space-y-3">
          {!run && <Notice>Run a prompt (bar above) to inspect the real intermediate calculations of layer {layer}.</Notice>}
          {run && (
            <Card>
              <CardHeader title={`Layer ${layer} · token “${showToken(run.tokens[position].token)}” (position ${position})`}
                sub={d ? `Computed in step ${d.src_step} (${d.phase}). Pick another token in the strip above.` : undefined}
                right={d && <Badge tone={Object.values(d.checks).every((c) => c.verified) ? "good" : "bad"}>{Object.values(d.checks).filter((c) => c.verified).length}/{Object.keys(d.checks).length} checks verified</Badge>} />
              <CardBody>
                {detail.isLoading && <Loading />}
                <ErrorNotice error={detail.error} />
                {d && (
                  <Tabs defaultValue="ops">
                    <TabsList>
                      <TabsTrigger value="ops">Operations</TabsTrigger><TabsTrigger value="norm1">Normalization 1</TabsTrigger><TabsTrigger value="resid">Residual connections</TabsTrigger>
                      <TabsTrigger value="norm2">Normalization 2</TabsTrigger><TabsTrigger value="mlp">Feed-forward (MLP)</TabsTrigger><TabsTrigger value="final">Embedding &amp; final norm</TabsTrigger>
                    </TabsList>
                    <TabsContent value="ops" className="mt-3">
                      <div className="overflow-x-auto">
                        <table className="w-full text-left text-xs">
                          <thead className="text-[11px] text-muted"><tr>{["#", "operation", "formula", "inputs", "outputs (click to inspect)", "actual shape", "time"].map((h) => <th key={h} className="border-b border-line px-2 py-1 font-medium">{h}</th>)}</tr></thead>
                          <tbody>
                            {ops.map((op, i) => {
                              const ev = trace.find(layer, op.out[op.out.length - 1]);
                              return (
                                <tr key={op.key} className="border-b border-line/60 align-top">
                                  <td className="num px-2 py-1 text-muted">{i + 1}</td>
                                  <td className="px-2 py-1"><div className="font-medium text-ink">{op.label}</div>{op.module && <div className="num text-[11px] text-muted">{op.module}</div>}</td>
                                  <td className="num px-2 py-1 text-ink-2">{op.formula}</td>
                                  <td className="px-2 py-1"><div className="flex flex-wrap gap-1">{op.in.map((n) => <TensorChip key={n} name={n} layer={layer} step={stepIdx} />)}</div></td>
                                  <td className="px-2 py-1"><div className="flex flex-wrap gap-1">{[...op.out, ...op.detail].map((n) => <TensorChip key={n} name={n} layer={layer} step={stepIdx} />)}</div></td>
                                  <td className="num px-2 py-1">{ev ? shapeStr(ev.shape) : "—"}{ev && <div className="mt-0.5"><VerifiedBadge check={ev} /></div>}</td>
                                  <td className="num px-2 py-1">{ms(ev?.execution_time_ms)}</td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </TabsContent>
                    <TabsContent value="norm1" className="mt-3"><NormPanel d={d} prefix="ln1" inputName="resid_pre" params={d.norm.ln1} layer={layer} /></TabsContent>
                    <TabsContent value="norm2" className="mt-3"><NormPanel d={d} prefix="ln2" inputName="resid_mid" params={d.norm.ln2} layer={layer} /></TabsContent>
                    <TabsContent value="resid" className="mt-3"><ResidualPanel d={d} layer={layer} /></TabsContent>
                    <TabsContent value="mlp" className="mt-3"><MlpPanel d={d} layer={layer} model={model} /></TabsContent>
                    <TabsContent value="final" className="mt-3 space-y-3">
                      {glob.data ? (
                        <>
                          <div className="flex flex-wrap items-center gap-1.5 text-xs text-ink-2">Embedding stage (before layer 0):
                            {["tok_emb", "pos_emb", "rope_cos", "rope_sin", "emb_out"].filter((n) => glob.data!.events.some((e) => e.tensor_name === n)).map((n) => <TensorChip key={n} name={n} layer={null} step={stepIdx} />)}
                            <VerifiedBadge check={glob.data.checks.emb_out} />
                          </div>
                          <div className="text-xs font-medium">Final normalization (after layer {model.n_layers - 1})</div>
                          <NormPanel d={glob.data} prefix="final_norm" inputName="final_norm_in" params={glob.data.norm.final_norm} layer={null} />
                        </>
                      ) : glob.error ? <ErrorNotice error={glob.error} /> : <Loading />}
                    </TabsContent>
                  </Tabs>
                )}
              </CardBody>
            </Card>
          )}
          {run && summary.data && (
            <Card>
              <CardHeader title="Residual stream through depth" sub={`L2 norm of the last position's vector (position ${summary.data.position}) at each point in every layer, step ${stepIdx}.`} />
              <CardBody>
                <Plot height={230} data={(["resid_pre", "attn_out", "mlp_out", "resid_post"] as const).map((n, i) => ({
                  type: "scatter" as const, mode: "lines+markers" as const, name: { resid_pre: "layer input", attn_out: "attention output", mlp_out: "MLP output", resid_post: "layer output" }[n],
                  x: summary.data!.layers.map((l) => l.layer), y: summary.data!.layers.map((l) => l.norms[n]), line: { color: p.series[i], width: 2 }, marker: { size: 7, line: { color: p.surface, width: 2 } },
                  hovertemplate: "layer %{x}<br>L2 norm %{y:.3f}<extra></extra>",
                }))} layout={{ xaxis: { title: { text: "layer" }, dtick: Math.ceil(model.n_layers / 12) }, yaxis: { title: { text: "L2 norm" }, rangemode: "tozero" } }}
                  onClick={(e) => set({ layer: summary.data!.layers[e.pointIndex].layer })} />
                <p className="text-[11px] text-muted">Click a point to open that layer. {summary.data.note}</p>
              </CardBody>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
