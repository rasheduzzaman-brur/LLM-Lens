"use client";
import { useQuery } from "@tanstack/react-query";
import { api, API_URL } from "@/services/api";
import { useStore } from "@/hooks/use-store";
import { useModel, useRun, useStepIndex } from "@/hooks/use-lens";
import { NeedModel, PageHeader, ProvenanceBadge, useTrace } from "@/components/common";
import { TraceTable } from "@/components/layout/bottom-panel";
import { Badge, Card, CardBody, CardHeader, Field, Input, Notice, Select, Stat } from "@/components/ui/primitives";
import { bytes, ms } from "@/lib/utils";
import type { CaptureConfig } from "@/types/api";

const EXAMPLE = { event_id: 1, stage: "attention", layer: 0, operation: "query_projection", tensor_name: "q", shape: [1, 12, 768], dtype: "float32", provenance: "hooked", execution_time_ms: 1.2 };

export default function TracesPage() {
  const model = useModel();
  const run = useRun();
  const stepIdx = useStepIndex(run);
  const capture = useStore((s) => s.capture);
  const set = useStore((s) => s.set);
  const trace = useTrace(run?.run_id ?? null, stepIdx, 0);
  const sys = useQuery({ queryKey: ["system"], queryFn: api.system, refetchInterval: 5000 });
  if (!model) return <><PageHeader title="Computation Traces" /><NeedModel /></>;
  const up = (p: Partial<CaptureConfig>) => set({ capture: { ...capture, ...p } });
  const evs = trace.data?.events ?? [];
  const count = (pv: string) => evs.filter((e) => e.provenance === pv).length;
  const checked = evs.filter((e) => e.verified !== null);
  const unavailable = evs.filter((e) => e.provenance === "unavailable");

  return (
    <div className="space-y-3">
      <PageHeader title="Computation Traces" sub="Every operation of a forward pass is recorded as a trace event in execution order: where it happened, what it computed, the tensor's real shape and dtype, and how the value was obtained." />
      <div className="grid gap-3 xl:grid-cols-2">
        <Card>
          <CardHeader title="Tensor capture for the next run" sub="Controls how much tensor data is kept in memory. Shapes, dtypes and statistics are always recorded." />
          <CardBody className="space-y-2.5">
            <div className="flex flex-wrap items-center gap-3">
              <Field label="mode"><Select value={capture.mode} onChange={(e) => up({ mode: e.target.value as CaptureConfig["mode"] })}><option value="full">full tensor capture</option><option value="selected">selected tensors</option><option value="metadata">metadata only</option></Select></Field>
              {run && <Badge>current run: {run.capture.mode}</Badge>}
            </div>
            {capture.mode === "selected" && (
              <div className="grid gap-2 sm:grid-cols-2">
                <Field label="tensor names" className="items-start"><Input className="num" value={capture.tensors.join(", ")} placeholder="weights, resid_post, attn_out" onChange={(e) => up({ tensors: e.target.value.split(",").map((s) => s.trim()).filter(Boolean) })} /></Field>
                <Field label="layers"><Input className="num" value={capture.layers.join(", ")} placeholder="all (or e.g. 0, 5, 11)" onChange={(e) => up({ layers: e.target.value.split(",").map((s) => parseInt(s.trim(), 10)).filter((n) => !Number.isNaN(n)) })} /></Field>
              </div>
            )}
            <Notice>{capture.mode === "full" ? "Every traced tensor is stored on the model's device and sliced on demand." : capture.mode === "selected" ? "Only the named tensors (in the listed layers) are stored. Views that need other tensors report that they are unavailable." : "No tensors are stored except the next-token logits needed for token selection. Verification still runs during the forward pass, so every check result is real."} Press Run in the top bar to start a new run with this setting.</Notice>
            {sys.data && <div className="grid grid-cols-3 gap-3"><Stat label="Stored now" value={bytes(sys.data.trace_store.stored_bytes)} sub={`budget ${bytes(sys.data.trace_store.budget_bytes)}`} /><Stat label="Runs kept" value={`${sys.data.trace_store.runs} / ${sys.data.trace_store.max_runs}`} /><Stat label="Decode steps with tensors" value={sys.data.trace_store.retain_steps} sub="older steps keep metadata" /></div>}
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="How each value was obtained" />
          <CardBody className="space-y-2 text-xs leading-relaxed text-ink-2">
            <div className="flex gap-2"><ProvenanceBadge p="hooked" /><span>Captured by a PyTorch forward hook on the real module (its input or output). Module time is measured around the module call.</span></div>
            <div className="flex gap-2"><ProvenanceBadge p="derived" /><span>Has no module boundary (for example Q·Kᵀ, the causal mask, softmax, a normalization's mean). Recomputed from hooked tensors with the model's own parameters, then compared with a tensor the model produced; the result is the “verified” badge.</span></div>
            <div className="flex gap-2"><ProvenanceBadge p="viz" /><span>PCA, cosine similarity, pooling of large matrices, means over heads. Shown for orientation; never part of the model computation.</span></div>
            <div className="flex gap-2"><ProvenanceBadge p="unavailable" /><span>The operation exists but cannot be inspected with this adapter; the event says why.</span></div>
            <pre className="num overflow-x-auto rounded-md border border-line bg-surface-2 p-2 text-[11px] text-ink">{JSON.stringify(EXAMPLE, null, 1)}</pre>
          </CardBody>
        </Card>
      </div>
      {run ? (
        <Card>
          <CardHeader title={`Trace of step ${stepIdx} (${run.steps[stepIdx].phase})`} sub="Click a row to open the event and its tensor in the inspector."
            right={<><Badge tone="accent">{count("hooked")} hooked</Badge><Badge tone="derived">{count("derived")} derived</Badge><Badge tone={checked.some((e) => !e.verified) ? "bad" : "good"}>{checked.filter((e) => e.verified).length}/{checked.length} checks verified</Badge><Badge>forward {ms(trace.data?.forward_ms)}</Badge><a className="text-xs text-accent hover:underline" href={`${API_URL}/api/inference/${run.run_id}/trace-export?step=${stepIdx}`}>Download CSV</a></>} />
          {unavailable.length > 0 && <CardBody className="border-b border-line"><Notice tone="warn">Not inspectable in this run: {unavailable.map((e) => `${e.operation}${e.layer !== null ? ` (layer ${e.layer})` : ""}`).join(", ")}.</Notice></CardBody>}
          <div className="h-[560px]"><TraceTable run={run} stepIdx={stepIdx} /></div>
        </Card>
      ) : <Notice>Run a prompt to record a trace.</Notice>}
    </div>
  );
}
