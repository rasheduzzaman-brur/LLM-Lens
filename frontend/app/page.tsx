"use client";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight } from "lucide-react";
import { api } from "@/services/api";
import { useModel, useRun } from "@/hooks/use-lens";
import { NeedModel, PageHeader } from "@/components/common";
import { Badge, Card, CardBody, CardHeader, Notice, Stat } from "@/components/ui/primitives";
import { bytes, compact, ms, showToken } from "@/lib/utils";

const PIPELINE: { href: string; title: string; text: string }[] = [
  { href: "/tokenization", title: "1 · Tokenization", text: "Text → tokens → ids → embedding rows." },
  { href: "/layers", title: "2 · Transformer layers", text: "Architecture diagram, normalization, residual stream, MLP." },
  { href: "/attention", title: "3 · Attention", text: "Q, K, V → QKᵀ/√d → causal mask → softmax → weights·V, per head." },
  { href: "/hidden-states", title: "4 · Hidden states", text: "How each token's vector changes through depth." },
  { href: "/generation", title: "5 · Generation", text: "Logits → probabilities → token selection, step by step." },
  { href: "/kv-cache", title: "6 · KV cache", text: "What is reused between decode steps, and what it saves." },
];

export default function Dashboard() {
  const model = useModel();
  const run = useRun();
  const sys = useQuery({ queryKey: ["system"], queryFn: api.system, refetchInterval: 5000 });
  if (!model) return <><PageHeader title="Dashboard" sub="LLM Lens runs a real Transformer and exposes every intermediate computation of its forward pass." /><NeedModel /></>;
  const c = model.capabilities;
  return (
    <div className="space-y-3">
      <PageHeader title="Dashboard" sub="Everything shown in this application is read from an actual forward pass of the loaded model. Derived quantities are recomputed from captured tensors and checked against the model's own outputs." />
      <div className="grid gap-3 xl:grid-cols-2">
        <Card>
          <CardHeader title={model.name} sub={`${model.model_class} · ${model.family} adapter`} right={<><Badge tone="accent">{model.device}</Badge><Badge>{model.dtype}</Badge></>} />
          <CardBody className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Stat label="Layers" value={model.n_layers} /><Stat label="Attention heads" value={model.n_heads} sub={c.grouped_query_attention ? `${model.n_kv_heads} K/V heads (GQA)` : `head dim ${model.head_dim}`} />
            <Stat label="Hidden size" value={model.hidden_size} /><Stat label="MLP size" value={model.intermediate_size} />
            <Stat label="Vocabulary" value={model.vocab_size.toLocaleString()} /><Stat label="Parameters" value={compact(model.parameter_count)} />
            <Stat label="Normalization" value={c.norm === "layernorm" ? "LayerNorm" : "RMSNorm"} sub={`${c.norm_placement}-norm`} /><Stat label="MLP" value={c.mlp === "standard" ? c.activation : "SwiGLU"} sub={c.positional === "rope" ? "RoPE positions" : "learned positions"} />
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Current run" sub={run ? `run ${run.run_id}` : "No run yet"} right={run && <Badge tone={run.checks.failed ? "bad" : "good"}>{run.checks.total - run.checks.failed}/{run.checks.total} checks verified</Badge>} />
          <CardBody>
            {run ? (
              <div className="space-y-3">
                <div className="num rounded-md border border-line bg-surface-2 p-2 text-xs leading-relaxed"><span className="text-ink-2">{run.prompt}</span><span className="text-accent">{run.generated_text}</span></div>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <Stat label="Prompt tokens" value={run.n_prompt_tokens} /><Stat label="Generated" value={run.steps.length} sub={run.state} />
                  <Stat label="Prefill forward" value={ms(run.steps[0].forward_ms)} /><Stat label="Stored tensors" value={bytes(run.stored_bytes)} sub={`${run.capture.mode} capture`} />
                </div>
                <p className="text-xs text-ink-2">Last selected token: <span className="num text-ink">“{showToken(run.steps[run.steps.length - 1].sampling.token)}”</span></p>
              </div>
            ) : <p className="text-xs text-ink-2">Type a prompt in the bar above and press Run to execute a traced forward pass.</p>}
          </CardBody>
        </Card>
      </div>
      <Card>
        <CardHeader title="Follow the computation" sub="Each section inspects one stage of the same run." />
        <CardBody className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
          {PIPELINE.map((p) => (
            <Link key={p.href} href={p.href} className="group rounded-md border border-line p-2.5 hover:border-accent">
              <div className="flex items-center justify-between text-xs font-semibold">{p.title}<ArrowRight className="size-3.5 text-muted group-hover:text-accent" /></div>
              <p className="mt-0.5 text-xs text-ink-2">{p.text}</p>
            </Link>
          ))}
        </CardBody>
      </Card>
      {sys.data && (
        <Card>
          <CardHeader title="Hardware" sub={`PyTorch ${sys.data.torch} · Python ${sys.data.python}`} />
          <CardBody className="space-y-2">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Stat label="CPU cores" value={sys.data.cpu.cores} sub={`${sys.data.cpu.torch_threads} torch threads`} />
              <Stat label="RAM available" value={bytes(sys.data.cpu.ram_available)} sub={`of ${bytes(sys.data.cpu.ram_total)}`} />
              <Stat label="Backend process" value={bytes(sys.data.cpu.process_rss)} />
              <Stat label="Trace store" value={bytes(sys.data.trace_store.stored_bytes)} sub={`budget ${bytes(sys.data.trace_store.budget_bytes)}`} />
            </div>
            {sys.data.cuda.available
              ? sys.data.cuda.devices.map((d) => <div key={d.index} className="num text-xs text-ink-2">cuda:{d.index} {d.name} · {bytes(d.total - d.free)} used of {bytes(d.total)} · {bytes(d.allocated)} allocated by PyTorch</div>)
              : <Notice>{sys.data.cuda.note}</Notice>}
          </CardBody>
        </Card>
      )}
      {model.precision_note && <Notice tone="warn">{model.precision_note}</Notice>}
    </div>
  );
}
