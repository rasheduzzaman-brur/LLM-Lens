"use client";
import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { ChevronLeft, ChevronRight, Loader2, Pause, Play, RotateCcw, StepForward } from "lucide-react";
import { api } from "@/services/api";
import { useStore } from "@/hooks/use-store";
import { useModel, usePalette, useRun, useSetRun, useStepIndex } from "@/hooks/use-lens";
import { useOpCursor } from "@/hooks/use-op-cursor";
import { ErrorNotice, Loading, NeedModel, NeedRun, PageHeader, TensorChip } from "@/components/common";
import { Plot } from "@/components/charts/plot";
import { Badge, Button, Card, CardBody, CardHeader, Field, Formula, Input, Notice, Select, Slider, Stat } from "@/components/ui/primitives";
import { bytes, cn, fmt, ms, pct, shapeStr, showToken } from "@/lib/utils";
import type { RunSummary, SamplingConfig } from "@/types/api";

const same = (a: SamplingConfig, b: SamplingConfig) => a.strategy === b.strategy && a.temperature === b.temperature && a.top_k === b.top_k && a.top_p === b.top_p;

function SamplingControls() {
  const s = useStore((st) => st.sampling);
  const set = useStore((st) => st.set);
  const up = (p: Partial<SamplingConfig>) => set({ sampling: { ...s, ...p } });
  return (
    <div className="grid gap-x-5 gap-y-2 sm:grid-cols-2 xl:grid-cols-4">
      <Field label="decoding"><Select value={s.strategy} onChange={(e) => up({ strategy: e.target.value as SamplingConfig["strategy"] })}><option value="greedy">greedy (argmax)</option><option value="sample">sampling</option></Select></Field>
      <div className="flex items-center gap-2 text-xs text-ink-2"><span className="whitespace-nowrap">temperature</span>
        <Slider min={0.05} max={3} step={0.05} value={[s.temperature]} onValueChange={(v) => up({ temperature: v[0] })} aria-label="Temperature" /><span className="num w-9 text-ink">{s.temperature.toFixed(2)}</span></div>
      <div className="flex items-center gap-2 text-xs text-ink-2"><span className="whitespace-nowrap">top-p</span>
        <Slider min={0.05} max={1} step={0.01} value={[s.top_p]} onValueChange={(v) => up({ top_p: v[0] })} aria-label="Top-p" /><span className="num w-9 text-ink">{s.top_p.toFixed(2)}</span></div>
      <div className="flex items-center gap-3">
        <Field label="top-k"><Input type="number" min={0} value={s.top_k} onChange={(e) => up({ top_k: Math.max(0, Number(e.target.value) || 0) })} className="num w-16" title="0 disables top-k" /></Field>
        <Field label="seed"><Input type="number" value={s.seed ?? 0} onChange={(e) => up({ seed: Number(e.target.value) || 0 })} className="num w-16" /></Field>
      </div>
    </div>
  );
}

function OutputText({ run, stepIdx }: { run: RunSummary; stepIdx: number }) {
  const set = useStore((s) => s.set);
  const p = usePalette();
  return (
    <div className="num rounded-md border border-line bg-surface-2 p-2.5 text-sm leading-7">
      <span className="text-ink-2">{run.prompt}</span>
      {run.steps.map((s) => (
        <motion.button key={s.index} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.18 }} onClick={() => set({ step: s.index, position: -1 })}
          title={`selected at step ${s.index} with probability ${pct(s.sampling.prob_final, 1)}`}
          className={cn("cursor-pointer rounded-sm px-px whitespace-pre-wrap text-ink", s.index === stepIdx && "outline-2 outline-accent")}
          style={{ background: p.sequential(0.15 + 0.5 * s.sampling.prob_final) }}>{s.sampling.token.replace(/\n/g, "↵\n")}</motion.button>
      ))}
    </div>
  );
}

export default function GenerationPage() {
  const model = useModel();
  const run = useRun();
  const p = usePalette();
  const stepIdx = useStepIndex(run);
  const sampling = useStore((s) => s.sampling);
  const set = useStore((s) => s.set);
  const setRun = useSetRun();
  const cursor = useOpCursor(run, stepIdx);
  const [topN, setTopN] = useState(10);
  const [delay, setDelay] = useState(150);
  const ok = (r: RunSummary) => { setRun(r); set({ step: -1, position: -1 }); };
  const next = useMutation({ mutationFn: () => api.step(run!.run_id, { kind: "next_token", sampling }), onSuccess: ok });
  const resume = useMutation({ mutationFn: () => api.resume(run!.run_id, { delay_ms: delay, sampling }), onSuccess: ok });
  const pause = useMutation({ mutationFn: () => api.pause(run!.run_id), onSuccess: ok });
  const reset = useMutation({ mutationFn: () => api.reset(run!.run_id), onSuccess: (r) => { setRun(r); set({ step: -1, position: -1, selection: null }); } });
  const step = run?.steps[stepIdx];
  const whatIf = !!step && !same(step.sampling.config, sampling);
  const lg = useQuery({
    queryKey: ["logits", run?.run_id, stepIdx, topN, whatIf ? sampling : null],
    queryFn: () => api.logits(run!.run_id, { step: stepIdx, top_n: topN, ...(whatIf ? { temperature: sampling.temperature, top_k: sampling.top_k, top_p: sampling.top_p, strategy: sampling.strategy } : {}) }),
    enabled: !!run, staleTime: Infinity, retry: false, placeholderData: (prev) => prev,
  });
  const grid = useQuery({ queryKey: ["agrid", run?.run_id, stepIdx, "mean", 0], queryFn: () => api.attentionGrid(run!.run_id, stepIdx, "mean", 0), enabled: !!run, staleTime: Infinity, retry: false });
  if (!model) return <><PageHeader title="Generation" /><NeedModel /></>;
  if (!run || !step) return <><PageHeader title="Generation" /><NeedRun model={model} /></>;
  const running = run.state === "running", finished = run.state === "finished";
  const l = lg.data;
  const lastRow = grid.data ? grid.data.matrices[grid.data.matrices.length - 1][step.seq_len - 1] : null;
  const fresh = run.steps.length === 1 && run.state === "ready";
  const err = next.error ?? resume.error ?? pause.error ?? reset.error;

  return (
    <div className="space-y-3">
      <PageHeader title="Step-by-Step Generation" sub="Autoregressive decoding, one real forward pass per token. The prompt is processed once (prefill); each later step feeds only the newest token and reuses the KV cache." />
      <Card>
        <CardHeader title="Controls" sub={finished ? `Finished: ${run.stop_reason === "eos" ? "end-of-sequence token" : `reached ${run.max_new_tokens} new tokens`}. Reset to generate again.` : `${run.steps.length} of up to ${run.max_new_tokens} tokens selected.`}
          right={<Badge tone={running ? "accent" : "neutral"}>{running && <Loader2 className="size-3 animate-spin" />}{run.state}</Badge>} />
        <CardBody className="space-y-3">
          <div className="flex flex-wrap items-center gap-1.5">
            {running
              ? <Button variant="default" onClick={() => pause.mutate()} disabled={pause.isPending}><Pause />Pause</Button>
              : <Button variant="default" onClick={() => resume.mutate()} disabled={finished || resume.isPending}><Play />{fresh ? "Start" : "Resume"}</Button>}
            <Button onClick={() => next.mutate()} disabled={running || finished || next.isPending}>{next.isPending ? <Loader2 className="animate-spin" /> : <StepForward />}Next token</Button>
            <span className="mx-1 h-5 w-px bg-line" />
            <Button onClick={cursor.prev} disabled={cursor.index <= 0} title="Previous operation of the selected step"><ChevronLeft />Previous operation</Button>
            <Button onClick={cursor.next} disabled={cursor.atEnd} title="Next operation of the selected step">Next operation<ChevronRight /></Button>
            <span className="mx-1 h-5 w-px bg-line" />
            <Button variant="danger" onClick={() => reset.mutate()} disabled={reset.isPending}><RotateCcw />Reset</Button>
            <Field label="delay between tokens" className="ml-auto"><Select value={delay} onChange={(e) => setDelay(Number(e.target.value))}><option value={0}>none</option><option value={150}>150 ms</option><option value={500}>500 ms</option><option value={1500}>1.5 s</option></Select></Field>
          </div>
          <ErrorNotice error={err} />
          {cursor.current && <div className="num rounded-md border border-line bg-surface-2 px-2 py-1 text-[11px] text-ink-2">operation {cursor.index + 1}/{cursor.total} of step {stepIdx}: {cursor.current.layer !== null ? `layer ${cursor.current.layer} · ` : ""}<span className="text-ink">{cursor.current.operation}</span> → {cursor.current.tensor_name} {shapeStr(cursor.current.shape)} (details in the inspector; position shown in the Transformer Layers diagram)</div>}
          <SamplingControls />
          <p className="text-[11px] leading-4 text-muted">Sampling settings apply to the tokens generated from now on. Changing them also previews, below, how the distribution of the selected step would look (the token already selected in that step does not change).</p>
          <OutputText run={run} stepIdx={stepIdx} />
          <p className="text-[11px] text-muted">Generated tokens are shaded by the probability they were selected with (darker surface = lower). Click a token to inspect its step.</p>
        </CardBody>
      </Card>

      <div className="grid gap-3 xl:grid-cols-[minmax(300px,400px)_minmax(0,1fr)]">
        <div className="space-y-3">
          <Card>
            <CardHeader title={`Step ${step.index}: ${step.phase}`} right={<Badge tone={step.phase === "prefill" ? "accent" : "derived"}>{step.phase}</Badge>} />
            <CardBody className="space-y-2.5">
              <Notice>{step.phase === "prefill"
                ? <>The complete prompt ({step.new_len} tokens) goes through every layer in one pass. Attention matrices are {step.new_len}×{step.new_len} and the KV cache is filled for all positions.</>
                : <>Only the newest token “<span className="num text-ink">{showToken(run.tokens[step.seq_len - 1].token)}</span>” is fed in. Its Query attends over {step.seq_len} cached Key/Value rows ({step.past_len} reused, 1 appended): attention is 1×{step.seq_len}.</>}</Notice>
              <div className="grid grid-cols-2 gap-2.5">
                <Stat label="Input this step" value={`${step.new_len} token${step.new_len > 1 ? "s" : ""}`} sub={`sequence length ${step.seq_len}`} />
                <Stat label="Current token" value={`“${showToken(run.tokens[step.seq_len - 1].token)}”`} sub={`id ${run.tokens[step.seq_len - 1].token_id} · position ${step.seq_len - 1}`} />
                <Stat label="Forward pass" value={ms(step.forward_ms)} sub="including hook overhead" /><Stat label="KV cache" value={bytes(step.cache_bytes)} sub={`${step.seq_len} positions`} />
                <Stat label="Selected next token" value={`“${showToken(step.sampling.token)}”`} sub={`id ${step.sampling.token_id}`} />
                <Stat label="Its probability" value={pct(step.sampling.prob_final, 2)} sub={`rank ${step.sampling.rank + 1} · logit ${fmt(step.sampling.logit, 4)}`} />
              </div>
              <div className="flex flex-wrap gap-1 text-[11px] text-muted">inspect:
                <TensorChip name="final_norm_out" layer={null} step={stepIdx} label="final hidden state" /><TensorChip name="logits" layer={null} step={stepIdx} />
                <TensorChip name="next_token_logits" layer={null} step={stepIdx} /><TensorChip name="next_token_probs" layer={null} step={stepIdx} /><TensorChip name="sampling_probs" layer={null} step={stepIdx} />
              </div>
            </CardBody>
          </Card>
          <Card>
            <CardHeader title="What the current token attends to" sub={`Last layer, mean over heads, row of position ${step.seq_len - 1}.`} />
            <CardBody>
              {lastRow ? <Plot height={Math.max(150, 20 * step.seq_len + 46)} data={[{ type: "bar", orientation: "h", y: grid.data!.tokens.map((t, i) => `${i} ${showToken(t)}`), x: lastRow as number[], marker: { color: p.series[0] }, hovertemplate: "%{y}<br>mean weight %{x:.4f}<extra></extra>" }]}
                layout={{ margin: { l: 96, r: 10, t: 6, b: 30 }, xaxis: { range: [0, 1], title: { text: "attention weight" } }, yaxis: { autorange: "reversed", tickfont: { size: 10, family: "ui-monospace, monospace" } } }} />
                : grid.error ? <ErrorNotice error={grid.error} /> : <Loading />}
              <p className="text-[11px] text-muted">Visualization transform: mean over heads. Per-head detail is in the Attention Explorer.</p>
            </CardBody>
          </Card>
        </div>

        <Card>
          <CardHeader title="Logits and next-token probabilities" sub={l ? `Distribution over ${l.vocab_size.toLocaleString()} vocabulary tokens after “${showToken(l.context_token)}”.` : undefined}
            right={<><Field label="show top"><Select value={topN} onChange={(e) => setTopN(Number(e.target.value))}><option value={5}>5</option><option value={10}>10</option><option value={50}>50</option></Select></Field>{l?.what_if && <Badge tone="warn">preview of current settings</Badge>}</>} />
          <CardBody className="space-y-3">
            <Formula>{"final hidden state → LM head → logits → ÷ temperature → top-k / top-p filter → softmax → select\np(token) = exp(logit / T) / Σ exp(logit / T)"}</Formula>
            <ErrorNotice error={lg.error} />
            {!l && !lg.error && <Loading />}
            {l && (
              <>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <Stat label="Temperature" value={l.config.temperature.toFixed(2)} sub={l.config.strategy} /><Stat label="Tokens kept by filters" value={l.kept.toLocaleString()} sub={`of ${l.vocab_size.toLocaleString()}`} />
                  <Stat label="Entropy (T = 1)" value={`${fmt(l.entropy_raw, 4)} nats`} /><Stat label="Entropy (these settings)" value={`${fmt(l.entropy_final, 4)} nats`} />
                </div>
                <Plot height={Math.max(220, 19 * l.top.length + 70)} data={[
                  { type: "bar", orientation: "h", name: "softmax at T = 1", y: l.top.map((r) => `${r.rank + 1}. ${showToken(r.token)}`), x: l.top.map((r) => r.prob_raw), marker: { color: p.series[0] }, hovertemplate: "%{y}<br>p %{x:.5f}<extra>T = 1</extra>" },
                  { type: "bar", orientation: "h", name: `with T = ${l.config.temperature}${l.config.top_k ? `, top-k ${l.config.top_k}` : ""}${l.config.top_p < 1 ? `, top-p ${l.config.top_p}` : ""}`, y: l.top.map((r) => `${r.rank + 1}. ${showToken(r.token)}`), x: l.top.map((r) => r.prob_final), marker: { color: p.series[1] }, hovertemplate: "%{y}<br>p %{x:.5f}<extra>these settings</extra>" },
                ]} layout={{ barmode: "group", margin: { l: 120, r: 12, t: 26, b: 34 }, xaxis: { title: { text: "probability" }, rangemode: "tozero" }, yaxis: { autorange: "reversed", tickfont: { size: 10, family: "ui-monospace, monospace" } } }} />
                <div className="max-h-80 overflow-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="sticky top-0 bg-surface text-[11px] text-muted"><tr>{["rank", "token", "id", "logit", "p (T = 1)", "p after temperature", "p after filters", "kept"].map((h) => <th key={h} className="border-b border-line px-2 py-1 font-medium">{h}</th>)}</tr></thead>
                    <tbody>{l.top.map((r) => (
                      <tr key={r.token_id} className={cn("border-b border-line/60", r.token_id === l.selected.token_id && "bg-accent-soft")}>
                        <td className="num px-2 py-0.5 text-muted">{r.rank + 1}</td><td className="num px-2 text-ink">“{showToken(r.token)}”{r.token_id === l.selected.token_id && <Badge tone="accent" className="ml-1.5">selected</Badge>}</td>
                        <td className="num px-2">{r.token_id}</td><td className="num px-2">{fmt(r.logit, 5)}</td><td className="num px-2">{pct(r.prob_raw, 3)}</td><td className="num px-2">{pct(r.prob_temperature, 3)}</td>
                        <td className="num px-2 font-semibold text-ink">{pct(r.prob_final, 3)}</td><td className="px-2">{r.kept ? "yes" : <span className="text-muted">filtered out</span>}</td>
                      </tr>))}</tbody>
                  </table>
                </div>
                <div className="num text-xs text-ink-2">Selected in this step: “<span className="text-ink">{showToken(l.selected.token)}</span>” (id {l.selected.token_id}) by {l.selected.config.strategy} decoding, probability {pct(l.selected.prob_final, 2)}, rank {l.selected.rank + 1}.
                  {l.what_if && <> Under the previewed settings it would have probability {pct(l.selected.prob_under_view, 2)}{l.selected.kept_under_view ? "" : " (filtered out)"}.</>}</div>
                <p className="text-[11px] leading-4 text-muted">{l.note} Logit range in this step: {fmt(l.logit_stats.min)} to {fmt(l.logit_stats.max)} ({l.dtype}).</p>
              </>
            )}
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
