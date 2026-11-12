"use client";
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/services/api";
import { useStore } from "@/hooks/use-store";
import { usePalette } from "@/hooks/use-lens";
import { VerifiedBadge, VizNote, Loading } from "@/components/common";
import { Plot } from "@/components/charts/plot";
import { Heatmap, VectorHeat } from "@/components/tensors/heatmap";
import { Badge, Field, Formula, Notice, Select, Slider } from "@/components/ui/primitives";
import { fmt } from "@/lib/utils";
import type { LayerDetail, ModelInfo, NormParams } from "@/types/api";

const l2 = (v: number[]) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));

function DimPicker({ n, value, onChange }: { n: number; value: number; onChange: (v: number) => void }) {
  return (
    <div className="flex items-center gap-2 text-xs text-ink-2">
      <span className="whitespace-nowrap">element index</span>
      <Slider className="max-w-64" min={0} max={n - 1} step={1} value={[value]} onValueChange={(v) => onChange(v[0])} aria-label="Element index" />
      <span className="num w-10 text-ink">{value}</span>
    </div>
  );
}

function Num({ label, value }: { label: React.ReactNode; value: number | null | undefined }) {
  return <div className="rounded-md border border-line px-2 py-1"><div className="text-[11px] text-muted">{label}</div><div className="num text-[13px] font-semibold text-ink">{fmt(value, 6)}</div></div>;
}

function Inspectable({ name, layer, children }: { name: string; layer: number | null; children: React.ReactNode }) {
  const select = useStore((s) => s.select);
  const step = useStore((s) => s.step);
  return <button className="num cursor-pointer text-accent hover:underline" onClick={() => select({ kind: "tensor", name, layer, step })}>{children}</button>;
}

/** Normalization of one token, every intermediate value read from the run. */
export function NormPanel({ d, prefix, inputName, params, layer }: { d: LayerDetail; prefix: string; inputName: string; params: NormParams; layer: number | null }) {
  const x = d.vectors[inputName], norm = d.vectors[`${prefix}_normalized`], out = d.vectors[`${prefix}_out`];
  const [i, setI] = useState(0);
  if (!x || !out) return <Notice tone="warn">Normalization tensors for this position are not stored.</Notice>;
  if (!norm) return <Notice tone="warn">This normalization module ({params.type}) cannot be decomposed by the adapter; only its input and output are available.</Notice>;
  const isLN = `${prefix}_mean` in d.scalars;
  const k = Math.min(i, x.length - 1);
  const g = params.gamma?.[k], b = params.beta?.[k];
  const mean = d.scalars[`${prefix}_mean`], vari = d.scalars[`${prefix}_var`], msq = d.scalars[`${prefix}_meansq`], rms = d.scalars[`${prefix}_rms`];
  const rows = isLN ? ["x", "normalized", "γ (gamma)", "β (beta)", "output"] : ["x", "normalized", "weight", "output"];
  const data = isLN ? [x, norm, params.gamma!, params.beta!, out] : [x, norm, params.gamma!, out];
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge>{params.type}</Badge><Badge>ε = {params.eps}</Badge><VerifiedBadge check={d.checks[`${prefix}_out`]} />
        <span className="text-[11px] text-muted">step-by-step result compared with the module's actual output</span>
      </div>
      <Formula>{isLN
        ? "mean       = mean(x)\nvariance   = mean((x − mean)²)\nnormalized = (x − mean) / √(variance + ε)\noutput     = normalized · γ + β"
        : "rms        = √(mean(x²) + ε)\nnormalized = x / rms\noutput     = normalized · weight"}</Formula>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {isLN ? <><Num label="mean(x)" value={mean} /><Num label="variance" value={vari} /></> : <><Num label="mean(x²)" value={msq} /><Num label="rms" value={rms} /></>}
        <Num label="‖x‖₂" value={l2(x)} /><Num label="‖output‖₂" value={l2(out)} />
      </div>
      <div>
        <div className="mb-1 text-[11px] text-muted">All {x.length} elements for token “{d.token}”. Rows: {rows.join(" · ")}. Each row has its own magnitude, so compare patterns, not colors, across rows.</div>
        {data.map((v, r) => (
          <div key={rows[r]} className="mb-1.5">
            <div className="mb-0.5 flex items-center gap-2 text-[11px] text-ink-2">
              {r === 0 ? <Inspectable name={inputName} layer={layer}>{rows[r]} = {inputName}</Inspectable> : rows[r] === "normalized" ? <Inspectable name={`${prefix}_normalized`} layer={layer}>normalized</Inspectable>
                : rows[r] === "output" ? <Inspectable name={`${prefix}_out`} layer={layer}>output = {prefix}_out</Inspectable> : <span className="num">{rows[r]} <span className="text-muted">(model parameter)</span></span>}
            </div>
            <Heatmap data={[v]} maxCell={8} minCell={1} legend={false} rowHeight={16} valueName={rows[r]} describe={(_r, c) => `element ${c}`} />
          </div>
        ))}
      </div>
      <div className="space-y-2 rounded-md border border-line p-2.5">
        <DimPicker n={x.length} value={k} onChange={setI} />
        <div className="num overflow-x-auto text-xs leading-6 whitespace-nowrap text-ink">
          {isLN ? (
            <>
              <div>x[{k}] = <b>{fmt(x[k], 6)}</b></div>
              <div>normalized[{k}] = ({fmt(x[k], 6)} − {fmt(mean, 6)}) / √({fmt(vari, 6)} + {params.eps}) = <b>{fmt(norm[k], 6)}</b></div>
              <div>output[{k}] = {fmt(norm[k], 6)} · γ[{k}] + β[{k}] = {fmt(norm[k], 6)} · {fmt(g, 6)} + {fmt(b, 6)} = <b>{fmt(out[k], 6)}</b></div>
            </>
          ) : (
            <>
              <div>x[{k}] = <b>{fmt(x[k], 6)}</b></div>
              <div>normalized[{k}] = {fmt(x[k], 6)} / {fmt(rms, 6)} = <b>{fmt(norm[k], 6)}</b></div>
              <div>output[{k}] = {fmt(norm[k], 6)} · weight[{k}] = {fmt(norm[k], 6)} · {fmt(g, 6)} = <b>{fmt(out[k], 6)}</b></div>
            </>
          )}
        </div>
        <p className="text-[11px] text-muted">Every number on these lines is read from the run or the model's parameters; nothing is recalculated in the browser.</p>
      </div>
    </div>
  );
}

/** Residual additions around attention and the MLP for one token. */
export function ResidualPanel({ d, layer }: { d: LayerDetail; layer: number }) {
  const p = usePalette();
  const v = d.vectors;
  const [i, setI] = useState(0);
  if (!v.resid_pre || !v.attn_out || !v.resid_mid || !v.mlp_out || !v.resid_post) return <Notice tone="warn">Residual tensors for this position are not stored.</Notice>;
  const k = Math.min(i, v.resid_pre.length - 1);
  const names = ["resid_pre", "attn_out", "resid_mid", "mlp_out", "resid_post"] as const;
  const labels = ["input x", "attention output", "x + attention", "MLP output", "layer output"];
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-muted">
        <span className="text-ink-2">attention add</span><VerifiedBadge check={d.checks.resid_mid} /><span className="ml-2 text-ink-2">MLP add</span><VerifiedBadge check={d.checks.resid_post} />
      </div>
      <Formula>{"resid_mid  = resid_pre + attn_out      (attention writes into the residual stream)\nresid_post = resid_mid + mlp_out       (the MLP writes into the residual stream)"}</Formula>
      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_260px]">
        <div className="min-w-0">
          <div className="mb-1 text-[11px] text-muted">All {v.resid_pre.length} elements, one row per tensor, on one shared color scale so magnitudes are comparable.</div>
          <Heatmap data={names.map((n) => v[n])} rowLabels={labels} plainLabels maxCell={8} minCell={1} rowHeight={20} selectedCol={k} onSelect={(_r, c) => setI(c)} colName="element" rowName="" />
        </div>
        <Plot height={190} data={[{ type: "bar", orientation: "h", y: [...labels].reverse(), x: [...names].reverse().map((n) => l2(v[n])), marker: { color: p.series[0] }, hovertemplate: "%{y}<br>L2 norm %{x:.4f}<extra></extra>" }]}
          layout={{ margin: { l: 104, r: 10, t: 6, b: 30 }, xaxis: { title: { text: "L2 norm" } } }} />
      </div>
      <div className="space-y-2 rounded-md border border-line p-2.5">
        <DimPicker n={v.resid_pre.length} value={k} onChange={setI} />
        <div className="num overflow-x-auto text-xs leading-6 whitespace-nowrap text-ink">
          <div><Inspectable name="resid_pre" layer={layer}>resid_pre</Inspectable>[{k}] + <Inspectable name="attn_out" layer={layer}>attn_out</Inspectable>[{k}] = {fmt(v.resid_pre[k], 6)} + {fmt(v.attn_out[k], 6)} = <b>{fmt(v.resid_mid[k], 6)}</b> = <Inspectable name="resid_mid" layer={layer}>resid_mid</Inspectable>[{k}]</div>
          <div><Inspectable name="resid_mid" layer={layer}>resid_mid</Inspectable>[{k}] + <Inspectable name="mlp_out" layer={layer}>mlp_out</Inspectable>[{k}] = {fmt(v.resid_mid[k], 6)} + {fmt(v.mlp_out[k], 6)} = <b>{fmt(v.resid_post[k], 6)}</b> = <Inspectable name="resid_post" layer={layer}>resid_post</Inspectable>[{k}]</div>
        </div>
      </div>
      <VizNote>L2 norms in the bar chart are computed in the browser from the vectors shown.</VizNote>
    </div>
  );
}

/** Feed-forward network of one token: standard (Linear → act → Linear) or gated (SwiGLU). */
export function MlpPanel({ d, layer, model }: { d: LayerDetail; layer: number; model: ModelInfo }) {
  const p = usePalette();
  const select = useStore((s) => s.select);
  const [refs, setRefs] = useState<string>("none");
  const curve = useQuery({ queryKey: ["activation", model.model_id], queryFn: () => api.activation(model.model_id), staleTime: Infinity });
  const v = d.vectors;
  const gated = d.capabilities.mlp !== "standard";
  const pre = gated ? v.mlp_gate : v.mlp_pre, post = gated ? v.mlp_gate_act : v.mlp_act;
  const act = d.capabilities.activation;
  const rows = useMemo(() => gated
    ? [["ln2_out", "input X"], ["mlp_gate", "gate = X·W_gate"], ["mlp_gate_act", `${act}(gate)`], ["mlp_up", "up = X·W_up"], ["mlp_hidden", `${act}(gate) ⊙ up`], ["mlp_out", "output"]]
    : [["ln2_out", "input X"], ["mlp_pre", "pre = X·W1 + b1"], ["mlp_act", `${act}(pre)`], ["mlp_out", "output"]], [gated, act]);
  if (!pre || !post || !v.mlp_out) return <Notice tone="warn">MLP tensors for this position are not stored.</Notice>;
  const dead = post.filter((x) => Math.abs(x) < 1e-3).length;
  const weights = gated ? ["W_gate", "W_up", "W_down"] : ["W1", "b1", "W2", "b2"];
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge>{gated ? "gated MLP (SwiGLU)" : "standard MLP"}</Badge><Badge>activation: {act}</Badge>
        <Badge>{v.ln2_out?.length} → {pre.length} → {v.mlp_out.length}</Badge>{gated && <VerifiedBadge check={d.checks.mlp_hidden} />}
      </div>
      <Formula>{gated
        ? `gate   = X · W_gate\nup     = X · W_up\nhidden = ${act}(gate) ⊙ up\noutput = hidden · W_down`
        : `hidden = ${act}(X · W1 + b1)\noutput = hidden · W2 + b2`}</Formula>
      <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-muted">weight matrices (open in the inspector):
        {weights.map((w) => <button key={w} onClick={() => select({ kind: "weight", name: w, layer })} className="num cursor-pointer rounded border border-line-strong px-1.5 py-px text-ink-2 hover:border-accent">{w}</button>)}
      </div>
      <div className="space-y-2">
        {rows.map(([name, label]) => v[name] && (
          <div key={name}>
            <div className="mb-0.5 flex items-center gap-2 text-[11px]"><Inspectable name={name} layer={layer}>{name}</Inspectable><span className="text-muted">{label} · {v[name].length} values</span></div>
            <VectorHeat values={v[name]} wrapAt={v[name].length > 1024 ? 192 : v[name].length > 256 ? 128 : 96} maxCell={7} legend={false} name="element" />
          </div>
        ))}
      </div>
      <div className="grid gap-3 lg:grid-cols-2">
        <div>
          <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
            <span className="text-xs font-medium">Activation function: {curve.data?.name ?? act}</span>
            <Field label="compare with"><Select value={refs} onChange={(e) => setRefs(e.target.value)}><option value="none">nothing</option><option value="relu">ReLU</option><option value="gelu">GELU</option><option value="silu">SiLU</option></Select></Field>
          </div>
          {curve.data ? (
            <Plot height={250} data={[
              { type: "scatter", mode: "lines", name: `${curve.data.name} (this model)`, x: curve.data.x, y: curve.data.y, line: { color: p.series[0], width: 2 }, hovertemplate: "x %{x:.3f}<br>f(x) %{y:.4f}<extra></extra>" },
              ...(refs !== "none" ? [{ type: "scatter" as const, mode: "lines" as const, name: `${refs} (reference)`, x: curve.data.x, y: curve.data.references[refs], line: { color: p.muted, width: 2, dash: "dot" as const } }] : []),
              { type: "scattergl", mode: "markers", name: `this token's ${pre.length} neurons`, x: pre, y: post, marker: { color: p.series[1], size: 4, opacity: 0.55 }, hovertemplate: "neuron %{pointNumber}<br>in %{x:.4f}<br>out %{y:.4f}<extra></extra>" },
            ]} layout={{ xaxis: { title: { text: gated ? "gate (pre-activation)" : "pre-activation" }, range: [-6, 6] }, yaxis: { title: { text: "activation output" }, range: [-1, 6] } }} />
          ) : <Loading />}
          <p className="text-[11px] leading-4 text-muted">The curve is the loaded model's activation module evaluated on a grid. Each dot is a real neuron of this token: its captured pre-activation and activation output. {refs !== "none" && "The dotted reference curve is shown for comparison only and is not used by this model."}</p>
        </div>
        <div>
          <div className="mb-1 text-xs font-medium">Distribution of this token's activations</div>
          <Plot height={250} data={[
            { type: "histogram", name: gated ? "gate" : "pre-activation", x: pre, marker: { color: p.series[0] }, opacity: 0.75, nbinsx: 60 },
            { type: "histogram", name: "after activation", x: post, marker: { color: p.series[1] }, opacity: 0.75, nbinsx: 60 },
          ]} layout={{ barmode: "overlay", bargap: 0.02, xaxis: { title: { text: "value" } }, yaxis: { title: { text: "neurons" } } }} />
          <p className="text-[11px] leading-4 text-muted"><span className="num text-ink-2">{dead}</span> of {post.length} neurons have |activation| &lt; 0.001 for this token (counted in the browser).</p>
        </div>
      </div>
    </div>
  );
}
