"use client";
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, HardDrive, Loader2, Trash2 } from "lucide-react";
import { api, ApiError } from "@/services/api";
import { useStore } from "@/hooks/use-store";
import { useModels } from "@/hooks/use-lens";
import { ErrorNotice, Loading, PageHeader } from "@/components/common";
import { Badge, Button, Card, CardBody, CardHeader, Field, Input, Notice, Select, Stat } from "@/components/ui/primitives";
import { bytes, compact } from "@/lib/utils";

interface Pending { repo_id: string; size_bytes: number | null }

export default function ModelsPage() {
  const qc = useQueryClient();
  const { data, error, isLoading } = useModels();
  const modelId = useStore((s) => s.modelId);
  const set = useStore((s) => s.set);
  const sys = useQuery({ queryKey: ["system"], queryFn: api.system, refetchInterval: 5000 });
  const [source, setSource] = useState("");
  const [device, setDevice] = useState("auto");
  const [dtype, setDtype] = useState("auto");
  const [pending, setPending] = useState<Pending | null>(null);

  const load = useMutation({
    mutationFn: (v: { source: string; allow_download?: boolean }) => api.loadModel({ source: v.source, device, dtype, allow_download: v.allow_download }),
    onSuccess: (m) => { setPending(null); set({ modelId: m.model_id, runId: null, selection: null, layer: 0, head: 0 }); qc.invalidateQueries({ queryKey: ["models"] }); },
    onError: (e, v) => {
      const d = e instanceof ApiError ? (e.detail as { code?: string; size_bytes?: number | null } | undefined) : undefined;
      if (d?.code === "download_required") setPending({ repo_id: v.source, size_bytes: d.size_bytes ?? null });
    },
  });
  const unload = useMutation({
    mutationFn: (id: string) => api.unloadModel(id),
    onSuccess: (_r, id) => { if (id === modelId) set({ modelId: null, runId: null, selection: null }); qc.invalidateQueries({ queryKey: ["models"] }); },
  });
  const needsDownload = load.error instanceof ApiError && (load.error.detail as { code?: string } | undefined)?.code === "download_required";

  return (
    <div className="space-y-3">
      <PageHeader title="Model Explorer" sub="Load a causal language model from a local directory or a Hugging Face model id. Weights are never downloaded without explicit confirmation." />
      <ErrorNotice error={error} />
      <Card>
        <CardHeader title="Load a model" sub={data ? `Supported architectures: ${data.supported_model_types.join(", ")}` : undefined} />
        <CardBody className="space-y-3">
          <form className="flex flex-wrap items-center gap-2" onSubmit={(e) => { e.preventDefault(); if (source.trim()) load.mutate({ source: source.trim() }); }}>
            <Input className="num min-w-52 flex-1" value={source} onChange={(e) => setSource(e.target.value)} placeholder="/path/to/model-directory  or  org/model-id" aria-label="Model source" />
            <Field label="device"><Select value={device} onChange={(e) => setDevice(e.target.value)}><option>auto</option><option>cpu</option><option disabled={!sys.data?.cuda.available}>cuda</option></Select></Field>
            <Field label="dtype"><Select value={dtype} onChange={(e) => setDtype(e.target.value)}><option value="auto">auto (float32)</option><option>float32</option><option>bfloat16</option><option disabled={!sys.data?.cuda.available}>float16</option></Select></Field>
            <Button type="submit" variant="default" disabled={load.isPending || !source.trim()}>{load.isPending ? <Loader2 className="animate-spin" /> : <HardDrive />}Load</Button>
          </form>
          {dtype !== "auto" && dtype !== "float32" && <Notice tone="warn">Reduced precision: the model computes in {dtype}. Every displayed value is that reduced-precision computation (upcast losslessly for transfer), and verification tolerances are widened accordingly.</Notice>}
          {load.error && !needsDownload ? <ErrorNotice error={load.error} /> : null}
          {pending && (
            <Notice tone="warn" className="space-y-2">
              <p><b className="text-ink">Download required.</b> <span className="num">{pending.repo_id}</span> is not available locally. {pending.size_bytes ? <>Download size is about <b className="text-ink">{bytes(pending.size_bytes)}</b>.</> : "The download size could not be determined."} It will be saved into the project's models folder.</p>
              <div className="flex gap-2">
                <Button variant="default" size="sm" disabled={load.isPending} onClick={() => load.mutate({ source: pending.repo_id, allow_download: true })}>{load.isPending ? <Loader2 className="animate-spin" /> : <Download />}Download and load</Button>
                <Button size="sm" onClick={() => setPending(null)}>Cancel</Button>
              </div>
            </Notice>
          )}
          {load.isPending && <p className="text-xs text-muted">Loading weights… larger models take a few seconds.</p>}
        </CardBody>
      </Card>

      <div className="grid gap-3 xl:grid-cols-2">
        <Card>
          <CardHeader title="Local models" sub={data ? `In ${data.local[0]?.path.split("/").slice(0, -1).join("/") ?? "the models folder"}` : undefined} />
          <CardBody className="space-y-1.5">
            {isLoading && <Loading />}
            {data?.local.length === 0 && <p className="text-xs text-ink-2">No model directories found. Enter a Hugging Face id above to download one.</p>}
            {data?.local.map((m) => (
              <div key={m.name} className="flex items-center justify-between gap-2 rounded-md border border-line px-2.5 py-1.5">
                <div className="min-w-0">
                  <div className="num truncate text-xs font-medium">{m.name}</div>
                  <div className="text-[11px] text-muted">{m.model_type} · {bytes(m.size_bytes)}{!m.supported && " · unsupported architecture"}{!m.has_weights && " · no weights"}</div>
                </div>
                {m.loaded ? <Badge tone="good">loaded</Badge> : <Button size="sm" disabled={!m.supported || !m.has_weights || load.isPending} onClick={() => load.mutate({ source: m.path })}>Load</Button>}
              </div>
            ))}
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Suggested small models" sub="Not downloaded until you confirm." />
          <CardBody className="space-y-1.5">
            {data?.suggested.map((s) => (
              <div key={s.repo_id} className="flex items-center justify-between gap-2 rounded-md border border-line px-2.5 py-1.5">
                <div className="min-w-0"><div className="num truncate text-xs font-medium">{s.repo_id}</div><div className="text-[11px] text-muted">{s.note}</div></div>
                <Button size="sm" disabled={load.isPending} onClick={() => { setSource(s.repo_id); load.mutate({ source: s.repo_id }); }}>Load</Button>
              </div>
            ))}
          </CardBody>
        </Card>
      </div>

      {data?.loaded.map((m) => (
        <Card key={m.model_id}>
          <CardHeader title={m.name} sub={<span className="num">{m.path}</span>} right={<>
            {m.model_id === modelId ? <Badge tone="accent">active</Badge> : <Button size="sm" onClick={() => set({ modelId: m.model_id, runId: null, selection: null })}>Make active</Button>}
            <Button size="sm" variant="danger" disabled={unload.isPending} onClick={() => unload.mutate(m.model_id)}><Trash2 />Unload</Button></>} />
          <CardBody className="space-y-3">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 xl:grid-cols-6">
              <Stat label="Architecture" value={m.model_type} sub={m.model_class} /><Stat label="Layers" value={m.n_layers} />
              <Stat label="Attention heads" value={m.n_heads} sub={`${m.n_kv_heads} K/V heads · dim ${m.head_dim}`} /><Stat label="Hidden dimension" value={m.hidden_size} />
              <Stat label="Intermediate dimension" value={m.intermediate_size} /><Stat label="Vocabulary size" value={m.vocab_size.toLocaleString()} />
              <Stat label="Parameters" value={compact(m.parameter_count)} sub={m.parameter_count.toLocaleString()} /><Stat label="Dtype" value={m.dtype} />
              <Stat label="Device" value={m.device} /><Stat label="Max positions" value={m.max_positions} />
              <Stat label="Tokenizer" value={m.tokenizer.class.replace("TokenizerFast", "")} sub={`${m.tokenizer.vocab_size.toLocaleString()} entries`} />
              <Stat label="Attention path" value={m.attn_implementation ?? "—"} />
            </div>
            <div className="flex flex-wrap gap-1">
              <Badge>{m.capabilities.norm === "layernorm" ? "LayerNorm" : "RMSNorm"} ({m.capabilities.norm_placement}-norm)</Badge>
              <Badge>{m.capabilities.positional === "rope" ? "rotary positions (RoPE)" : "learned absolute positions"}</Badge>
              <Badge>MLP: {m.capabilities.mlp === "standard" ? `Linear → ${m.capabilities.activation} → Linear` : `SwiGLU (${m.capabilities.activation})`}</Badge>
              <Badge>{m.capabilities.fused_qkv ? "fused Q/K/V projection" : "separate Q/K/V projections"}</Badge>
              {m.capabilities.grouped_query_attention && <Badge>grouped-query attention</Badge>}
              <Badge>Q/K/V bias: {m.capabilities.qkv_bias ? "yes" : "no"}</Badge>
            </div>
            <Notice>Loaded with the eager attention implementation so attention scores and weights exist as real tensors. Fused kernels (scaled-dot-product / flash attention) compute the same result without exposing intermediates.</Notice>
            {m.precision_note && <Notice tone="warn">{m.precision_note}</Notice>}
          </CardBody>
        </Card>
      ))}

      {sys.data && (
        <Card>
          <CardHeader title="Memory" />
          <CardBody className="space-y-2">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Stat label="RAM used" value={`${sys.data.cpu.ram_used_percent.toFixed(0)}%`} sub={`${bytes(sys.data.cpu.ram_available)} available of ${bytes(sys.data.cpu.ram_total)}`} />
              <Stat label="Backend process" value={bytes(sys.data.cpu.process_rss)} />
              <Stat label="Captured tensors" value={bytes(sys.data.trace_store.stored_bytes)} sub={`${sys.data.trace_store.runs} run(s)`} />
              <Stat label="GPU" value={sys.data.cuda.available ? `${sys.data.cuda.devices.length} device(s)` : "none"} />
            </div>
            {sys.data.cuda.devices.map((d) => (
              <div key={d.index} className="text-xs text-ink-2">
                <div className="num mb-0.5">cuda:{d.index} {d.name} · {bytes(d.total - d.free)} / {bytes(d.total)}</div>
                <div className="h-1.5 rounded-full bg-surface-3"><div className="h-full rounded-full bg-accent" style={{ width: `${(100 * (d.total - d.free)) / d.total}%` }} /></div>
              </div>
            ))}
            {!sys.data.cuda.available && <Notice>{sys.data.cuda.note}</Notice>}
          </CardBody>
        </Card>
      )}
    </div>
  );
}
