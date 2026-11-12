"use client";
import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/services/api";
import { useStore } from "@/hooks/use-store";
import { useModel, usePalette } from "@/hooks/use-lens";
import { ErrorNotice, Loading, NeedModel, PageHeader, VizNote } from "@/components/common";
import { ValueTable } from "@/components/inspector/inspector";
import { Plot } from "@/components/charts/plot";
import { VectorHeat } from "@/components/tensors/heatmap";
import { Badge, Card, CardBody, CardHeader, Notice, Stat, Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/primitives";
import { cn, fmt, showToken } from "@/lib/utils";

function useDebounced<T>(v: T, delay = 250): T {
  const [d, setD] = useState(v);
  useEffect(() => { const t = setTimeout(() => setD(v), delay); return () => clearTimeout(t); }, [v, delay]);
  return d;
}

export default function TokenizationPage() {
  const model = useModel();
  const p = usePalette();
  const prompt = useStore((s) => s.prompt);
  const [text, setText] = useState(prompt);
  const [sel, setSel] = useState(0);
  const debounced = useDebounced(text);
  const tk = useQuery({ queryKey: ["tokenize", model?.model_id, debounced], queryFn: () => api.tokenize(model!.model_id, debounced), enabled: !!model, placeholderData: (prev) => prev });
  const tokens = tk.data?.tokens ?? [];
  const cur = tokens[Math.min(sel, tokens.length - 1)];
  const emb = useQuery({ queryKey: ["embedding", model?.model_id, cur?.token_id], queryFn: () => api.embedding(model!.model_id, cur!.token_id), enabled: !!model && !!cur, staleTime: Infinity });
  const ids = tokens.map((t) => t.token_id);
  const pca = useQuery({ queryKey: ["emb-pca", model?.model_id, ids.join(",")], queryFn: () => api.embeddingsPca(model!.model_id, ids), enabled: !!model && ids.length >= 2, staleTime: Infinity });
  if (!model) return <><PageHeader title="Tokenization" /><NeedModel /></>;
  const v = tk.data?.vocabulary;

  return (
    <div className="space-y-3">
      <PageHeader title="Tokenization Explorer" sub="How the model's real tokenizer splits text into tokens, the integer id of each token, and the embedding row that id selects." />
      <Card>
        <CardHeader title="Text" sub="Edit freely; tokenization updates as you type. This does not change the prompt of the current run." />
        <CardBody className="space-y-3">
          <textarea value={text} onChange={(e) => { setText(e.target.value); }} rows={3} aria-label="Text to tokenize"
            className="num w-full resize-y rounded-md border border-line-strong bg-surface-2 p-2 text-xs text-ink focus-visible:outline-2 focus-visible:outline-accent" />
          <ErrorNotice error={tk.error} />
          {tk.data && (
            <>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Stat label="Tokens" value={tk.data.count} /><Stat label="Characters" value={tk.data.characters} sub={tk.data.count ? `${(tk.data.characters / tk.data.count).toFixed(2)} chars per token` : undefined} />
                <Stat label="Vocabulary size" value={v!.vocab_size.toLocaleString()} sub={v!.algorithm ? `${v!.algorithm} · ${v!.tokenizer_class}` : v!.tokenizer_class} />
                <Stat label="Decodes back exactly" value={tk.data.round_trip ? "yes" : "no"} />
              </div>
              <div>
                <div className="mb-1 text-[11px] text-muted">Token boundaries. Alternating fills mark where one token ends and the next begins; a dashed border marks a special token; “·” is a space that belongs to the token.</div>
                <div className="num flex flex-wrap gap-y-1 text-sm leading-7">
                  {tokens.map((t, i) => (
                    <button key={i} onClick={() => setSel(i)} title={`index ${t.index} · id ${t.token_id} · chars ${t.start}–${t.end}`}
                      className={cn("cursor-pointer border-y px-0.5", i % 2 ? "bg-surface-3" : "bg-accent-soft", t.special ? "border-dashed border-warn" : "border-transparent",
                        i === sel && "outline-2 outline-accent")}>{showToken(t.text)}</button>
                  ))}
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="text-[11px] text-muted"><tr>{["index", "decoded token", "vocabulary entry", "token id", "characters", "UTF-8 bytes", ""].map((h) => <th key={h} className="border-b border-line px-2 py-1 font-medium">{h}</th>)}</tr></thead>
                  <tbody>
                    {tokens.map((t, i) => (
                      <tr key={i} onClick={() => setSel(i)} className={cn("cursor-pointer border-b border-line/60 hover:bg-surface-2", i === sel && "bg-accent-soft")}>
                        <td className="num px-2 py-0.5 text-muted">{t.index}</td><td className="num px-2 text-ink">“{showToken(t.text)}”</td><td className="num px-2">{t.raw}</td>
                        <td className="num px-2 text-ink">{t.token_id}</td><td className="num px-2">{t.start === null ? "—" : `${t.start}–${t.end}`}</td><td className="num px-2">{t.bytes}</td>
                        <td className="px-2">{t.special && <Badge tone="warn">special</Badge>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="flex flex-wrap gap-1 text-[11px]">
                {Object.entries(v!.special_tokens).map(([k, s]) => <Badge key={k}>{k.replace("_token", "")}: <span className="num">{s}</span></Badge>)}
                {v!.model_max_length && <Badge>max length {v!.model_max_length}</Badge>}
                <Badge>base vocabulary {v!.base_vocab_size.toLocaleString()}</Badge>
              </div>
            </>
          )}
        </CardBody>
      </Card>

      {cur && (
        <Card>
          <CardHeader title={<>Embedding of “<span className="num">{showToken(cur.text)}</span>”</>} sub={<>Row <span className="num">{cur.token_id}</span> of the token-embedding matrix W_E {emb.data && <span className="num">[{emb.data.matrix_shape.join(" × ")}]</span>}. Read directly from the model's parameters.</>}
            right={<Badge tone="accent">model parameter</Badge>} />
          <CardBody>
            {emb.isLoading && <Loading />}
            <ErrorNotice error={emb.error} />
            {emb.data && (
              <Tabs defaultValue="heat">
                <TabsList><TabsTrigger value="heat">Heatmap</TabsTrigger><TabsTrigger value="num">Numerical vector</TabsTrigger><TabsTrigger value="pca">2-D projection (PCA)</TabsTrigger><TabsTrigger value="nn">Nearest tokens</TabsTrigger></TabsList>
                <TabsContent value="heat" className="mt-3 space-y-2">
                  <VectorHeat values={emb.data.values} wrapAt={emb.data.values.length > 512 ? 64 : 48} maxCell={14} />
                  <div className="num text-[11px] text-muted">{emb.data.values.length} dimensions · {emb.data.dtype} · min {fmt(emb.data.stats.min)} · max {fmt(emb.data.stats.max)} · mean {fmt(emb.data.stats.mean)} · L2 norm {fmt(emb.data.stats.l2)}</div>
                </TabsContent>
                <TabsContent value="num" className="mt-3"><ValueTable values={emb.data.values} page={96} /></TabsContent>
                <TabsContent value="pca" className="mt-3">
                  {tokens.length < 2 ? <Notice>PCA needs at least two tokens.</Notice> : pca.data ? (
                    <>
                      <Plot height={320} data={[{
                        type: "scatter", mode: "text+markers", x: pca.data.points.map((q) => q[0]), y: pca.data.points.map((q) => q[1]),
                        text: tokens.map((t) => showToken(t.text)), textposition: "top center", textfont: { color: p.ink2, size: 10, family: "ui-monospace, monospace" },
                        marker: { size: tokens.map((_, i) => (i === sel ? 13 : 8)), color: tokens.map((_, i) => (i === sel ? p.series[1] : p.series[0])), line: { color: p.surface, width: 2 } },
                        hovertemplate: "%{text}<br>PC1 %{x:.3f}<br>PC2 %{y:.3f}<extra></extra>",
                      }]} layout={{ xaxis: { title: { text: `PC1 (${(pca.data.explained_variance_ratio[0] * 100).toFixed(1)}% of variance)` } }, yaxis: { title: { text: `PC2 (${(pca.data.explained_variance_ratio[1] * 100).toFixed(1)}%)` } } }}
                        onClick={(e) => setSel(e.pointIndex)} />
                      <VizNote>{pca.data.note} The selected token is highlighted. Repeated tokens share one point.</VizNote>
                    </>
                  ) : <Loading />}
                </TabsContent>
                <TabsContent value="nn" className="mt-3">
                  <div className="grid gap-x-6 sm:grid-cols-2">
                    {emb.data.neighbors?.map((n) => (
                      <div key={n.token_id} className="grid grid-cols-[minmax(0,1fr)_auto_80px] items-center gap-2 border-b border-line/60 py-0.5 text-xs">
                        <span className="num text-ink">“{showToken(n.token)}” <span className="text-muted">#{n.token_id}</span></span><span className="num">{n.cosine.toFixed(3)}</span>
                        <span className="h-1.5 rounded-sm bg-surface-3"><span className="block h-full rounded-sm bg-accent" style={{ width: `${Math.max(0, n.cosine) * 100}%` }} /></span>
                      </div>
                    ))}
                  </div>
                  <VizNote>{emb.data.neighbors_note}</VizNote>
                </TabsContent>
              </Tabs>
            )}
          </CardBody>
        </Card>
      )}
    </div>
  );
}
