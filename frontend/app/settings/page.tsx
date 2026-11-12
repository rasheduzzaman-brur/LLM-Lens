"use client";
import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, API_URL } from "@/services/api";
import { useStore } from "@/hooks/use-store";
import { ErrorNotice, Loading, PageHeader } from "@/components/common";
import { Button, Card, CardBody, CardHeader, Input, Notice, Select } from "@/components/ui/primitives";
import { bytes } from "@/lib/utils";
import type { AppSettings } from "@/types/api";

type NumKey = Exclude<keyof AppSettings, "models_dir">;
const FIELDS: { key: NumKey; label: string; help: string; scale?: number; unit?: string }[] = [
  { key: "retain_steps", label: "Decode steps that keep tensors", help: "Older decode steps of a run become metadata-only. The prefill step is always kept." },
  { key: "max_store_bytes", label: "Tensor memory budget", help: "When exceeded, tensors of older runs and then older steps are released.", scale: 1_000_000, unit: "MB" },
  { key: "max_runs", label: "Runs kept", help: "The oldest run is discarded when a new one would exceed this." },
  { key: "max_prompt_tokens", label: "Maximum prompt tokens", help: "Attention tensors grow with the square of the sequence length." },
  { key: "max_new_tokens", label: "Maximum new tokens per run", help: "Upper bound for the “max new” field in the top bar." },
  { key: "max_cells", label: "Maximum values per response", help: "Larger 2-D slices are mean-pooled (flagged as a visualization transform); other slices must be narrowed." },
];

export default function SettingsPage() {
  const qc = useQueryClient();
  const theme = useStore((s) => s.theme);
  const set = useStore((s) => s.set);
  const q = useQuery({ queryKey: ["settings"], queryFn: api.settings });
  const [draft, setDraft] = useState<Partial<Record<NumKey, number>>>({});
  useEffect(() => { if (q.data) setDraft(Object.fromEntries(FIELDS.map((f) => [f.key, q.data[f.key] / (f.scale ?? 1)]))); }, [q.data]);
  const save = useMutation({
    mutationFn: () => api.patchSettings(Object.fromEntries(FIELDS.map((f) => [f.key, Math.round((draft[f.key] ?? 0) * (f.scale ?? 1))]))),
    onSuccess: (d) => { qc.setQueryData(["settings"], d); qc.invalidateQueries({ queryKey: ["system"] }); },
  });
  return (
    <div className="max-w-3xl space-y-3">
      <PageHeader title="Settings" />
      <Card>
        <CardHeader title="Appearance" />
        <CardBody className="flex items-center gap-2 text-xs text-ink-2">Theme
          <Select value={theme} onChange={(e) => set({ theme: e.target.value as "dark" | "light" })}><option value="dark">dark</option><option value="light">light</option></Select>
        </CardBody>
      </Card>
      <Card>
        <CardHeader title="Memory and trace retention" sub="Applied by the backend immediately; values reset when the backend restarts." />
        <CardBody className="space-y-3">
          {q.isLoading && <Loading />}
          <ErrorNotice error={q.error ?? save.error} />
          {q.data && FIELDS.map((f) => (
            <div key={f.key} className="grid items-center gap-x-3 gap-y-0.5 sm:grid-cols-[1fr_150px]">
              <div><div className="text-xs font-medium">{f.label}</div><div className="text-[11px] text-muted">{f.help}</div></div>
              <div className="flex items-center gap-1.5"><Input type="number" className="num" value={draft[f.key] ?? ""} onChange={(e) => setDraft({ ...draft, [f.key]: Number(e.target.value) })} aria-label={f.label} />{f.unit && <span className="text-xs text-muted">{f.unit}</span>}</div>
            </div>
          ))}
          {q.data && <div className="flex items-center gap-2"><Button variant="default" onClick={() => save.mutate()} disabled={save.isPending}>Save</Button>{save.isSuccess && <span className="text-xs text-good">Saved. Budget is now {bytes(save.data.max_store_bytes)}.</span>}</div>}
        </CardBody>
      </Card>
      <Card>
        <CardHeader title="Backend" />
        <CardBody className="space-y-2 text-xs text-ink-2">
          <div>API <span className="num text-ink">{API_URL}</span> · interactive documentation at <a className="num text-accent hover:underline" href={`${API_URL}/docs`} target="_blank" rel="noreferrer">{API_URL}/docs</a></div>
          {q.data && <div>Models folder <span className="num text-ink">{q.data.models_dir}</span></div>}
          <Notice>Tensor capture mode is chosen per run on the Computation Traces page. Sampling settings are on the Generation page.</Notice>
        </CardBody>
      </Card>
    </div>
  );
}
