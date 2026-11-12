"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Activity, Boxes, Braces, Cpu, Database, Grid3x3, LayoutDashboard, Layers, ListTree, Loader2, Menu, Moon, PanelRight, Play, Settings, Sparkles, Sun, Type } from "lucide-react";
import { useStore } from "@/hooks/use-store";
import { useModel, usePosition, useRun, useRunStream, useStartRun, useStepIndex } from "@/hooks/use-lens";
import { ErrorNotice, StepSelect, TokenStrip } from "@/components/common";
import { Inspector } from "@/components/inspector/inspector";
import { BottomPanel } from "@/components/layout/bottom-panel";
import { Badge, Button, Input } from "@/components/ui/primitives";
import { cn, compact } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/models", label: "Model Explorer", icon: Boxes },
  { href: "/tokenization", label: "Tokenization", icon: Type },
  { href: "/layers", label: "Transformer Layers", icon: Layers },
  { href: "/attention", label: "Attention Explorer", icon: Grid3x3 },
  { href: "/hidden-states", label: "Hidden States", icon: Activity },
  { href: "/kv-cache", label: "KV Cache", icon: Database },
  { href: "/generation", label: "Generation", icon: Sparkles },
  { href: "/traces", label: "Computation Traces", icon: ListTree },
  { href: "/settings", label: "Settings", icon: Settings },
];

function RunBar() {
  const prompt = useStore((s) => s.prompt);
  const maxNew = useStore((s) => s.maxNewTokens);
  const set = useStore((s) => s.set);
  const model = useModel();
  const start = useStartRun();
  return (
    <div className="min-w-0 flex-1">
      <form className="flex items-center gap-2" onSubmit={(e) => { e.preventDefault(); if (prompt.length) start.mutate(); }}>
        <Input value={prompt} onChange={(e) => set({ prompt: e.target.value })} placeholder="Enter a prompt…" aria-label="Prompt" className="num min-w-0 flex-1" />
        <label className="hidden items-center gap-1 text-xs text-muted lg:flex" title="Maximum number of tokens to generate">max new
          <Input type="number" min={1} max={256} value={maxNew} onChange={(e) => set({ maxNewTokens: Math.max(1, Number(e.target.value) || 1) })} className="num w-14" aria-label="Max new tokens" />
        </label>
        <Button type="submit" variant="default" disabled={!model || start.isPending || !prompt.length} title={model ? "Tokenize and run the traced prefill pass" : "Load a model first"}>
          {start.isPending ? <Loader2 className="animate-spin" /> : <Play />}Run
        </Button>
      </form>
      {start.error ? <div className="mt-1"><ErrorNotice error={start.error} /></div> : null}
    </div>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  const theme = useStore((s) => s.theme);
  const inspectorOpen = useStore((s) => s.inspectorOpen);
  const set = useStore((s) => s.set);
  const model = useModel();
  const run = useRun();
  const stepIdx = useStepIndex(run);
  const position = usePosition(run, stepIdx);
  const [nav, setNav] = useState(false);
  useRunStream();
  useEffect(() => { document.documentElement.classList.toggle("dark", theme === "dark"); }, [theme]);
  useEffect(() => setNav(false), [path]);

  return (
    <div className="flex h-dvh overflow-hidden bg-page text-ink">
      <nav className={cn("z-40 flex w-52 shrink-0 flex-col border-r border-line bg-surface max-md:fixed max-md:inset-y-0 max-md:left-0 max-md:shadow-xl", !nav && "max-md:hidden")}>
        <div className="flex h-11 items-center gap-2 border-b border-line px-3">
          <Braces className="size-4 text-accent" /><span className="text-sm font-semibold tracking-tight">LLM Lens</span>
        </div>
        <ul className="flex-1 space-y-px overflow-y-auto p-1.5">
          {NAV.map((n) => {
            const active = n.href === "/" ? path === "/" : path.startsWith(n.href);
            return (
              <li key={n.href}>
                <Link href={n.href} className={cn("flex items-center gap-2 rounded-md px-2 py-1.5 text-xs font-medium", active ? "bg-accent-soft text-accent" : "text-ink-2 hover:bg-surface-2 hover:text-ink")}>
                  <n.icon className="size-3.5" />{n.label}
                </Link>
              </li>
            );
          })}
        </ul>
        <div className="space-y-1 border-t border-line p-2.5 text-[11px] text-muted">
          {model ? (
            <>
              <div className="flex items-center gap-1.5 text-ink"><Cpu className="size-3.5 text-accent" /><span className="truncate font-medium">{model.name}</span></div>
              <div className="num">{model.n_layers} layers · {model.n_heads} heads · d={model.hidden_size}</div>
              <div className="num">{compact(model.parameter_count)} params · {model.dtype} · {model.device}</div>
            </>
          ) : <div>No model loaded</div>}
        </div>
      </nav>
      {nav && <div className="fixed inset-0 z-30 bg-black/40 md:hidden" onClick={() => setNav(false)} />}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex min-h-11 items-start gap-2 border-b border-line bg-surface px-3 py-1.5">
          <Button size="icon" variant="ghost" className="md:hidden" onClick={() => setNav(true)} aria-label="Open navigation"><Menu /></Button>
          <RunBar />
          <Button size="icon" variant="ghost" onClick={() => set({ theme: theme === "dark" ? "light" : "dark" })} aria-label="Toggle theme">{theme === "dark" ? <Sun /> : <Moon />}</Button>
          <Button size="icon" variant={inspectorOpen ? "outline" : "ghost"} onClick={() => set({ inspectorOpen: !inspectorOpen })} aria-label="Toggle inspector"><PanelRight /></Button>
        </header>
        {run && (
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-b border-line bg-surface px-3 py-1.5">
            <StepSelect run={run} stepIdx={stepIdx} />
            <TokenStrip run={run} stepIdx={stepIdx} position={position} />
            <div className="ml-auto flex items-center gap-1.5">
              <Badge tone={run.state === "running" ? "accent" : "neutral"}>{run.state}{run.stop_reason ? ` · ${run.stop_reason}` : ""}</Badge>
              <Badge>capture: {run.capture.mode}</Badge>
            </div>
          </div>
        )}
        <div className="flex min-h-0 flex-1">
          <div className="flex min-w-0 flex-1 flex-col">
            <main className="min-h-0 flex-1 overflow-y-auto p-3 md:p-4">{children}</main>
            <BottomPanel />
          </div>
          {inspectorOpen && <div className="w-[340px] shrink-0 max-lg:fixed max-lg:inset-y-0 max-lg:right-0 max-lg:z-40 max-lg:w-[min(340px,92vw)] max-lg:shadow-2xl"><Inspector /></div>}
        </div>
      </div>
    </div>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const [qc] = useState(() => new QueryClient({ defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 } } }));
  const [ready, setReady] = useState(false);
  useEffect(() => setReady(true), []); // persisted client state is only known after mount
  return <QueryClientProvider client={qc}>{ready ? <Shell>{children}</Shell> : <div className="h-dvh bg-page" />}</QueryClientProvider>;
}
