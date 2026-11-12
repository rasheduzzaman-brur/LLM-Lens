"use client";
import { useMemo } from "react";
import { Background, Controls, Handle, Position, ReactFlow, type Edge, type Node, type NodeProps } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useStore } from "@/hooks/use-store";
import { usePalette } from "@/hooks/use-lens";
import { cn, shapeStr } from "@/lib/utils";
import type { Architecture, TraceEvent } from "@/types/api";

interface NodeData extends Record<string, unknown> {
  label: string; sub?: string; shape: string; order?: number; kind: string; active: boolean; executing: boolean; indent: boolean; stage?: string;
}

const STAGE_BAR: Record<string, string> = { embedding: "#199e70", norm: "#c98500", final_norm: "#c98500", attention: "#3987e5", residual: "#898781", mlp: "#d95926", lm_head: "#9085e9", sampling: "#d55181" };

function LensNode({ data }: NodeProps<Node<NodeData>>) {
  return (
    <div className={cn("relative rounded-md border bg-surface px-2.5 py-1.5 text-left shadow-sm transition-colors", data.indent ? "w-[236px]" : "w-[264px]",
      data.executing ? "border-warn ring-2 ring-warn/40" : data.active ? "border-accent ring-1 ring-accent/50" : "border-line-strong hover:border-accent")}>
      <Handle type="target" position={Position.Top} /><Handle type="source" position={Position.Bottom} />
      <Handle type="source" position={Position.Right} id="r-out" /><Handle type="target" position={Position.Right} id="r-in" />
      {data.stage && <span className="absolute inset-y-1 left-0 w-[3px] rounded-r" style={{ background: STAGE_BAR[data.stage] ?? "#898781" }} />}
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-xs font-semibold text-ink">{data.label}</span>
        {data.order !== undefined && <span className="num text-[10px] text-muted">#{data.order}</span>}
      </div>
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-[10.5px] text-muted">{data.sub}</span>
        <span className="num shrink-0 text-[10.5px] text-ink-2">{data.shape}</span>
      </div>
    </div>
  );
}
const nodeTypes = { lens: LensNode };

export interface FlowProps {
  arch: Architecture; layer: number; find?: (layer: number | null, name: string) => TraceEvent | undefined; current?: TraceEvent; hasRun: boolean;
  onLayer: (i: number) => void;
}

/** Architecture diagram built from the loaded model. Shapes come from the trace when a run exists. */
export function ArchitectureFlow({ arch, layer, find, current, hasRun, onLayer }: FlowProps) {
  const p = usePalette();
  const select = useStore((s) => s.select);
  const sel = useStore((s) => s.selection);
  const stepSel = useStore((s) => s.step);

  const { nodes, edges } = useMemo(() => {
    const nodes: Node<NodeData>[] = [];
    const edges: Edge[] = [];
    let y = 0, prev: string | null = null, order = 0;
    const real = (l: number | null, name?: string) => { const e = name && find ? find(l, name) : undefined; return e ? shapeStr(e.shape) : undefined; };
    const selName = sel?.kind === "tensor" ? sel.name : current?.tensor_name;
    const selLayer = sel?.kind === "tensor" ? sel.layer : current?.layer ?? null;
    const push = (id: string, data: NodeData, x: number, h: number) => {
      nodes.push({ id, type: "lens", position: { x, y }, data, draggable: false });
      if (prev) edges.push({ id: `${prev}>${id}`, source: prev, target: id, style: { stroke: p.axis, strokeWidth: 1.5 } });
      prev = id; y += h;
    };
    for (const n of arch.nodes) {
      if (n.kind !== "layer") {
        const hit = n.tensors.includes(selName ?? "") && selLayer === null;
        push(n.id, { label: n.label, sub: n.module ?? n.formula, shape: real(null, n.tensors[n.tensors.length - 1]) ?? n.shape, order: order++, kind: n.kind,
          active: hit && sel !== null, executing: !!current && current.layer === null && n.tensors.includes(current.tensor_name), indent: false, stage: n.stage }, 0, 58);
        continue;
      }
      const open = n.layer === layer;
      push(n.id, { label: n.label, sub: open ? n.module ?? "" : "click to expand", shape: real(n.layer!, "resid_post") ?? n.shape, order: order++, kind: "layer",
        active: open, executing: !open && !!current && current.layer === n.layer, indent: false }, 0, open ? 58 : 50);
      if (!open) continue;
      const ids: Record<string, string> = {};
      for (const op of arch.layer_template) {
        const id = `${n.id}.${op.key}`; ids[op.key] = id;
        const names = [...op.out, ...op.detail];
        push(id, { label: op.label, sub: op.module ?? op.formula, shape: real(n.layer!, op.out[op.out.length - 1]) ?? op.shape, order: order++, kind: "op",
          active: selLayer === n.layer && names.includes(selName ?? ""), executing: !!current && current.layer === n.layer && names.includes(current.tensor_name), indent: true, stage: op.stage }, 28, 54);
      }
      y += 6;
      const skip = { stroke: p.muted, strokeWidth: 1.5, strokeDasharray: "5 4" };
      const lbl = { labelStyle: { fill: p.ink2, fontSize: 10 }, labelBgStyle: { fill: p.surface }, labelBgPadding: [4, 2] as [number, number] };
      if (ids.res1) edges.push({ id: `${n.id}.skip1`, source: n.id, sourceHandle: "r-out", target: ids.res1, targetHandle: "r-in", type: "smoothstep", style: skip, label: "residual", ...lbl });
      if (ids.res1 && ids.res2) edges.push({ id: `${n.id}.skip2`, source: ids.res1, sourceHandle: "r-out", target: ids.res2, targetHandle: "r-in", type: "smoothstep", style: skip, label: "residual", ...lbl });
    }
    return { nodes, edges };
  }, [arch, layer, find, current, sel, p]);
  // frame the expanded layer (the whole stack is too tall to read at once); pan or scroll for the rest
  const focus = useMemo(() => [{ id: `layer${layer}` }, ...arch.layer_template.map((o) => ({ id: `layer${layer}.${o.key}` }))], [arch, layer]);

  const onNodeClick = (_: unknown, node: Node<NodeData>) => {
    const an = arch.nodes.find((n) => n.id === node.id);
    if (an?.kind === "layer") { onLayer(an.layer!); return; }
    if (!hasRun) return;
    if (an) { const name = an.tensors[an.tensors.length - 1]; if (name) select({ kind: "tensor", name, layer: null, step: stepSel }); return; }
    const [lid, key] = node.id.split(".");
    const op = arch.layer_template.find((o) => o.key === key);
    const ln = arch.nodes.find((n) => n.id === lid);
    if (op && ln) select({ kind: "tensor", name: op.out[op.out.length - 1], layer: ln.layer!, step: stepSel });
  };

  return (
    <div className="h-[680px] w-full overflow-hidden rounded-md border border-line bg-surface-2">
      <ReactFlow key={layer} nodes={nodes} edges={edges} nodeTypes={nodeTypes} onNodeClick={onNodeClick} fitView fitViewOptions={{ nodes: focus, padding: 0.12, maxZoom: 1, minZoom: 0.72 }}
        minZoom={0.15} maxZoom={1.5} panOnScroll nodesConnectable={false} nodesDraggable={false} elementsSelectable={false} colorMode={p.dark ? "dark" : "light"} proOptions={{ hideAttribution: true }}>
        <Background color={p.grid} gap={18} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
