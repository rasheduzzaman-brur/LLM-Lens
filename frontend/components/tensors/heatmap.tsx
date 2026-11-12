"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { usePalette } from "@/hooks/use-lens";
import { makeScale, type ScaleKind } from "@/lib/colors";
import { cn, fmt, showToken } from "@/lib/utils";
import type { Matrix } from "@/types/api";

export interface HeatmapProps {
  data: Matrix; rowLabels?: string[]; colLabels?: string[]; kind?: ScaleKind; domain?: [number, number];
  maxCell?: number; minCell?: number; maxHeight?: number; rowHeight?: number; plainLabels?: boolean; legend?: boolean; className?: string;
  selectedRow?: number | null; selectedCol?: number | null; rowName?: string; colName?: string; valueName?: string;
  onSelect?: (row: number, col: number) => void;
  /** tooltip helpers: map a displayed cell to the user's own index (e.g. wrapped vectors) */
  describe?: (row: number, col: number, v: number | null) => string;
}

const isMasked = (v: number | null) => v === null || v < -1e30;

/** Canvas heatmap. Colors encode the values exactly as given; null / finfo.min cells are drawn as masked. */
export function Heatmap({ data, rowLabels, colLabels, kind, domain, maxCell = 26, minCell = 1, maxHeight = 520, rowHeight, plainLabels, legend = true, className,
  selectedRow, selectedCol, rowName = "row", colName = "col", valueName = "value", onSelect, describe }: HeatmapProps) {
  const p = usePalette();
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(0);
  const [hover, setHover] = useState<{ r: number; c: number; x: number; y: number } | null>(null);
  const rows = data.length, cols = rows ? data[0].length : 0;
  const show = plainLabels ? (t: string) => t : showToken; // token labels make whitespace visible; plain labels are shown as written

  useEffect(() => {
    if (!wrap.current) return;
    const ro = new ResizeObserver((e) => setWidth(Math.floor(e[0].contentRect.width)));
    ro.observe(wrap.current);
    return () => ro.disconnect();
  }, []);

  const range = useMemo(() => {
    let lo = Infinity, hi = -Infinity;
    for (const r of data) for (const v of r) if (!isMasked(v)) { if (v! < lo) lo = v!; if (v! > hi) hi = v!; }
    return lo === Infinity ? { lo: 0, hi: 1 } : { lo, hi };
  }, [data]);
  const scale = useMemo(() => makeScale(p, range.lo, range.hi, kind, domain), [p, range, kind, domain]);

  const wantLabels = !!(rowLabels || colLabels);
  const left = rowLabels ? Math.min(110, Math.max(...rowLabels.map((l) => show(l).length), 2) * 6.6 + 10) : 0;
  const availW = Math.max(0, width - (rowLabels && (rowHeight ?? 11) >= 11 ? left : 0) - 2);
  let cell = cols ? Math.floor(availW / cols) : 0;
  cell = Math.max(minCell, Math.min(maxCell, cell));
  const fitH = rows ? Math.floor(maxHeight / rows) || 1 : 0;
  // square cells for ordinary matrices; wide ones (hundreds of columns) keep readable row heights
  const cellH = rowHeight ?? (cell >= 6 ? Math.max(minCell, Math.min(cell, fitH)) : Math.max(minCell, Math.min(18, fitH)));
  const rowLabelsOn = !!rowLabels && cellH >= 11;
  const colLabelsOn = !!colLabels && cell >= 11;
  const labelsOn = wantLabels && (rowLabelsOn || colLabelsOn);
  const L = rowLabelsOn ? left : 0;
  const top = colLabelsOn ? Math.min(74, Math.max(...colLabels.map((l) => show(l).length), 2) * 5.2 + 12) : 0;
  const W = L + cols * cell, H = top + rows * cellH;

  useEffect(() => {
    const cv = canvas.current;
    if (!cv || !width || !rows) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = W * dpr; cv.height = H * dpr;
    const ctx = cv.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const gap = cell >= 6 && cellH >= 6 ? 1 : 0;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const v = data[r][c];
      ctx.fillStyle = isMasked(v) ? p.masked : scale.color(v!);
      ctx.fillRect(L + c * cell, top + r * cellH, cell - gap, cellH - gap);
      if (isMasked(v) && cell >= 9 && cellH >= 9) {
        ctx.strokeStyle = p.grid; ctx.lineWidth = 1; ctx.beginPath();
        ctx.moveTo(L + c * cell + 2, top + (r + 1) * cellH - 3); ctx.lineTo(L + (c + 1) * cell - 3, top + r * cellH + 2); ctx.stroke();
      }
    }
    if (labelsOn) {
      ctx.fillStyle = p.ink2; ctx.font = "10.5px ui-monospace, monospace"; ctx.textBaseline = "middle";
      if (rowLabelsOn && rowLabels) { ctx.textAlign = "right"; for (let r = 0; r < rows; r++) {
        ctx.fillStyle = r === selectedRow ? p.ink : p.ink2;
        ctx.fillText(show(rowLabels[r] ?? "").slice(0, 17), L - 6, top + r * cellH + cellH / 2);
      } }
      if (colLabelsOn && colLabels) { ctx.textAlign = "left"; for (let c = 0; c < cols; c++) {
        ctx.save(); ctx.translate(L + c * cell + cell / 2, top - 5); ctx.rotate(-Math.PI / 3);
        ctx.fillStyle = c === selectedCol ? p.ink : p.ink2;
        ctx.fillText(show(colLabels[c] ?? "").slice(0, 12), 0, 0); ctx.restore();
      } }
    }
    ctx.strokeStyle = p.ink; ctx.lineWidth = 1.5;
    if (selectedRow != null && selectedRow < rows) ctx.strokeRect(L + 0.75, top + selectedRow * cellH + 0.75, cols * cell - 1.5, cellH - 1.5);
    if (selectedCol != null && selectedCol < cols) ctx.strokeRect(L + selectedCol * cell + 0.75, top + 0.75, cell - 1.5, rows * cellH - 1.5);
  }, [data, width, W, H, L, top, cell, cellH, rows, cols, scale, p, labelsOn, rowLabelsOn, colLabelsOn, rowLabels, colLabels, selectedRow, selectedCol, plainLabels]); // eslint-disable-line react-hooks/exhaustive-deps

  const locate = (e: React.MouseEvent) => {
    const b = canvas.current!.getBoundingClientRect();
    const c = Math.floor((e.clientX - b.left - L) / cell), r = Math.floor((e.clientY - b.top - top) / cellH);
    return r >= 0 && c >= 0 && r < rows && c < cols ? { r, c, x: e.clientX - b.left, y: e.clientY - b.top } : null;
  };
  if (!rows || !cols) return <div className="text-xs text-muted">empty tensor</div>;
  const hv = hover ? data[hover.r][hover.c] : null;

  return (
    <div ref={wrap} className={cn("relative w-full", className)}>
      <div className="relative overflow-auto" style={{ maxHeight: maxHeight + top + 4 }}>
        <canvas ref={canvas} style={{ width: W, height: H, cursor: onSelect ? "pointer" : "crosshair" }}
          onMouseMove={(e) => setHover(locate(e))} onMouseLeave={() => setHover(null)}
          onClick={(e) => { const h = locate(e); if (h && onSelect) onSelect(h.r, h.c); }} />
        {hover && (
          <div className="pointer-events-none absolute z-20 rounded-md border border-line-strong bg-surface-2 px-2 py-1 text-[11px] leading-4 shadow-lg whitespace-nowrap"
            style={{ left: Math.min(hover.x + 12, Math.max(0, W - 150)), top: hover.y + 14 }}>
            {describe ? <span className="text-ink-2">{describe(hover.r, hover.c, hv)}</span> : (
              <>
                <div className="text-ink-2">{rowName} {hover.r}{rowLabels?.[hover.r] !== undefined && <> <span className="num text-ink">“{show(rowLabels[hover.r])}”</span></>}</div>
                <div className="text-ink-2">{colName} {hover.c}{colLabels?.[hover.c] !== undefined && <> <span className="num text-ink">“{show(colLabels[hover.c])}”</span></>}</div>
              </>
            )}
            <div className="num font-semibold text-ink">{isMasked(hv) ? "masked" : `${valueName} ${fmt(hv, 5)}`}</div>
          </div>
        )}
      </div>
      {legend && (
        <div className="mt-1.5 flex items-center gap-2 text-[11px] text-muted" style={{ paddingLeft: L }}>
          <span className="num">{fmt(scale.lo, 3)}</span>
          <span className="h-2 w-28 rounded-sm" style={{ background: `linear-gradient(to right, ${Array.from({ length: 9 }, (_, i) => scale.color(scale.lo + ((scale.hi - scale.lo) * i) / 8)).join(",")})` }} />
          <span className="num">{fmt(scale.hi, 3)}</span>
          <span>{scale.kind === "diverging" ? "diverging, centred on 0" : "sequential"}</span>
          <span className="num ml-auto">{rows} × {cols}</span>
        </div>
      )}
    </div>
  );
}

/** A 1-D vector wrapped into rows so every element stays visible. */
export function VectorHeat({ values, wrapAt = 64, kind, domain, maxCell = 12, name = "dim", legend = true }:
  { values: number[]; wrapAt?: number; kind?: ScaleKind; domain?: [number, number]; maxCell?: number; name?: string; legend?: boolean }) {
  const data = useMemo(() => {
    const out: Matrix = [];
    for (let i = 0; i < values.length; i += wrapAt) {
      const row: (number | null)[] = values.slice(i, i + wrapAt);
      while (row.length < wrapAt && values.length > wrapAt) row.push(null);
      out.push(row);
    }
    return out;
  }, [values, wrapAt]);
  return <Heatmap data={data} kind={kind} domain={domain} maxCell={maxCell} legend={legend} maxHeight={260}
    describe={(r, c) => `${name} ${r * wrapAt + c}`} />;
}
