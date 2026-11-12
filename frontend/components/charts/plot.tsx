"use client";
import dynamic from "next/dynamic";
import type { Data, Layout } from "plotly.js";
import { usePalette } from "@/hooks/use-lens";

const Plotly = dynamic(
  async () => {
    const [{ default: factory }, plotly] = await Promise.all([import("react-plotly.js/factory"), import("plotly.js-dist-min")]);
    return factory(plotly.default ?? plotly);
  },
  { ssr: false, loading: () => <div className="flex h-full min-h-24 items-center justify-center text-xs text-muted">loading chart…</div> },
);

/** Plotly with the app theme: recessive grid, text in ink tokens, one y-axis. */
export function Plot({ data, layout, height = 260, onClick }: { data: Data[]; layout?: Partial<Layout>; height?: number;
  onClick?: (p: { pointIndex: number; curveNumber: number; customdata?: unknown }) => void }) {
  const p = usePalette();
  const axis = { gridcolor: p.grid, zerolinecolor: p.axis, linecolor: p.axis, tickfont: { color: p.muted, size: 10 }, title: { font: { color: p.ink2, size: 11 } }, automargin: true };
  const merged: Partial<Layout> = {
    height, autosize: true, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", margin: { l: 44, r: 12, t: 10, b: 36 },
    font: { family: "var(--font-geist-sans), system-ui, sans-serif", color: p.ink2, size: 11 }, colorway: p.series,
    showlegend: data.length > 1, legend: { orientation: "h", y: 1.12, x: 0, font: { color: p.ink2, size: 11 }, bgcolor: "rgba(0,0,0,0)" },
    hoverlabel: { bgcolor: p.surface2, bordercolor: p.axis, font: { color: p.ink, size: 11, family: "ui-monospace, monospace" } },
    bargap: 0.25, ...layout,
    xaxis: { ...axis, ...layout?.xaxis, title: { ...axis.title, ...(typeof layout?.xaxis?.title === "object" ? layout.xaxis.title : {}) } },
    yaxis: { ...axis, ...layout?.yaxis, title: { ...axis.title, ...(typeof layout?.yaxis?.title === "object" ? layout.yaxis.title : {}) } },
  };
  return (
    <Plotly data={data} layout={merged} config={{ displayModeBar: false, responsive: true }} useResizeHandler style={{ width: "100%", height }}
      onClick={onClick ? (e) => { const pt = e.points[0]; if (pt) onClick({ pointIndex: pt.pointIndex, curveNumber: pt.curveNumber, customdata: pt.customdata }); } : undefined} />
  );
}
