import { interpolateRgbBasis } from "d3";

export interface Palette {
  dark: boolean; surface: string; surface2: string; page: string; ink: string; ink2: string; muted: string; grid: string; axis: string;
  series: string[]; good: string; warning: string; critical: string; masked: string; accent: string;
  sequential: (t: number) => string; diverging: (t: number) => string;
}

// Sequential: one hue (blue), anchored on the surface so "no value" recedes.
// Diverging: blue <-> orange around a neutral gray midpoint.
const darkSeq = interpolateRgbBasis(["#1a1a19", "#0d366b", "#184f95", "#256abf", "#3987e5", "#86b6ef", "#cde2fb"]);
const lightSeq = interpolateRgbBasis(["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]);
const darkDiv = interpolateRgbBasis(["#9ec5f4", "#3987e5", "#1c5cab", "#383835", "#a8431b", "#d95926", "#f3a583"]);
const lightDiv = interpolateRgbBasis(["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a583", "#eb6834", "#8f3611"]);

export const DARK: Palette = {
  dark: true, surface: "#1a1a19", surface2: "#222221", page: "#0d0d0d", ink: "#f0efec", ink2: "#c3c2b7", muted: "#898781",
  grid: "#2c2c2a", axis: "#383835", series: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
  good: "#0ca30c", warning: "#fab219", critical: "#d03b3b", masked: "#141413", accent: "#3987e5", sequential: darkSeq, diverging: darkDiv,
};
export const LIGHT: Palette = {
  dark: false, surface: "#fcfcfb", surface2: "#f4f4f1", page: "#f9f9f7", ink: "#0b0b0b", ink2: "#52514e", muted: "#898781",
  grid: "#e1e0d9", axis: "#c3c2b7", series: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
  good: "#0ca30c", warning: "#fab219", critical: "#d03b3b", masked: "#ecebe6", accent: "#2a78d6", sequential: lightSeq, diverging: lightDiv,
};

export type ScaleKind = "sequential" | "diverging";
export interface ColorScale { kind: ScaleKind; lo: number; hi: number; color: (v: number) => string }

/** Signed data gets a diverging scale symmetric about 0; non-negative data a sequential one. */
export function makeScale(p: Palette, lo: number, hi: number, kind?: ScaleKind, domain?: [number, number]): ColorScale {
  const k = kind ?? (lo < 0 ? "diverging" : "sequential");
  if (k === "diverging") {
    const m = domain ? Math.max(Math.abs(domain[0]), Math.abs(domain[1])) : Math.max(Math.abs(lo), Math.abs(hi)) || 1;
    return { kind: k, lo: -m, hi: m, color: (v) => p.diverging(Math.max(0, Math.min(1, (v + m) / (2 * m)))) };
  }
  const a = domain ? domain[0] : Math.min(0, lo); const b = domain ? domain[1] : hi;
  const span = b - a || 1;
  return { kind: k, lo: a, hi: b, color: (v) => p.sequential(Math.max(0, Math.min(1, (v - a) / span))) };
}
