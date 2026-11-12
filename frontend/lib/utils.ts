import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) { return twMerge(clsx(inputs)); }

/** Compact number for dense tables: 4 significant digits, exponent for extremes. */
export function fmt(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a < -1e30 || v < -1e30) return "−∞";
  if (a >= 1e5 || a < 1e-3) return v.toExponential(Math.max(1, digits - 2)).replace("e+", "e").replace("-", "−");
  return v.toPrecision(digits).replace(/\.?0+$/, (m) => (m.startsWith(".") ? "" : m)).replace("-", "−");
}
export const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;
export function bytes(n: number): string {
  if (!n) return "0 B";
  const u = ["B", "KB", "MB", "GB", "TB"]; const i = Math.min(Math.floor(Math.log(n) / Math.log(1024)), u.length - 1);
  return `${(n / 1024 ** i).toFixed(i ? 1 : 0)} ${u[i]}`;
}
export const compact = (n: number) => Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(n);
export const shapeStr = (s: number[]) => `[${s.join(", ")}]`;
/** Make whitespace in a token visible. */
export function showToken(t: string): string {
  if (t === "") return "∅";
  return t.replace(/ /g, "·").replace(/\n/g, "↵").replace(/\t/g, "⇥");
}
export const ms = (v: number | null | undefined) => (v == null ? "—" : v < 1 ? `${(v * 1000).toFixed(0)} µs` : v < 1000 ? `${v.toFixed(v < 10 ? 2 : 1)} ms` : `${(v / 1000).toFixed(2)} s`);
