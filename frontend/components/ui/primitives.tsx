"use client";
import * as React from "react";
import * as TabsPrimitive from "@radix-ui/react-tabs";
import * as SliderPrimitive from "@radix-ui/react-slider";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-md text-xs font-medium transition-colors focus-visible:outline-2 focus-visible:outline-accent disabled:pointer-events-none disabled:opacity-45 [&_svg]:size-3.5 [&_svg]:shrink-0 cursor-pointer",
  {
    variants: {
      variant: {
        default: "bg-accent text-accent-ink hover:brightness-110",
        outline: "border border-line-strong bg-surface text-ink hover:bg-surface-2",
        ghost: "text-ink-2 hover:bg-surface-2 hover:text-ink",
        danger: "border border-line-strong text-bad hover:bg-surface-2",
      },
      size: { default: "h-8 px-3", sm: "h-7 px-2", icon: "size-7" },
    },
    defaultVariants: { variant: "outline", size: "default" },
  },
);
export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> { asChild?: boolean }
export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(({ className, variant, size, asChild, ...props }, ref) => {
  const Comp = asChild ? Slot : "button";
  return <Comp ref={ref} className={cn(buttonVariants({ variant, size }), className)} {...props} />;
});
Button.displayName = "Button";

export function Card({ className, ...p }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("rounded-lg border border-line bg-surface", className)} {...p} />;
}
export function CardHeader({ title, sub, right, className }: { title: React.ReactNode; sub?: React.ReactNode; right?: React.ReactNode; className?: string }) {
  return (
    <div className={cn("flex flex-wrap items-start justify-between gap-2 border-b border-line px-3 py-2", className)}>
      <div className="min-w-0">
        <h3 className="text-[13px] font-semibold text-ink">{title}</h3>
        {sub && <p className="mt-0.5 text-xs text-ink-2">{sub}</p>}
      </div>
      {right && <div className="flex flex-wrap items-center gap-1.5">{right}</div>}
    </div>
  );
}
export function CardBody({ className, ...p }: React.HTMLAttributes<HTMLDivElement>) { return <div className={cn("p-3", className)} {...p} />; }

const badgeVariants = cva("inline-flex items-center gap-1 rounded border px-1.5 py-px text-[11px] font-medium leading-4 whitespace-nowrap", {
  variants: {
    tone: {
      neutral: "border-line-strong text-ink-2", accent: "border-accent/50 text-accent bg-accent-soft", good: "border-good/50 text-good",
      warn: "border-warn/50 text-warn", bad: "border-bad/50 text-bad", derived: "border-derived/50 text-derived",
    },
  },
  defaultVariants: { tone: "neutral" },
});
export function Badge({ className, tone, ...p }: React.HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeVariants>) {
  return <span className={cn(badgeVariants({ tone }), className)} {...p} />;
}

export const Tabs = TabsPrimitive.Root;
export function TabsList({ className, ...p }: React.ComponentProps<typeof TabsPrimitive.List>) {
  return <TabsPrimitive.List className={cn("inline-flex flex-wrap items-center gap-0.5 rounded-md border border-line bg-surface-2 p-0.5", className)} {...p} />;
}
export function TabsTrigger({ className, ...p }: React.ComponentProps<typeof TabsPrimitive.Trigger>) {
  return <TabsPrimitive.Trigger className={cn("rounded px-2.5 py-1 text-xs font-medium text-ink-2 transition-colors hover:text-ink data-[state=active]:bg-surface data-[state=active]:text-ink data-[state=active]:shadow-sm cursor-pointer", className)} {...p} />;
}
export const TabsContent = TabsPrimitive.Content;

export function Slider({ className, ...p }: React.ComponentProps<typeof SliderPrimitive.Root>) {
  return (
    <SliderPrimitive.Root className={cn("relative flex h-5 w-full touch-none items-center select-none", className)} {...p}>
      <SliderPrimitive.Track className="relative h-1 grow rounded-full bg-surface-3">
        <SliderPrimitive.Range className="absolute h-full rounded-full bg-accent" />
      </SliderPrimitive.Track>
      <SliderPrimitive.Thumb className="block size-3.5 rounded-full border-2 border-accent bg-surface shadow focus-visible:outline-2 focus-visible:outline-accent cursor-grab" />
    </SliderPrimitive.Root>
  );
}

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(({ className, ...p }, ref) => (
  <input ref={ref} className={cn("h-8 w-full rounded-md border border-line-strong bg-surface px-2 text-xs text-ink placeholder:text-muted focus-visible:outline-2 focus-visible:outline-accent", className)} {...p} />
));
Input.displayName = "Input";

export function Select({ className, ...p }: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return <select className={cn("h-7 rounded-md border border-line-strong bg-surface px-1.5 text-xs text-ink focus-visible:outline-2 focus-visible:outline-accent cursor-pointer", className)} {...p} />;
}

export function Field({ label, children, className }: { label: React.ReactNode; children: React.ReactNode; className?: string }) {
  return <label className={cn("flex items-center gap-1.5 text-xs text-ink-2", className)}><span className="whitespace-nowrap">{label}</span>{children}</label>;
}

export function Stat({ label, value, sub }: { label: string; value: React.ReactNode; sub?: React.ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="text-[11px] text-muted">{label}</div>
      <div className="num truncate text-[15px] font-semibold text-ink">{value}</div>
      {sub && <div className="text-[11px] text-ink-2">{sub}</div>}
    </div>
  );
}

export function Notice({ tone = "neutral", children, className }: { tone?: "neutral" | "warn" | "bad"; children: React.ReactNode; className?: string }) {
  const c = tone === "bad" ? "border-bad/40 text-bad" : tone === "warn" ? "border-warn/40 text-ink-2" : "border-line text-ink-2";
  return <div className={cn("rounded-md border bg-surface-2 px-2.5 py-1.5 text-xs leading-relaxed", c, className)}>{children}</div>;
}

export function Formula({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("num overflow-x-auto rounded-md border border-line bg-surface-2 px-2.5 py-1.5 text-xs whitespace-pre text-ink", className)}>{children}</div>;
}
