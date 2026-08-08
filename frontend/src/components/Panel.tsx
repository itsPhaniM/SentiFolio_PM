import type { ReactNode } from "react";

interface PanelProps {
  title: string;
  tag?: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  loading?: boolean;
  error?: boolean;
}

// A terminal-style panel: a titled header bar over a bordered body.
export function Panel({
  title,
  tag,
  right,
  children,
  className = "",
  loading,
  error,
}: PanelProps) {
  return (
    <section
      className={`flex flex-col border border-term-border bg-term-panel ${className}`}
    >
      <header className="flex items-center justify-between border-b border-term-border bg-term-head px-3 py-1.5">
        <div className="flex items-center gap-2">
          <span className="text-term-amber">▸</span>
          <span className="text-[11px] font-medium uppercase tracking-widest text-term-text">
            {title}
          </span>
          {tag && (
            <span className="text-[10px] uppercase tracking-wider text-term-muted">
              {tag}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {loading && (
            <span className="animate-pulse text-[10px] uppercase tracking-wider text-term-amber">
              loading
            </span>
          )}
          {right}
        </div>
      </header>
      <div className="flex-1 p-3">
        {error ? (
          <div className="py-6 text-center text-[12px] text-term-red">
            Backend unreachable — start the API on :8000.
          </div>
        ) : (
          children
        )}
      </div>
    </section>
  );
}
