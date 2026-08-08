import { useQuery } from "@tanstack/react-query";
import { Panel } from "@/components/Panel";
import { getRegime } from "@/lib/api";

const ALLOC_LABEL: Record<string, string> = {
  top_ew: "Equal weight",
  top_mv: "Mean-variance",
  top_rp: "Risk parity",
};

function DeltaCell({ v }: { v: number }) {
  const pos = v >= 0;
  return (
    <span className={pos ? "text-term-green" : "text-term-red"}>
      {pos ? "+" : ""}
      {v.toFixed(2)}
    </span>
  );
}

// The standout finding: adding sentiment helps in high-volatility regimes and
// hurts in calm ones, so the flat aggregate hides two opposite effects.
export function RegimePanel() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["regime"],
    queryFn: getRegime,
  });

  return (
    <Panel
      title="Regime effect"
      tag="sentiment Δ Sharpe by market state"
      loading={isLoading}
      error={isError}
      className="lg:col-span-2"
    >
      <table className="w-full text-[12px]">
        <thead>
          <tr className="text-[10px] uppercase tracking-wider text-term-muted">
            <th className="pb-1 text-left font-normal">Allocator</th>
            <th className="pb-1 text-right font-normal">High volatility</th>
            <th className="pb-1 text-right font-normal">Calm</th>
            <th className="pb-1 pl-4 text-left font-normal">Read</th>
          </tr>
        </thead>
        <tbody>
          {data?.sentiment_delta.map((d) => (
            <tr key={d.allocator} className="border-t border-term-border">
              <td className="py-1.5 text-term-text">
                {ALLOC_LABEL[d.allocator] ?? d.allocator}
              </td>
              <td className="py-1.5 text-right">
                <DeltaCell v={d.high_vol} />
              </td>
              <td className="py-1.5 text-right">
                <DeltaCell v={d.calm} />
              </td>
              <td className="py-1.5 pl-4 text-[11px] text-term-muted">
                {d.high_vol > 0 ? "helps when turbulent" : "mixed"} · hurts when calm
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-[11px] leading-relaxed text-term-muted">
        Sentiment adds risk-adjusted value in high-volatility periods but detracts in
        calm ones. The two effects roughly cancel in the headline number — the reason the
        aggregate looks flat.
      </p>
    </Panel>
  );
}
