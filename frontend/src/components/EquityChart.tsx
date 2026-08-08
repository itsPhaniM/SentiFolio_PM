import { useQuery } from "@tanstack/react-query";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Panel } from "@/components/Panel";
import { getEquity } from "@/lib/api";

// The four curves worth showing: the two passive benchmarks and the two best
// active books. Colours match the terminal palette.
const SERIES: { key: string; label: string; color: string }[] = [
  { key: "buy_and_hold", label: "BUY & HOLD", color: "#6b7885" },
  { key: "equal_weight", label: "EQUAL WT", color: "#4cc9f0" },
  { key: "top_ew[price_only]", label: "TOP-5 · PO", color: "#f5a623" },
  { key: "top_ew[price+sentiment]", label: "TOP-5 · PS", color: "#2ee6a0" },
];

export function EquityChart() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["equity"],
    queryFn: getEquity,
  });

  // thin the series to ~180 points so the SVG stays light
  const rows = data?.curves ?? [];
  const step = Math.max(1, Math.floor(rows.length / 180));
  const thinned = rows.filter((_, i) => i % step === 0);

  return (
    <Panel
      title="Equity curves"
      tag="growth of 1.0 · net of costs"
      loading={isLoading}
      error={isError}
      className="lg:col-span-2"
    >
      <div className="h-[260px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={thinned} margin={{ left: -12, right: 12, top: 4 }}>
            <CartesianGrid stroke="#1e2733" vertical={false} />
            <XAxis
              dataKey="date"
              tick={{ fill: "#6b7885", fontSize: 10 }}
              tickFormatter={(d: string) => String(d).slice(0, 4)}
              minTickGap={40}
              axisLine={{ stroke: "#1e2733" }}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: "#6b7885", fontSize: 10 }}
              axisLine={false}
              tickLine={false}
              width={40}
              domain={["auto", "auto"]}
            />
            <Tooltip
              contentStyle={{
                background: "#0f141d",
                border: "1px solid #2b3745",
                fontSize: 11,
                color: "#c7d2de",
              }}
              labelStyle={{ color: "#6b7885" }}
              formatter={(v) => Number(v).toFixed(2)}
            />
            {SERIES.map((s) => (
              <Line
                key={s.key}
                type="monotone"
                dataKey={s.key}
                name={s.label}
                stroke={s.color}
                strokeWidth={1.5}
                dot={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-1 flex flex-wrap gap-4 text-[11px] text-term-muted">
        {SERIES.map((s) => (
          <span key={s.key} className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
    </Panel>
  );
}
