import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Panel } from "@/components/Panel";
import { EquityChart } from "@/components/EquityChart";
import { RegimePanel } from "@/components/RegimePanel";
import {
  getBacktest,
  getForecasts,
  getHealth,
  getPortfolio,
  getRisk,
  getShap,
  pct,
  signedPct,
  SENT_FEATURES,
  type Arm,
} from "@/lib/api";

const AMBER = "#f5a623";
const CYAN = "#4cc9f0";
const GREEN = "#2ee6a0";
const RED = "#ff5f6d";

const ARMS: { id: Arm; label: string }[] = [
  { id: "price_only", label: "PRICE ONLY" },
  { id: "price+sentiment", label: "PRICE + SENT" },
];

function num(v: number, dp = 2) {
  return v.toFixed(dp);
}

export default function App() {
  const [arm, setArm] = useState<Arm>("price_only");
  const health = useQuery({ queryKey: ["health"], queryFn: getHealth });

  return (
    <div className="min-h-full bg-term-bg text-term-text">
      <TopBar arm={arm} setArm={setArm} online={health.data?.status === "ok"} />
      <main className="mx-auto max-w-[1400px] px-4 py-4">
        <KpiStrip arm={arm} />
        <div className="mt-3 grid grid-cols-1 gap-3 lg:grid-cols-2">
          <PortfolioPanel arm={arm} />
          <ShapPanel />
          <ForecastsPanel arm={arm} />
          <BacktestPanel />
          <EquityChart />
          <RegimePanel />
        </div>
        <footer className="mt-4 flex items-center justify-between border-t border-term-border pt-2 text-[11px] text-term-muted">
          <span>SENTIFOLIO · LIGHTGBM + SHAP · WALK-FORWARD BACKTESTED</span>
          <span>DATA: FTSE 100 · FINBERT SENTIMENT</span>
        </footer>
      </main>
    </div>
  );
}

function TopBar({
  arm,
  setArm,
  online,
}: {
  arm: Arm;
  setArm: (a: Arm) => void;
  online: boolean;
}) {
  return (
    <header className="flex flex-wrap items-center justify-between gap-3 border-b border-term-border bg-term-head px-4 py-2">
      <div className="flex items-baseline gap-3">
        <span className="text-base font-bold tracking-widest text-term-amber">
          ◆ SENTIFOLIO
        </span>
        <span className="text-[11px] uppercase tracking-wider text-term-muted">
          Explainable, sentiment-aware FTSE 100 portfolios
        </span>
      </div>
      <div className="flex items-center gap-4">
        <div className="flex border border-term-borderlt">
          {ARMS.map((a) => (
            <button
              key={a.id}
              onClick={() => setArm(a.id)}
              className={`px-3 py-1 text-[11px] font-medium tracking-wider transition-colors ${
                arm === a.id
                  ? "bg-term-amber text-term-bg"
                  : "text-term-muted hover:text-term-text"
              }`}
            >
              {a.label}
            </button>
          ))}
        </div>
        <span className="flex items-center gap-1.5 text-[11px] tracking-wider">
          <span
            className={`inline-block h-2 w-2 rounded-full ${
              online ? "bg-term-green" : "bg-term-red"
            }`}
          />
          <span className={online ? "text-term-green" : "text-term-red"}>
            {online ? "LIVE" : "OFFLINE"}
          </span>
        </span>
      </div>
    </header>
  );
}

function KpiStrip({ arm }: { arm: Arm }) {
  const { data } = useQuery({
    queryKey: ["risk", arm],
    queryFn: () => getRisk(arm),
  });
  const bt = data?.backtest_risk;
  const cells: { label: string; value: string; tone?: string }[] = [
    { label: "PORTFOLIO VOL", value: data ? pct(data.portfolio_vol_annual) : "—" },
    { label: "BACKTEST SHARPE", value: bt ? num(bt.Sharpe) : "—", tone: "text-term-amber" },
    { label: "CAGR", value: bt ? pct(bt.CAGR) : "—", tone: "text-term-green" },
    { label: "MAX DRAWDOWN", value: bt ? pct(bt.maxDD) : "—", tone: "text-term-red" },
    { label: "STRATEGY", value: data ? data.strategy.toUpperCase() : "—" },
    { label: "AS OF", value: data ? data.as_of : "—", tone: "text-term-muted" },
  ];
  return (
    <div className="grid grid-cols-2 gap-px border border-term-border bg-term-border sm:grid-cols-3 lg:grid-cols-6">
      {cells.map((c) => (
        <div key={c.label} className="bg-term-panel px-3 py-2">
          <div className="text-[10px] uppercase tracking-wider text-term-muted">
            {c.label}
          </div>
          <div className={`mt-0.5 text-lg font-medium ${c.tone ?? "text-term-text"}`}>
            {c.value}
          </div>
        </div>
      ))}
    </div>
  );
}

function PortfolioPanel({ arm }: { arm: Arm }) {
  const port = useQuery({ queryKey: ["portfolio", arm], queryFn: () => getPortfolio(arm) });
  const risk = useQuery({ queryKey: ["risk", arm], queryFn: () => getRisk(arm) });
  const rc = new Map(risk.data?.holdings.map((h) => [h.ticker, h.risk_contribution]));
  const maxRc = Math.max(...(risk.data?.holdings.map((h) => h.risk_contribution) ?? [1]));

  return (
    <Panel
      title="Portfolio"
      tag="top-5 equal weight"
      loading={port.isLoading}
      error={port.isError}
    >
      <table className="w-full text-[12px]">
        <thead>
          <tr className="text-left text-[10px] uppercase tracking-wider text-term-muted">
            <th className="pb-1 font-normal">Ticker</th>
            <th className="pb-1 font-normal">Name</th>
            <th className="pb-1 text-right font-normal">Forecast</th>
            <th className="pb-1 text-right font-normal">Wt</th>
            <th className="pb-1 pl-3 font-normal">Risk share</th>
          </tr>
        </thead>
        <tbody>
          {port.data?.holdings.map((h) => {
            const r = rc.get(h.ticker) ?? 0;
            return (
              <tr key={h.ticker} className="border-t border-term-border">
                <td className="py-1.5 text-term-amber">{h.ticker}</td>
                <td className="py-1.5 text-term-text">{h.name}</td>
                <td className="py-1.5 text-right text-term-green">
                  {signedPct(h.pred)}
                </td>
                <td className="py-1.5 text-right text-term-muted">{pct(h.weight, 0)}</td>
                <td className="py-1.5 pl-3">
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-24 bg-term-border">
                      <div
                        className="h-1.5 bg-term-cyan"
                        style={{ width: `${(r / maxRc) * 100}%` }}
                      />
                    </div>
                    <span className="w-9 text-right text-term-muted">{pct(r, 0)}</span>
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-2 text-[11px] text-term-muted">
        Equal weights, unequal risk — each holds{" "}
        {port.data ? pct(1 / port.data.n_holdings, 0) : "20%"} of capital.
      </p>
    </Panel>
  );
}

function ShapPanel() {
  const { data, isLoading, isError } = useQuery({ queryKey: ["shap"], queryFn: getShap });
  const rows =
    data?.features.map((f) => ({
      feature: f.feature,
      value: f.mean_abs_shap,
      sentiment: SENT_FEATURES.has(f.feature),
    })) ?? [];

  return (
    <Panel
      title="Factor attribution"
      tag="mean |SHAP|"
      loading={isLoading}
      error={isError}
    >
      <div className="h-[240px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 16 }}>
            <XAxis type="number" hide />
            <YAxis
              type="category"
              dataKey="feature"
              width={92}
              tick={{ fill: "#6b7885", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              cursor={{ fill: "#151c28" }}
              contentStyle={{
                background: "#0f141d",
                border: "1px solid #2b3745",
                fontSize: 12,
                color: "#c7d2de",
              }}
              formatter={(v) => [Number(v).toFixed(5), "mean |SHAP|"]}
            />
            <Bar dataKey="value" barSize={12}>
              {rows.map((r, i) => (
                <Cell key={i} fill={r.sentiment ? CYAN : AMBER} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-1 flex gap-4 text-[11px] text-term-muted">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5" style={{ background: AMBER }} /> Price
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5" style={{ background: CYAN }} /> Sentiment
        </span>
      </div>
    </Panel>
  );
}

function ForecastsPanel({ arm }: { arm: Arm }) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["forecasts", arm],
    queryFn: () => getForecasts(arm),
  });
  return (
    <Panel
      title="Forecasts"
      tag={`${data?.horizon_days ?? 20}-day horizon`}
      loading={isLoading}
      error={isError}
    >
      <table className="w-full text-[12px]">
        <tbody>
          {data?.forecasts.map((f) => (
            <tr key={f.ticker} className="border-t border-term-border first:border-t-0">
              <td className="w-6 py-1 text-term-muted">{f.rank}</td>
              <td className="py-1 text-term-amber">{f.ticker}</td>
              <td className="py-1 text-term-text">{f.name}</td>
              <td
                className={`py-1 text-right ${
                  f.pred >= 0 ? "text-term-green" : "text-term-red"
                }`}
              >
                {signedPct(f.pred)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}

function BacktestPanel() {
  const { data, isLoading, isError } = useQuery({ queryKey: ["backtest"], queryFn: getBacktest });
  const label = (s: string) =>
    s.replace("[price_only]", " · PO").replace("[price+sentiment]", " · PS");
  return (
    <Panel
      title="Backtest"
      tag="walk-forward · net of costs"
      loading={isLoading}
      error={isError}
    >
      <table className="w-full text-[12px]">
        <thead>
          <tr className="text-right text-[10px] uppercase tracking-wider text-term-muted">
            <th className="pb-1 text-left font-normal">Strategy</th>
            <th className="pb-1 font-normal">Sharpe</th>
            <th className="pb-1 font-normal">CAGR</th>
            <th className="pb-1 font-normal">Max DD</th>
            <th className="pb-1 font-normal">DSR</th>
          </tr>
        </thead>
        <tbody>
          {data?.strategies.map((s) => (
            <tr key={s.strategy} className="border-t border-term-border">
              <td className="py-1.5 text-left text-term-text">{label(s.strategy)}</td>
              <td className="py-1.5 text-right font-medium" style={{ color: AMBER }}>
                {num(s.Sharpe)}
              </td>
              <td className="py-1.5 text-right" style={{ color: GREEN }}>
                {pct(s.CAGR)}
              </td>
              <td className="py-1.5 text-right" style={{ color: RED }}>
                {pct(s.maxDD)}
              </td>
              <td className="py-1.5 text-right text-term-muted">{num(s.DSR)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}
