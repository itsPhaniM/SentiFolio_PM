import axios from "axios";

// In dev, Vite proxies /api/* -> FastAPI on :8000 (see vite.config.ts).
const api = axios.create({ baseURL: "/api" });

export type Arm = "price_only" | "price+sentiment";

export interface Forecast {
  rank: number;
  ticker: string;
  name: string;
  pred: number;
}
export interface Holding {
  ticker: string;
  name: string;
  pred: number;
  weight: number;
}
export interface RiskHolding {
  ticker: string;
  name: string;
  weight: number;
  risk_contribution: number;
}
export interface BacktestRisk {
  Sharpe: number;
  CAGR: number;
  vol: number;
  maxDD: number;
}
export interface Portfolio {
  as_of: string;
  arm: Arm;
  strategy: string;
  n_holdings: number;
  holdings: Holding[];
}
export interface Risk {
  as_of: string;
  arm: Arm;
  strategy: string;
  portfolio_vol_annual: number;
  holdings: RiskHolding[];
  backtest_risk: BacktestRisk | null;
}
export interface ForecastsResp {
  as_of: string;
  arm: Arm;
  horizon_days: number;
  forecasts: Forecast[];
}
export interface ShapFeature {
  feature: string;
  mean_abs_shap: number;
}
export interface ShapResp {
  features: ShapFeature[];
}
export interface Strategy {
  strategy: string;
  CAGR: number;
  vol: number;
  Sharpe: number;
  Sortino: number;
  maxDD: number;
  DSR: number;
}
export interface BacktestResp {
  strategies: Strategy[];
}
export interface EquityResp {
  strategies: string[];
  curves: Record<string, number | string>[]; // each row: { date, <strategy>: value, ... }
}
export interface RegimeRow {
  strategy: string;
  high_vol: number;
  calm: number;
}
export interface RegimeDelta {
  allocator: string;
  high_vol: number;
  calm: number;
}
export interface RegimeResp {
  strategies: RegimeRow[];
  sentiment_delta: RegimeDelta[];
}

export const getHealth = () =>
  api.get<{ status: string }>("/health").then((r) => r.data);
export const getPortfolio = (arm: Arm) =>
  api.get<Portfolio>("/portfolio", { params: { arm } }).then((r) => r.data);
export const getRisk = (arm: Arm) =>
  api.get<Risk>("/risk", { params: { arm } }).then((r) => r.data);
export const getForecasts = (arm: Arm) =>
  api.get<ForecastsResp>("/forecasts", { params: { arm } }).then((r) => r.data);
export const getShap = () => api.get<ShapResp>("/shap").then((r) => r.data);
export const getBacktest = () =>
  api.get<BacktestResp>("/backtest").then((r) => r.data);
export const getEquity = () =>
  api.get<EquityResp>("/equity").then((r) => r.data);
export const getRegime = () =>
  api.get<RegimeResp>("/regime").then((r) => r.data);

// Sentiment features (used to colour the SHAP chart).
export const SENT_FEATURES = new Set([
  "sent_mean",
  "sent_vol",
  "sent_disp",
  "sent_pos_ratio",
  "sent_mean_3d",
  "sent_mean_7d",
  "sent_vol_7d",
]);

export const pct = (v: number, dp = 1) => `${(v * 100).toFixed(dp)}%`;
export const signedPct = (v: number, dp = 2) =>
  `${v >= 0 ? "+" : ""}${(v * 100).toFixed(dp)}%`;
