"""Portfolio construction + transaction-cost-aware walk-forward backtest.

Consumes the walk-forward forecasts from signals.py and turns them into portfolios,
then simulates them day-by-day with realistic turnover costs and no look-ahead.

Allocators
  benchmarks (forecast-agnostic):
    equal_weight   all names, equal weight, rebalanced every period
    buy_and_hold   all names, equal weight once, then left to drift
  signal-driven (run per arm -> the RQ1 portfolio-level ablation):
    top_ew         top-K names by forecast, equal weight
    top_mv         top-K names, max-Sharpe mean-variance weights (PyPortfolioOpt)
    top_rp         top-K names, inverse-volatility (risk-parity) weights

Metrics: CAGR, annualised vol, Sharpe, Sortino, max drawdown, and the Deflated
Sharpe Ratio (Bailey & Lopez de Prado 2014) which corrects the Sharpe for the
number of strategy trials, non-normal returns and sample length.

Run (after signals.py):
    .venv/Scripts/python.exe -m src.portfolio.backtest
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pypfopt import EfficientFrontier, risk_models

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from config import PROCESSED_DIR, ROOT as PROJECT_ROOT

TOP_K = 5                     # hold the top third of the 15-name universe
COST_BPS = 10                 # one-way transaction cost per unit turnover (10 bps)
LOOKBACK = 252               # trading days of history for covariance / vol estimates
TRADING_DAYS = 252
MAX_WEIGHT = 0.40            # per-name cap for the mean-variance optimiser
REPORT_DIR = PROJECT_ROOT / "reports" / "week5"


# ---- weight builders (called at each rebalance date) --------------------------

def _inv_vol(trailing: pd.DataFrame, names) -> pd.Series:
    vol = trailing[names].std()
    w = (1.0 / vol).replace([np.inf, np.nan], 0.0)
    return w / w.sum() if w.sum() > 0 else pd.Series(1.0 / len(names), index=names)


def _max_sharpe(trailing: pd.DataFrame, names, preds: pd.Series) -> pd.Series:
    """Max-Sharpe mean-variance weights; forecasts as expected returns, shrunk cov."""
    mu = preds.loc[names] * (TRADING_DAYS / 20.0)              # 20d forecast -> annualised
    cov = risk_models.CovarianceShrinkage(trailing[names], returns_data=True).ledoit_wolf()
    try:
        ef = EfficientFrontier(mu, cov, weight_bounds=(0.0, MAX_WEIGHT))
        ef.max_sharpe(risk_free_rate=0.0)
        w = pd.Series(ef.clean_weights())
        return w.reindex(names).fillna(0.0)
    except Exception:
        return pd.Series(1.0 / len(names), index=names)        # fall back to equal weight


def build_schedules(sig: pd.DataFrame, rets: pd.DataFrame, tickers) -> dict[str, dict]:
    """Return {strategy_name: {rebalance_date: target-weight Series}} for every strategy."""
    rebal_dates = np.array(sorted(sig["date"].unique()))
    schedules: dict[str, dict] = {
        "equal_weight": {}, "buy_and_hold": {},
    }
    for arm in ("price_only", "price+sentiment"):
        for style in ("top_ew", "top_mv", "top_rp"):
            schedules[f"{style}[{arm}]"] = {}

    eq_all = pd.Series(1.0 / len(tickers), index=tickers)
    for t in rebal_dates:
        trailing = rets.loc[:t].tail(LOOKBACK)
        # benchmarks
        schedules["equal_weight"][t] = eq_all.copy()
        if t == rebal_dates[0]:
            schedules["buy_and_hold"][t] = eq_all.copy()       # set once, then drift
        # signal-driven, per arm
        for arm in ("price_only", "price+sentiment"):
            day = sig[(sig["date"] == t) & (sig["arm"] == arm)]
            preds = day.set_index("ticker")["pred"]
            top = preds.sort_values(ascending=False).head(TOP_K).index.tolist()
            schedules[f"top_ew[{arm}]"][t] = pd.Series(1.0 / TOP_K, index=top)
            schedules[f"top_rp[{arm}]"][t] = _inv_vol(trailing, top)
            schedules[f"top_mv[{arm}]"][t] = _max_sharpe(trailing, top, preds)
    return schedules


# ---- simulation ---------------------------------------------------------------

def simulate(weights_by_date: dict, rets: pd.DataFrame, tickers, cost_rate: float) -> pd.Series:
    """Day-by-day net returns. Each day: apply returns to held weights (they drift),
    then, if it is a rebalance date, trade to the new target and charge turnover cost.
    New weights therefore take effect the following day -> no look-ahead."""
    h = pd.Series(0.0, index=tickers)
    invested = False
    out = []
    for d in rets.index:
        r = rets.loc[d].reindex(tickers).fillna(0.0)
        if invested:
            gross = float((h * r).sum())
            h = h * (1.0 + r)
            s = h.sum()
            if s > 0:
                h = h / s
        else:
            gross = 0.0
        cost = 0.0
        if d in weights_by_date:
            w = weights_by_date[d].reindex(tickers).fillna(0.0)
            cost = cost_rate * float((w - h).abs().sum())
            h = w.copy()
            invested = True
        out.append(gross - cost)
    return pd.Series(out, index=rets.index, name="ret")


# ---- metrics ------------------------------------------------------------------

def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def deflated_sharpe(r: pd.Series, sr_daily_all, n_trials: int) -> float:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio (probability the true
    Sharpe > 0 after correcting for trial selection, skew/kurtosis and sample size)."""
    r = r[r != 0.0] if (r != 0.0).any() else r
    sr = r.mean() / r.std(ddof=1)
    T = len(r)
    sk = float(skew(r)); ku = float(kurtosis(r, fisher=False))
    var_sr = float(np.var(sr_daily_all, ddof=1))
    gamma = 0.5772156649015329
    sr0 = np.sqrt(var_sr) * ((1 - gamma) * norm.ppf(1 - 1.0 / n_trials)
                             + gamma * norm.ppf(1 - 1.0 / (n_trials * np.e)))
    num = (sr - sr0) * np.sqrt(T - 1)
    den = np.sqrt(max(1e-12, 1 - sk * sr + (ku - 1) / 4.0 * sr ** 2))
    return float(norm.cdf(num / den))


def metrics(r: pd.Series) -> dict:
    n = len(r)
    equity = (1 + r).cumprod()
    cagr = float(equity.iloc[-1] ** (TRADING_DAYS / n) - 1)
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0
    downside = r[r < 0].std(ddof=1)
    sortino = float(r.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else np.nan
    return dict(CAGR=cagr, vol=vol, Sharpe=sharpe, Sortino=sortino, maxDD=max_drawdown(equity))


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sig = pd.read_parquet(PROCESSED_DIR / "signals.parquet")
    feat = pd.read_parquet(PROCESSED_DIR / "features.parquet")

    close = feat.pivot_table(index="date", columns="ticker", values="close").sort_index()
    rets_full = close.pct_change(fill_method=None).fillna(0.0)
    tickers = list(close.columns)
    first_rebal = sig["date"].min()
    cost_rate = COST_BPS / 10000.0

    schedules = build_schedules(sig, rets_full, tickers)       # trailing stats use full history
    rets = rets_full.loc[rets_full.index >= first_rebal]       # simulate from when signals begin
    daily = {name: simulate(sched, rets, tickers, cost_rate) for name, sched in schedules.items()}

    # Deflated Sharpe needs the spread of (daily) Sharpes across all trials
    sr_daily_all = [r.mean() / r.std(ddof=1) for r in daily.values()]
    n_trials = len(daily)

    rows = []
    for name, r in daily.items():
        m = metrics(r)
        m["DSR"] = deflated_sharpe(r, sr_daily_all, n_trials)
        rows.append({"strategy": name, **m})
    table = pd.DataFrame(rows).set_index("strategy")
    order = ["CAGR", "vol", "Sharpe", "Sortino", "maxDD", "DSR"]
    table = table[order].sort_values("Sharpe", ascending=False)

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(f"\n=== Walk-forward backtest {first_rebal.date()} -> {rets.index.max().date()} "
          f"| K={TOP_K}, cost={COST_BPS}bps/side, {n_trials} strategies ===")
    print(table.to_string())
    out_csv = REPORT_DIR / "backtest_metrics.csv"
    table.to_csv(out_csv)

    # portfolio-level RQ1 ablation: does adding sentiment help the actual portfolios?
    print("\n=== RQ1 at the portfolio level (Sharpe: price+sentiment - price_only) ===")
    for style in ("top_ew", "top_mv", "top_rp"):
        po = table.loc[f"{style}[price_only]", "Sharpe"]
        ps = table.loc[f"{style}[price+sentiment]", "Sharpe"]
        verdict = "adds" if ps > po else "does NOT add"
        print(f"  {style:8} price_only={po:+.3f}  price+sent={ps:+.3f}  "
              f"delta={ps - po:+.3f}  -> sentiment {verdict} value")

    # equity curves
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, r in daily.items():
        style = "--" if name in ("equal_weight", "buy_and_hold") else "-"
        lw = 2.2 if name in ("equal_weight", "buy_and_hold") else 1.4
        ax.plot((1 + r).cumprod(), label=name, linestyle=style, linewidth=lw)
    ax.set_title(f"SentiFolio walk-forward equity curves ({COST_BPS}bps/side costs)")
    ax.set_ylabel("growth of 1.0"); ax.set_xlabel("date")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "equity_curves.png", dpi=130)
    print(f"\nSaved metrics -> {out_csv}")
    print(f"Saved equity curves -> {REPORT_DIR / 'equity_curves.png'}")


if __name__ == "__main__":
    main()
