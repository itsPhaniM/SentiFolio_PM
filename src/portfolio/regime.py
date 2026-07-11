"""Regime-conditioned robustness check for the backtest.

A single headline Sharpe can hide the fact that a strategy only works in one kind of
market. This splits the walk-forward period into two regimes -- high-volatility and
calm -- and recomputes the key metrics, and the price-only vs price+sentiment
ablation, separately in each. That answers a direct piece of supervisor feedback:
does the edge (and does sentiment) hold up across different market conditions?

Regime definition (leak-free): the market proxy is the equal-weight daily return of
the 15-name universe; its trailing 21-day realised volatility is compared to the
median over the whole test window. Days at or above the median are "high_vol", the
rest "calm". Every day is classified using only its own trailing window.

Run (after backtest.py's inputs exist):
    .venv/Scripts/python.exe -m src.portfolio.regime
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from config import PROCESSED_DIR, ROOT as PROJECT_ROOT
from src.portfolio.backtest import load_daily, metrics, TRADING_DAYS, COMMISSION_BPS, SLIPPAGE_BPS

VOL_WINDOW = 21               # trading days for the market-volatility estimate
REPORT_DIR = PROJECT_ROOT / "reports" / "week5"


def market_regime(rets: pd.DataFrame) -> pd.Series:
    """Label each date 'high_vol' or 'calm' from the market proxy's trailing vol."""
    market = rets.mean(axis=1)                                 # equal-weight market return
    trail_vol = market.rolling(VOL_WINDOW).std()
    med = trail_vol.median()
    regime = np.where(trail_vol >= med, "high_vol", "calm")
    return pd.Series(regime, index=rets.index, name="regime").where(trail_vol.notna(), np.nan)


def regime_metrics(r: pd.Series, regime: pd.Series, label: str) -> dict:
    sub = r[regime == label]
    if len(sub) < 20:
        return dict(days=len(sub), CAGR=np.nan, vol=np.nan, Sharpe=np.nan, maxDD=np.nan)
    m = metrics(sub)
    return dict(days=len(sub), CAGR=m["CAGR"], vol=m["vol"], Sharpe=m["Sharpe"], maxDD=m["maxDD"])


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    daily, rets, first_rebal = load_daily()
    regime = market_regime(rets)
    n_hi = int((regime == "high_vol").sum()); n_ca = int((regime == "calm").sum())
    print(f"=== Regime robustness {first_rebal.date()} -> {rets.index.max().date()} "
          f"| commission+slippage {COMMISSION_BPS}+{SLIPPAGE_BPS}bps ===")
    print(f"Regime split (market trailing {VOL_WINDOW}d vol vs its median): "
          f"high_vol={n_hi} days, calm={n_ca} days\n")

    rows = []
    for name, r in daily.items():
        for lab in ("high_vol", "calm"):
            rm = regime_metrics(r, regime, lab)
            rows.append({"strategy": name, "regime": lab, **rm})
    table = pd.DataFrame(rows)
    wide = table.pivot(index="strategy", columns="regime", values="Sharpe")
    wide = wide.rename(columns={"high_vol": "Sharpe_high_vol", "calm": "Sharpe_calm"})
    wide = wide.sort_values("Sharpe_calm", ascending=False)

    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print("=== Sharpe by regime (per strategy) ===")
    print(wide.to_string())

    out_csv = REPORT_DIR / "regime_metrics.csv"
    table.set_index(["strategy", "regime"]).to_csv(out_csv)

    # sentiment ablation inside each regime
    print("\n=== RQ1 by regime (Sharpe: price+sentiment - price_only) ===")
    for lab in ("high_vol", "calm"):
        print(f"  [{lab}]")
        for style in ("top_ew", "top_mv", "top_rp"):
            po = regime_metrics(daily[f"{style}[price_only]"], regime, lab)["Sharpe"]
            ps = regime_metrics(daily[f"{style}[price+sentiment]"], regime, lab)["Sharpe"]
            verdict = "adds" if ps > po else "does NOT add"
            print(f"    {style:8} price_only={po:+.3f}  price+sent={ps:+.3f}  "
                  f"delta={ps - po:+.3f}  -> sentiment {verdict}")

    # visualise the regime bands under the equal-weight benchmark equity curve
    fig, ax = plt.subplots(figsize=(11, 4.5))
    eq = (1 + daily["equal_weight"]).cumprod()
    ax.plot(eq.index, eq.values, color="#1f4e79", lw=1.6, label="equal_weight equity")
    hi = regime == "high_vol"
    ax.fill_between(eq.index, eq.min(), eq.max(), where=hi.values, color="#e06666",
                    alpha=0.15, label="high-volatility regime")
    ax.set_title("Market regimes over the backtest (shaded = high volatility)")
    ax.set_ylabel("growth of 1.0"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "regime_bands.png", dpi=130)
    print(f"\nSaved regime metrics -> {out_csv}")
    print(f"Saved regime chart   -> {REPORT_DIR / 'regime_bands.png'}")


if __name__ == "__main__":
    main()
