"""Generate walk-forward out-of-sample return forecasts for the portfolio stage.

Using the best configuration from the modelling experiment (20-day forward-return
horizon, regularised 'reg_shallow' LightGBM), this retrains on an expanding window
and forecasts the cross-section of returns at each rebalance date -- for BOTH arms
(price_only and price+sentiment) so the portfolio-level sentiment ablation can be
run downstream.

No look-ahead: at each rebalance date t the model is trained only on samples whose
forward-return window had fully closed on or before t (i.e. feature date <= t - H),
then used to score every stock as of t.

Run:
    .venv/Scripts/python.exe -m src.portfolio.signals
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from config import PROCESSED_DIR
# The horizon, model configuration and target builder are imported from src/model/train.py
# so there is a single source of truth: the model explained by SHAP in train.py is the same
# model that produces these signals. (They used to be duplicated here, which let the two
# files drift apart.) Re-exported for src/serve/inference.py and scripts/.
from src.model.train import (TECH, SENT, daily_rank_ic,
                             HORIZON, CONFIG, COMMON, TRAIN_START, build_target)

ARMS = {"price_only": TECH, "price+sentiment": TECH + SENT}
INITIAL_TRAIN_DAYS = 252     # ~1 year warm-up before the first rebalance


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    df = build_target(df, HORIZON)
    df = df[df["date"] >= TRAIN_START].sort_values(["date", "ticker"]).reset_index(drop=True)

    dates = np.array(sorted(df["date"].unique()))
    rebal_idx = list(range(INITIAL_TRAIN_DAYS, len(dates), HORIZON))
    rebal_dates = dates[rebal_idx]
    print(f"OOS rebalances: {len(rebal_dates)} every {HORIZON}d "
          f"({pd.Timestamp(rebal_dates[0]).date()} -> {pd.Timestamp(rebal_dates[-1]).date()})")

    rows = []
    for i in rebal_idx:
        t = dates[i]
        cutoff = dates[i - HORIZON]                       # fwd window must have closed by t
        tr = df[(df["date"] <= cutoff) & df["tgt"].notna()]
        cur = df[df["date"] == t]
        for arm, feats in ARMS.items():
            m = lgb.LGBMRegressor(**CONFIG, **COMMON).fit(tr[feats], tr["tgt"])
            pred = m.predict(cur[feats])
            for tk, p, fr in zip(cur["ticker"], pred, cur["tgt"]):
                rows.append((t, tk, arm, float(p), float(fr) if pd.notna(fr) else np.nan))

    sig = pd.DataFrame(rows, columns=["date", "ticker", "arm", "pred", "fwd_ret"])
    out = PROCESSED_DIR / "signals.parquet"
    sig.to_parquet(out, index=False)

    # OOS sanity check: rank IC of the forecasts actually used for allocation
    print("\n=== OOS forecast quality (rank IC over rebalance dates) ===")
    for arm in ARMS:
        s = sig[(sig["arm"] == arm) & sig["fwd_ret"].notna()]
        ic = daily_rank_ic(s["date"].values, s["fwd_ret"].values, s["pred"].values)
        print(f"  {arm:16} rank IC = {ic:+.4f}  ({s['date'].nunique()} dates)")
    print(f"\nSaved {len(sig):,} signal rows -> {out}")


if __name__ == "__main__":
    main()
