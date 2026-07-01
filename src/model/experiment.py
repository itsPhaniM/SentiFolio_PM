"""Give the model a fair shot: forward-return horizon sweep + light hyperparameter tuning.

For each horizon (1/5/10/20 trading days) it runs the price-only vs price+sentiment
ablation with date-based walk-forward CV, scored by daily cross-sectional rank IC.
Then it runs a small hyperparameter grid at the most promising horizon.

    .venv/Scripts/python.exe -m src.model.experiment
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from config import PROCESSED_DIR
from src.model.train import TECH, SENT, daily_rank_ic

HORIZONS = [1, 5, 10, 20]
ARMS = {"price_only": TECH, "price+sentiment": TECH + SENT}
COMMON = dict(random_state=42, n_jobs=-1, verbose=-1)
GRID = {
    "default":     dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=50,  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0),
    "reg_shallow": dict(n_estimators=700, learning_rate=0.02, num_leaves=15, min_child_samples=150, subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0),
    "deeper":      dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=30,  subsample=0.9, subsample_freq=1, colsample_bytree=0.9, reg_lambda=0.5),
}


def prep(df: pd.DataFrame, h: int) -> pd.DataFrame:
    d = df.sort_values(["ticker", "date"]).copy()
    d["tgt"] = d.groupby("ticker")["close"].transform(lambda s: s.shift(-h) / s - 1)
    return d[(d["date"] >= "2020-01-01") & d["tgt"].notna()].sort_values(["date", "ticker"]).reset_index(drop=True)


def cv_ic(d: pd.DataFrame, feats, params) -> float:
    dates = np.array(sorted(d["date"].unique()))
    ics = []
    for tr_idx, te_idx in TimeSeriesSplit(n_splits=5).split(dates):
        tr = d[d["date"].isin(dates[tr_idx])]
        te = d[d["date"].isin(dates[te_idx])]
        m = lgb.LGBMRegressor(**params, **COMMON).fit(tr[feats], tr["tgt"])
        ics.append(daily_rank_ic(te["date"].values, te["tgt"].values, m.predict(te[feats])))
    return float(np.mean(ics))


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")

    print("=== Horizon sweep (default params, mean rank IC over 5 folds) ===")
    print(f"{'horizon':8} {'price_only':>12} {'price+sent':>12} {'delta':>9}")
    best = None
    for h in HORIZONS:
        d = prep(df, h)
        ic_po = cv_ic(d, ARMS["price_only"], GRID["default"])
        ic_ps = cv_ic(d, ARMS["price+sentiment"], GRID["default"])
        print(f"{str(h) + 'd':8} {ic_po:>12.4f} {ic_ps:>12.4f} {ic_ps - ic_po:>+9.4f}")
        if best is None or ic_ps > best[1]:
            best = (h, ic_ps)

    bh = best[0]
    print(f"\nMost promising horizon (highest price+sentiment IC): {bh}d")
    print(f"\n=== Hyperparameter grid at {bh}d (mean rank IC over 5 folds) ===")
    d = prep(df, bh)
    print(f"{'config':14} {'price_only':>12} {'price+sent':>12} {'delta':>9}")
    for name, params in GRID.items():
        ic_po = cv_ic(d, ARMS["price_only"], params)
        ic_ps = cv_ic(d, ARMS["price+sentiment"], params)
        print(f"{name:14} {ic_po:>12.4f} {ic_ps:>12.4f} {ic_ps - ic_po:>+9.4f}")


if __name__ == "__main__":
    main()
