"""Train LightGBM return-forecasting models with time-series CV and the sentiment ablation.

Two model 'arms' are trained under identical conditions:
  - price_only        : technical indicators only
  - price+sentiment   : technical indicators + FinBERT sentiment features

Evaluation is a date-based, walk-forward (expanding-window) cross-validation, so no
future information leaks into training. The key metric is the daily cross-sectional
rank Information Coefficient (rank IC) — the rank correlation between predicted and
actual forward returns across stocks each day — which is what matters for ranking
stocks into a portfolio. SHAP is computed on the price+sentiment model to see which
features (and whether the sentiment ones) drive the forecasts.

Run:
    .venv/Scripts/python.exe -m src.model.train
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from config import PROCESSED_DIR

TECH = ["ret_1d", "ma_gap_5", "ma_gap_20", "vol_20", "mom_10", "mom_20"]
SENT = ["sent_mean", "sent_vol", "sent_disp", "sent_pos_ratio", "sent_mean_3d", "sent_mean_7d", "sent_vol_7d"]
ARMS = {"price_only": TECH, "price+sentiment": TECH + SENT}
TARGET = "tgt"

# The configuration chosen by src/model/experiment.py: a 20-day forward-return horizon with
# the regularised ("reg_shallow") settings. These live here because train.py sits at the
# bottom of the import graph; src/portfolio/signals.py imports them, so the model explained
# by SHAP below is exactly the model that generates the trading signals.
HORIZON = 20
TRAIN_START = "2020-01-01"          # sentiment era only, so the ablation is like-for-like
CONFIG = dict(n_estimators=700, learning_rate=0.02, num_leaves=15, min_child_samples=150,
              subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0)
COMMON = dict(random_state=42, n_jobs=-1, verbose=-1)
LGB_PARAMS = {**CONFIG, **COMMON}


def build_target(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Attach `tgt`: the return over the next `h` trading days (strictly forward)."""
    d = df.sort_values(["ticker", "date"]).copy()
    d["tgt"] = d.groupby("ticker")["close"].transform(lambda s: s.shift(-h) / s - 1)
    return d


def daily_rank_ic(dates, y_true, y_pred) -> float:
    """Mean over days of the cross-sectional Spearman correlation (pred vs actual)."""
    d = pd.DataFrame({"date": dates, "y": y_true, "p": y_pred})
    ics = d.groupby("date").apply(
        lambda g: spearmanr(g["p"], g["y"]).statistic if len(g) > 2 else np.nan, include_groups=False)
    return float(np.nanmean(ics.values))


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    df = build_target(df, HORIZON)
    # fair ablation: only the period where sentiment exists, and a usable target
    df = df[(df["date"] >= TRAIN_START) & df[TARGET].notna()].sort_values(["date", "ticker"]).reset_index(drop=True)
    print(f"Modelling rows: {len(df):,} ({df['ticker'].nunique()} tickers, "
          f"{df['date'].min().date()} -> {df['date'].max().date()}) "
          f"| {HORIZON}-day horizon, regularised settings")

    dates = np.array(sorted(df["date"].unique()))
    tscv = TimeSeriesSplit(n_splits=5)
    results = {arm: {"rmse": [], "ic": [], "dir": []} for arm in ARMS}

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(dates), 1):
        tr = df[df["date"].isin(dates[tr_idx])]
        te = df[df["date"].isin(dates[te_idx])]
        for arm, feats in ARMS.items():
            model = lgb.LGBMRegressor(**LGB_PARAMS)
            model.fit(tr[feats], tr[TARGET])
            pred = model.predict(te[feats])
            actual = te[TARGET].values
            rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
            ic = daily_rank_ic(te["date"].values, actual, pred)
            dacc = float(np.mean(np.sign(pred) == np.sign(actual)))
            results[arm]["rmse"].append(rmse); results[arm]["ic"].append(ic); results[arm]["dir"].append(dacc)
        print(f"  fold {fold}: train<= {pd.Timestamp(dates[tr_idx].max()).date()}, "
              f"test {pd.Timestamp(dates[te_idx].min()).date()}..{pd.Timestamp(dates[te_idx].max()).date()} "
              f"({len(te):,} rows)")

    # naive baseline: predict zero
    base_rmse = float(np.sqrt(np.mean(df[TARGET].values ** 2)))
    print(f"\n=== Cross-validated results (mean over 5 folds) ===")
    print(f"{'arm':16} {'RMSE':>9} {'rank IC':>9} {'dir.acc':>9}")
    for arm in ARMS:
        r = results[arm]
        print(f"{arm:16} {np.mean(r['rmse']):9.5f} {np.mean(r['ic']):9.4f} {np.mean(r['dir']):9.3f}")
    print(f"{'baseline(zero)':16} {base_rmse:9.5f} {'0.0000':>9} {'n/a':>9}")

    ic_po = np.mean(results["price_only"]["ic"])
    ic_ps = np.mean(results["price+sentiment"]["ic"])
    delta = ic_ps - ic_po
    # Guard: a rank IC at or below zero means the arm ranks no better than chance, so
    # comparing two such arms is meaningless -- report that instead of declaring a winner.
    if max(ic_po, ic_ps) <= 0:
        verdict = ("NEITHER arm ranks better than chance at this horizon (both rank ICs <= 0), "
                   "so the comparison is not meaningful")
    else:
        verdict = f"sentiment {'adds' if delta > 0 else 'does NOT add'} value"
    print(f"\nAblation (RQ1): rank IC price-only = {ic_po:.4f} vs price+sentiment = {ic_ps:.4f} "
          f"-> {verdict} (delta = {delta:+.4f})")

    # SHAP on the full price+sentiment model
    import shap
    feats = ARMS["price+sentiment"]
    final = lgb.LGBMRegressor(**LGB_PARAMS).fit(df[feats], df[TARGET])
    expl = shap.TreeExplainer(final)
    sv = expl.shap_values(df[feats].sample(min(4000, len(df)), random_state=0))
    imp = pd.Series(np.abs(sv).mean(axis=0), index=feats).sort_values(ascending=False)
    print("\n=== SHAP mean|impact| (which features drive the forecast) ===")
    for f, v in imp.items():
        tag = "  <- sentiment" if f in SENT else ""
        print(f"  {f:16} {v:.5f}{tag}")

    out = PROCESSED_DIR / "model_shap_importance.csv"
    imp.rename("mean_abs_shap").to_csv(out)
    print(f"\nSaved SHAP importances -> {out}")


if __name__ == "__main__":
    main()
