"""Persist the model-selection and per-fold evidence that the pipeline computes
but never wrote to disk.

`src/model/experiment.py` prints the horizon sweep and the hyperparameter grid,
and `src/model/train.py` prints only the mean of its five per-fold rank ICs.
Both are needed as reportable evidence, so this script recomputes them and
saves them as CSVs, without touching any existing artefact.

Outputs (reports/week4/):
    config_sweep.csv     horizon x settings-profile sweep, both arms
    per_fold_ic.csv      per-fold rank IC for each arm, plus the paired difference
    fold_significance.txt  paired test on the per-fold differences

Run:
    .venv/Scripts/python.exe scripts/evidence_report.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import wilcoxon, ttest_rel
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from config import PROCESSED_DIR                      # noqa: E402
from src.model.train import (                          # noqa: E402
    ARMS, TARGET, TRAIN_START, COMMON, build_target, daily_rank_ic,
)

OUT = ROOT / "reports" / "week4"
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 5, 10, 20]
PROFILES = {
    "balanced":    dict(n_estimators=500, learning_rate=0.05, num_leaves=31,
                        min_child_samples=40),
    "regularised": dict(n_estimators=700, learning_rate=0.02, num_leaves=15,
                        min_child_samples=150, subsample=0.7, subsample_freq=1,
                        colsample_bytree=0.7, reg_lambda=5.0),
    "aggressive":  dict(n_estimators=900, learning_rate=0.08, num_leaves=63,
                        min_child_samples=20),
}


def fold_ics(df: pd.DataFrame, feats: list[str], params: dict) -> list[float]:
    """Rank IC per walk-forward fold for one arm."""
    dates = np.sort(df["date"].unique())
    ics = []
    for tr_idx, te_idx in TimeSeriesSplit(n_splits=5).split(dates):
        tr = df[df["date"].isin(dates[tr_idx])]
        te = df[df["date"].isin(dates[te_idx])]
        model = lgb.LGBMRegressor(**{**params, **COMMON}).fit(tr[feats], tr[TARGET])
        pred = model.predict(te[feats])
        ics.append(daily_rank_ic(te["date"].values, te[TARGET].values, pred))
    return ics


def frame_for(horizon: int) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    df = build_target(df, horizon)
    return (df[(df["date"] >= TRAIN_START) & df[TARGET].notna()]
            .sort_values(["date", "ticker"]).reset_index(drop=True))


def main() -> None:
    # ---------- 1. configuration sweep -------------------------------------
    rows = []
    for h in HORIZONS:
        d = frame_for(h)
        for pname, params in PROFILES.items():
            rec = {"horizon_days": h, "profile": pname, "rows": len(d)}
            for arm, feats in ARMS.items():
                rec[arm] = float(np.mean(fold_ics(d, feats, params)))
            rec["delta"] = rec["price+sentiment"] - rec["price_only"]
            rows.append(rec)
            print(f"  {h:>2}d {pname:<12} "
                  f"PO {rec['price_only']:+.4f}  PS {rec['price+sentiment']:+.4f}  "
                  f"delta {rec['delta']:+.4f}")
    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / "config_sweep.csv", index=False)
    print(f"\nwrote {OUT / 'config_sweep.csv'}  ({len(sweep)} configurations)")

    # ---------- 2. per-fold rank IC at the chosen configuration -------------
    d20 = frame_for(20)
    per_arm = {arm: fold_ics(d20, feats, PROFILES["regularised"])
               for arm, feats in ARMS.items()}
    folds = pd.DataFrame({
        "fold": range(1, 6),
        "price_only": per_arm["price_only"],
        "price+sentiment": per_arm["price+sentiment"],
    })
    folds["delta"] = folds["price+sentiment"] - folds["price_only"]
    folds.to_csv(OUT / "per_fold_ic.csv", index=False)
    print(f"wrote {OUT / 'per_fold_ic.csv'}")
    print(folds.round(4).to_string(index=False))

    # ---------- 3. is the ablation difference distinguishable from zero? ----
    d = folds["delta"].values
    lines = [
        "Paired comparison of per-fold rank IC, price+sentiment vs price-only",
        "20-day horizon, regularised configuration, five walk-forward folds",
        "",
        f"mean difference : {d.mean():+.5f}",
        f"std deviation   : {d.std(ddof=1):.5f}",
        f"folds positive  : {int((d > 0).sum())} of {len(d)}",
        "",
    ]
    t_stat, t_p = ttest_rel(folds["price+sentiment"], folds["price_only"])
    lines.append(f"paired t-test      : t = {t_stat:+.3f},  p = {t_p:.3f}")
    try:
        w_stat, w_p = wilcoxon(folds["price+sentiment"], folds["price_only"])
        lines.append(f"Wilcoxon signed-rank: W = {w_stat:.1f},  p = {w_p:.3f}")
    except ValueError as e:
        lines.append(f"Wilcoxon signed-rank: not computed ({e})")
    lines += [
        "",
        "With five folds the test has very little power, so a non-significant",
        "result is not evidence of equivalence. It is reported because a",
        "difference that cannot be distinguished from zero on the metric that",
        "most favours sentiment is consistent with the cost-aware backtest,",
        "which is treated as the arbiter throughout.",
    ]
    (OUT / "fold_significance.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
