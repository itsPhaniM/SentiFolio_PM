"""Inference layer for the deployment: latest features -> portfolio recommendation.

Trains the best configuration (20-day horizon, 'reg_shallow' LightGBM) on every sample
whose forward-return window has already closed, then forecasts the most recent
cross-section of returns and forms a top-K equal-weight portfolio. Also surfaces the
saved SHAP importances and backtest metrics so the API / dashboard can present the full
story. Fitted models are cached in-process (per arm) so repeat calls are cheap.

No look-ahead: the newest date is scored using features as-of its close, while training
uses only rows whose 20-day target had already realised.
"""
from __future__ import annotations
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from config import PROCESSED_DIR, TICKERS, ROOT as PROJECT_ROOT
from src.portfolio.signals import HORIZON, CONFIG, COMMON, ARMS, TRAIN_START, build_target

TOP_K = 5
REPORT_DIR = PROJECT_ROOT / "reports" / "week5"
ARMS_AVAILABLE = tuple(ARMS)


@lru_cache(maxsize=4)
def _fit_and_predict(arm: str):
    """Fit the best-config model for `arm` on closed-target rows; return (as_of_date, ranked df)."""
    df = build_target(pd.read_parquet(PROCESSED_DIR / "features.parquet"), HORIZON)
    df = df[df["date"] >= TRAIN_START]
    feats = ARMS[arm]
    as_of = df["date"].max()
    model = lgb.LGBMRegressor(**CONFIG, **COMMON).fit(df.loc[df["tgt"].notna(), feats],
                                                      df.loc[df["tgt"].notna(), "tgt"])
    cur = df[df["date"] == as_of].copy()
    cur["pred"] = model.predict(cur[feats])
    cur["name"] = cur["ticker"].map(TICKERS)
    out = cur.sort_values("pred", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return as_of, out[["rank", "ticker", "name", "pred"]]


def latest_forecasts(arm: str = "price_only") -> dict:
    as_of, out = _fit_and_predict(arm)
    return {"as_of": str(pd.Timestamp(as_of).date()), "arm": arm, "horizon_days": HORIZON,
            "forecasts": out.to_dict("records")}


def recommend_portfolio(arm: str = "price_only") -> dict:
    as_of, out = _fit_and_predict(arm)
    top = out.head(TOP_K).copy()
    top["weight"] = round(1.0 / TOP_K, 4)
    return {"as_of": str(pd.Timestamp(as_of).date()), "arm": arm, "strategy": "top_ew",
            "n_holdings": TOP_K, "holdings": top[["ticker", "name", "pred", "weight"]].to_dict("records")}


def shap_importances() -> dict:
    s = pd.read_csv(PROCESSED_DIR / "model_shap_importance.csv", index_col=0).iloc[:, 0]
    return {"features": [{"feature": k, "mean_abs_shap": float(v)} for k, v in s.items()]}


def backtest_metrics() -> dict:
    m = pd.read_csv(REPORT_DIR / "backtest_metrics.csv")
    return {"strategies": m.to_dict("records")}
