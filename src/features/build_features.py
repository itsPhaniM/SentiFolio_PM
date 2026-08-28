"""Build the modelling table: technical indicators + daily sentiment features.

Combines:
  - data/raw/prices.parquet                  (OHLCV per ticker)
  - data/processed/news_history_scored.parquet (FinBERT-scored headlines)
into data/processed/features.parquet, one row per (ticker, trading day):

  Technical:  ret_1d, ma_gap_5, ma_gap_20, vol_20, mom_10, mom_20
  Sentiment:  sent_mean, sent_vol, sent_disp, sent_pos_ratio (+ 3d/7d rolling)

Features are as-of the close of each trading day, so there is no look-ahead
leakage. The forecast target is deliberately NOT stored here: src/model/train.py
builds it at train time from the single shared HORIZON, so the model explained by
SHAP is exactly the model that generates the trading signals.

Run:
    .venv/Scripts/python.exe -m src.features.build_features
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from config import RAW_DIR, PROCESSED_DIR


def technical_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    close = g["close"]
    g["ret_1d"] = close.pct_change()
    g["ma_gap_5"] = close / close.rolling(5).mean() - 1
    g["ma_gap_20"] = close / close.rolling(20).mean() - 1
    g["vol_20"] = g["ret_1d"].rolling(20).std()
    g["mom_10"] = close / close.shift(10) - 1
    g["mom_20"] = close / close.shift(20) - 1
    return g


def daily_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    n = news.copy()
    n["date"] = pd.to_datetime(n["datetime"]).dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    agg = n.groupby(["ticker", "date"]).agg(
        sent_mean=("sent_score", "mean"),
        sent_vol=("sent_score", "size"),
        sent_disp=("sent_score", "std"),
        sent_pos_ratio=("sent_label", lambda s: (s == "positive").mean()),
    ).reset_index()
    agg["sent_disp"] = agg["sent_disp"].fillna(0.0)  # single-item days -> 0 dispersion
    return agg


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(RAW_DIR / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    news = pd.read_parquet(PROCESSED_DIR / "news_history_scored.parquet")
    sent = daily_sentiment(news)

    out = []
    for ticker, g in prices.groupby("ticker"):
        g = technical_features(g)
        s = sent[sent["ticker"] == ticker][["date", "sent_mean", "sent_vol", "sent_disp", "sent_pos_ratio"]]
        m = g.merge(s, on="date", how="left")
        # days with no news: neutral mean, zero volume/dispersion
        m["sent_vol"] = m["sent_vol"].fillna(0.0)
        m["sent_mean"] = m["sent_mean"].fillna(0.0)
        m["sent_disp"] = m["sent_disp"].fillna(0.0)
        m["sent_pos_ratio"] = m["sent_pos_ratio"].fillna(0.0)
        # rolling sentiment to smooth sparsity
        m["sent_mean_3d"] = m["sent_mean"].rolling(3, min_periods=1).mean()
        m["sent_mean_7d"] = m["sent_mean"].rolling(7, min_periods=1).mean()
        m["sent_vol_7d"] = m["sent_vol"].rolling(7, min_periods=1).sum()
        out.append(m)

    feats = pd.concat(out, ignore_index=True)
    feats = feats.sort_values(["date", "ticker"]).reset_index(drop=True)
    out_path = PROCESSED_DIR / "features.parquet"
    feats.to_parquet(out_path, index=False)

    feat_cols = ["ret_1d", "ma_gap_5", "ma_gap_20", "vol_20", "mom_10", "mom_20",
                 "sent_mean", "sent_vol", "sent_disp", "sent_pos_ratio",
                 "sent_mean_3d", "sent_mean_7d", "sent_vol_7d"]
    print(f"Saved {len(feats):,} rows ({feats['ticker'].nunique()} tickers) -> {out_path}")
    print(f"date span: {feats['date'].min().date()} -> {feats['date'].max().date()}")
    print(f"feature columns: {feat_cols}")
    print(f"days with news coverage: {(feats['sent_vol'] > 0).mean():.1%} of ticker-days")


if __name__ == "__main__":
    main()
