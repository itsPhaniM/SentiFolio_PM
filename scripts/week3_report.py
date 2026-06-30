"""Generate human-readable Week 3 outputs (CSV summaries + charts) to reports/week3/.

    .venv/Scripts/python.exe scripts/week3_report.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from config import PROCESSED_DIR

OUT = ROOT / "reports" / "week3"
OUT.mkdir(parents=True, exist_ok=True)

scored = pd.read_parquet(PROCESSED_DIR / "news_history_scored.parquet")
feats = pd.read_parquet(PROCESSED_DIR / "features.parquet")

# ---- CSV 1: sample of scored headlines ----
sample = scored.sample(120, random_state=1)[["datetime", "ticker", "headline", "sent_label", "sent_score"]]
sample = sample.sort_values("datetime")
sample.to_csv(OUT / "scored_headlines_sample.csv", index=False)

# ---- CSV 2: sentiment summary by ticker ----
by_t = scored.groupby("ticker").agg(
    n_headlines=("headline", "size"),
    mean_sent=("sent_score", "mean"),
    pct_positive=("sent_label", lambda s: (s == "positive").mean()),
    pct_negative=("sent_label", lambda s: (s == "negative").mean()),
    pct_neutral=("sent_label", lambda s: (s == "neutral").mean()),
).round(3).sort_values("mean_sent", ascending=False)
by_t.to_csv(OUT / "sentiment_by_ticker.csv")

# ---- CSV 3: EDA correlations + quintiles ----
d = feats[(feats["date"] >= "2020-01-01") & feats["target_fwd_5d"].notna()].copy()
fcols = ["sent_mean", "sent_mean_3d", "sent_mean_7d", "sent_vol", "sent_pos_ratio",
         "mom_10", "mom_20", "vol_20", "ma_gap_20"]
corr = d[fcols + ["target_fwd_5d"]].corr()["target_fwd_5d"].drop("target_fwd_5d").round(4)
corr.rename("corr_with_fwd_5d_return").to_csv(OUT / "eda_feature_correlations.csv")

d["sent_quintile"] = pd.qcut(d["sent_mean_7d"].rank(method="first"), 5,
                             labels=["Q1 most neg", "Q2", "Q3", "Q4", "Q5 most pos"])
quint = (d.groupby("sent_quintile", observed=True)["target_fwd_5d"].mean() * 100).round(3)
quint.rename("mean_next5d_return_pct").to_csv(OUT / "eda_sentiment_quintiles.csv")

# ---- Charts ----
plt.figure(figsize=(5, 3.2))
scored["sent_label"].value_counts().reindex(["positive", "neutral", "negative"]).plot.bar(color=["#2e7d32", "#9e9e9e", "#c62828"])
plt.title("FinBERT sentiment label distribution (86k headlines)"); plt.ylabel("headlines"); plt.tight_layout()
plt.savefig(OUT / "sentiment_label_distribution.png", dpi=130); plt.close()

plt.figure(figsize=(6, 3.6))
by_t["mean_sent"].sort_values().plot.barh(color="#1565c0")
plt.title("Mean sentiment score by ticker"); plt.xlabel("mean P(pos) - P(neg)"); plt.tight_layout()
plt.savefig(OUT / "mean_sentiment_by_ticker.png", dpi=130); plt.close()

ts = scored.copy()
ts["month"] = pd.to_datetime(ts["datetime"]).dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
monthly = ts.groupby("month")["sent_score"].mean()
plt.figure(figsize=(7, 3.2))
monthly.plot(color="#6a1b9a"); plt.axhline(0, color="grey", lw=0.7)
plt.title("Average news sentiment over time (all FTSE tickers)"); plt.ylabel("mean sentiment"); plt.tight_layout()
plt.savefig(OUT / "sentiment_over_time.png", dpi=130); plt.close()

plt.figure(figsize=(5.5, 3.2))
quint.plot.bar(color="#00838f"); plt.title("Next-5d return by sentiment quintile")
plt.ylabel("mean return (%)"); plt.xticks(rotation=20); plt.tight_layout()
plt.savefig(OUT / "forward_return_by_sentiment_quintile.png", dpi=130); plt.close()

print("Week 3 report written to:", OUT)
for f in sorted(OUT.iterdir()):
    print("  -", f.name)
