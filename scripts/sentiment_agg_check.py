"""Robustness check: does the sentiment AGGREGATION choice change the answer?

The main pipeline aggregates FinBERT headline scores into a simple daily average
(plus rolling smoothing). A fair question is whether a smarter aggregation would let
sentiment add value. This compares three schemes on the same model (best config, 20d
horizon) by walk-forward rank IC, against the fixed price-only baseline:

  daily_mean       - plain daily average, smoothed over 3/7 days (the current approach)
  volume_weighted  - the 3/7-day smoothing weights each day by its news volume, so
                     heavily-covered days count more ("weighted news impact")
  event_only       - sentiment only registers on days that actually had news, carried
                     forward up to 5 days ("event-based scoring")

If none of them lifts the price+sentiment IC above price-only, the null result is not
an artefact of naive averaging.

Run:
    .venv/Scripts/python.exe -m scripts.sentiment_agg_check
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from config import PROCESSED_DIR
from src.model.train import TECH, daily_rank_ic
from src.portfolio.signals import CONFIG, COMMON

H = 20                        # forecast horizon (matches the portfolio signals)
PARAMS = {**CONFIG, **COMMON}
SENT_COLS = ["s0", "s3", "s7", "sent_vol", "sent_disp", "sent_pos_ratio", "sent_vol_7d"]


def daily_agg(news: pd.DataFrame) -> pd.DataFrame:
    n = news.copy()
    n["date"] = pd.to_datetime(n["datetime"]).dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    g = n.groupby(["ticker", "date"]).agg(
        smean=("sent_score", "mean"), nvol=("sent_score", "size"),
        disp=("sent_score", "std"), pos=("sent_label", lambda s: (s == "positive").mean()),
    ).reset_index()
    g["disp"] = g["disp"].fillna(0.0)
    return g


def make_scheme(feat: pd.DataFrame, agg: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """Attach sentiment columns (s0/s3/s7 + vol/disp/pos) built with `scheme` aggregation."""
    out = []
    for tk, gp in feat.groupby("ticker"):
        gp = gp.sort_values("date")
        s = agg[agg["ticker"] == tk].set_index("date")
        idx = pd.Index(gp["date"].values)
        smean = pd.Series(s["smean"].reindex(idx).values)
        nvol = pd.Series(s["nvol"].reindex(idx).fillna(0.0).values)
        disp = pd.Series(s["disp"].reindex(idx).fillna(0.0).values)
        pos = pd.Series(s["pos"].reindex(idx).fillna(0.0).values)

        b = pd.DataFrame({"ticker": tk, "date": gp["date"].values})
        b["sent_vol"] = nvol.values
        b["sent_disp"] = disp.values
        b["sent_pos_ratio"] = pos.values
        b["sent_vol_7d"] = nvol.rolling(7, min_periods=1).sum().values

        if scheme == "daily_mean":
            m = smean.fillna(0.0)
            b["s0"], b["s3"], b["s7"] = m.values, m.rolling(3, 1).mean().values, m.rolling(7, 1).mean().values
        elif scheme == "volume_weighted":
            m, w = smean.fillna(0.0), nvol
            b["s0"] = m.values
            b["s3"] = ((m * w).rolling(3, 1).sum() / w.rolling(3, 1).sum()).fillna(0.0).values
            b["s7"] = ((m * w).rolling(7, 1).sum() / w.rolling(7, 1).sum()).fillna(0.0).values
        elif scheme == "event_only":
            evf = smean.ffill(limit=5).fillna(0.0)               # carry news forward up to 5 days
            b["s0"], b["s3"], b["s7"] = evf.values, evf.rolling(3, 1).mean().values, evf.rolling(7, 1).mean().values
        out.append(b)
    S = pd.concat(out, ignore_index=True)
    return feat.merge(S, on=["ticker", "date"], how="left")


def cv_ic(d: pd.DataFrame, feats) -> float:
    dates = np.array(sorted(d["date"].unique()))
    ics = []
    for tr, te in TimeSeriesSplit(n_splits=5).split(dates):
        trn = d[d["date"].isin(dates[tr])]; ten = d[d["date"].isin(dates[te])]
        m = lgb.LGBMRegressor(**PARAMS).fit(trn[feats], trn["tgt"])
        ics.append(daily_rank_ic(ten["date"].values, ten["tgt"].values, m.predict(ten[feats])))
    return float(np.mean(ics))


def main() -> None:
    feat = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    feat = feat.sort_values(["ticker", "date"]).copy()
    feat["tgt"] = feat.groupby("ticker")["close"].transform(lambda s: s.shift(-H) / s - 1)
    # keep only the technical columns; the sentiment columns are rebuilt per scheme below
    feat = feat.loc[feat["date"] >= "2020-01-01", ["date", "ticker", "close", "tgt"] + TECH]
    agg = daily_agg(pd.read_parquet(PROCESSED_DIR / "news_history_scored.parquet"))

    base = feat[feat["tgt"].notna()]
    ic_price = cv_ic(base, TECH)
    print(f"=== Sentiment-aggregation robustness (20d horizon, walk-forward rank IC) ===")
    print(f"{'scheme':16} {'price+sent':>12} {'price_only':>12} {'delta':>9}")
    print(f"{'(baseline)':16} {'-':>12} {ic_price:>12.4f} {'-':>9}")
    for scheme in ("daily_mean", "volume_weighted", "event_only"):
        d = make_scheme(feat, agg, scheme)
        d = d[d["tgt"].notna()]
        ic = cv_ic(d, TECH + SENT_COLS)
        print(f"{scheme:16} {ic:>12.4f} {ic_price:>12.4f} {ic - ic_price:>+9.4f}")
    print("\nRead: if every scheme's price+sentiment IC sits at or below price_only, the "
          "'sentiment adds no clear value' result is robust to how sentiment is aggregated.")


if __name__ == "__main__":
    main()
