"""Survivorship-bias check for the FTSE universe.

Survivorship bias creeps in when a backtest only includes names that still exist (or
are still large) today, quietly dropping companies that were delisted or fell out of
the index. This script makes the risk explicit: for each of the 15 tickers it reports
the first/last date and how much of the common trading calendar it actually covers,
and flags anything that is not present for the whole window.

Finding (documented as a limitation): the universe is a fixed set of large, long-lived
FTSE 100 names that were continuously listed across 2018-2026, so within-window
survivorship bias is minimal. The genuine, honest caveat is selection at the START --
these names were chosen because they are prominent today. Point-in-time index
membership (which would remove that too) needs paid constituent history and is noted
as future work.

Run:
    .venv/Scripts/python.exe -m scripts.survivorship_check
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from config import PROCESSED_DIR, TICKERS

COVERAGE_FLAG = 0.98          # flag tickers present for < 98% of the common calendar


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")[["date", "ticker", "close"]]
    calendar = pd.Index(sorted(df["date"].unique()))
    start, end = calendar.min().date(), calendar.max().date()
    print(f"Universe: {len(TICKERS)} tickers | common calendar {start} -> {end} "
          f"({len(calendar):,} trading days)\n")

    rows = []
    for tk in TICKERS:
        g = df[df["ticker"] == tk]
        present = g["close"].notna().sum()
        cov = present / len(calendar)
        rows.append({
            "ticker": tk, "name": TICKERS[tk],
            "first": g["date"].min().date() if len(g) else None,
            "last": g["date"].max().date() if len(g) else None,
            "coverage": cov,
            "full_window": bool(cov >= COVERAGE_FLAG),
        })
    rep = pd.DataFrame(rows).sort_values("coverage")
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(rep.to_string(index=False))

    flagged = rep[~rep["full_window"]]
    print()
    if flagged.empty:
        print(f"PASS: all {len(TICKERS)} names cover >= {COVERAGE_FLAG:.0%} of the window "
              "-> no within-window survivorship gaps. Selection-at-start bias remains a "
              "documented limitation (see module docstring).")
    else:
        print(f"NOTE: {len(flagged)} name(s) below {COVERAGE_FLAG:.0%} coverage -> "
              "inspect for delisting / late listing:")
        print(flagged[["ticker", "name", "coverage"]].to_string(index=False))

    out = PROCESSED_DIR / "survivorship_check.csv"
    rep.to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
