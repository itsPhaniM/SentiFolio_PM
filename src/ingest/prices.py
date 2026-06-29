"""Ingest daily FTSE price/volume data via yfinance and store it as Parquet.

Run from the project root:
    .venv/Scripts/python.exe -m src.ingest.prices
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

# allow running as a script or as a module
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import TICKERS, START_DATE, END_DATE, RAW_DIR

COLS = ["date", "open", "high", "low", "close", "volume", "ticker"]


def fetch_one(ticker: str, start: str, end: str | None) -> pd.DataFrame | None:
    """Download a single ticker and return a tidy long-format frame."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        print(f"  WARN: no data returned for {ticker}")
        return None
    df = df.reset_index()
    # flatten any (field, ticker) MultiIndex columns -> field
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.rename(columns=str.lower)
    df["ticker"] = ticker
    return df[COLS]


def fetch_prices(tickers, start, end) -> pd.DataFrame:
    frames = []
    for t in tickers:
        out = fetch_one(t, start, end)
        if out is not None:
            frames.append(out)
            print(f"  {t:8s} {len(out):>5,} rows  {out['date'].min().date()} -> {out['date'].max().date()}")
    if not frames:
        raise RuntimeError("No price data fetched for any ticker.")
    return pd.concat(frames, ignore_index=True).dropna(subset=["close"])


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(TICKERS)} FTSE tickers from {START_DATE} ...")
    prices = fetch_prices(list(TICKERS), START_DATE, END_DATE)
    out_path = RAW_DIR / "prices.parquet"
    prices.to_parquet(out_path, index=False)
    print(f"\nSaved {len(prices):,} rows ({prices['ticker'].nunique()} tickers) -> {out_path}")


if __name__ == "__main__":
    main()
