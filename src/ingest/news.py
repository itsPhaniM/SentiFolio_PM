"""Ingest recent financial news headlines per FTSE company via Google News RSS.

No credentials required. Google News already tags results to the company we search
for, which avoids the ticker-disambiguation problem (e.g. "BP" the stock vs. other uses).
Headlines (not full articles) are collected, which is what FinBERT scores in the next step.

Run from the project root:
    .venv/Scripts/python.exe -m src.ingest.news
"""
from __future__ import annotations
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import feedparser

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import TICKERS, RAW_DIR

GNEWS_URL = "https://news.google.com/rss/search?q={query}&hl=en-GB&gl=GB&ceid=GB:en"
AGENT = "Mozilla/5.0 (SentiFolio MSc research project)"
DELAY_S = 1.0  # be polite between requests


def build_query(company: str) -> str:
    """Finance-focused query so generic names (BP, Shell) don't pull unrelated noise."""
    q = f'"{company}" (share price OR shares OR stock OR FTSE)'
    return urllib.parse.quote(q)


def fetch_company(ticker: str, company: str) -> list[dict]:
    feed = feedparser.parse(GNEWS_URL.format(query=build_query(company)), agent=AGENT)
    rows = []
    for e in feed.entries:
        published = None
        if getattr(e, "published_parsed", None):
            published = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
        source = ""
        src = getattr(e, "source", None)
        if src is not None and getattr(src, "title", None):
            source = src.title
        rows.append({
            "datetime": published,
            "ticker": ticker,
            "company": company,
            "headline": getattr(e, "title", "").strip(),
            "source": source,
            "link": getattr(e, "link", ""),
        })
    return rows


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    print(f"Fetching news for {len(TICKERS)} FTSE companies via Google News RSS ...")
    for ticker, company in TICKERS.items():
        rows = fetch_company(ticker, company)
        all_rows.extend(rows)
        print(f"  {ticker:8s} {company:22s} {len(rows):>3} headlines")
        time.sleep(DELAY_S)

    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=["headline"])
    df = df[df["headline"] != ""].drop_duplicates(subset=["ticker", "headline"])
    df = df.sort_values("datetime", ascending=False, na_position="last").reset_index(drop=True)

    out_path = RAW_DIR / "news.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df):,} unique headlines ({df['ticker'].nunique()} tickers) -> {out_path}")
    if len(df):
        print("\nMost recent sample:")
        print(df[["datetime", "ticker", "headline", "source"]].head(6).to_string(index=False))


if __name__ == "__main__":
    main()
