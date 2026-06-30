"""Historical news ingestion via date-windowed Google News RSS queries.

Google News RSS honours the `after:`/`before:` search operators, so we can pull
*historical* headlines per company, one month at a time, back to NEWS_HISTORY_START.
Free, no API key, and the same source as the live scraper. The run is resumable: a
progress checkpoint records completed (ticker, month) windows, so re-running continues
where it left off (useful if Google throttles a long run).

Run (full history, all tickers):
    .venv/Scripts/python.exe -m src.ingest.news_history

Scoped test (env overrides):
    NEWS_HISTORY_ONLY=TSCO.L  NEWS_HISTORY_START=2025-01-01  ... -m src.ingest.news_history
"""
from __future__ import annotations
import os
import sys
import json
import time
import urllib.parse
from datetime import datetime, timezone, date
from pathlib import Path

import pandas as pd
import feedparser

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from config import TICKERS, RAW_DIR

GNEWS_URL = "https://news.google.com/rss/search?q={q}&hl=en-GB&gl=GB&ceid=GB:en"
AGENT = "Mozilla/5.0 (SentiFolio MSc research project)"
DELAY_S = 1.5
OUT = RAW_DIR / "news_history.parquet"
PROGRESS = RAW_DIR / "news_history_progress.json"

START = os.getenv("NEWS_HISTORY_START", "2020-01-01")
ONLY = [t.strip() for t in os.getenv("NEWS_HISTORY_ONLY", "").split(",") if t.strip()]
COLS = ["datetime", "ticker", "company", "headline", "source", "link"]


def month_windows(start: str) -> list[tuple[str, str]]:
    """Inclusive-start, exclusive-end month windows from `start` to the current month."""
    y, m = (int(x) for x in start.split("-")[:2])
    today = date.today()
    out = []
    while (y, m) <= (today.year, today.month):
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        out.append((date(y, m, 1).isoformat(), date(ny, nm, 1).isoformat()))
        y, m = ny, nm
    return out


def fetch_window(company: str, w_start: str, w_end: str) -> list[dict]:
    q = f'"{company}" after:{w_start} before:{w_end}'
    feed = feedparser.parse(GNEWS_URL.format(q=urllib.parse.quote(q)), agent=AGENT)
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
            "datetime": published, "ticker": None, "company": company,
            "headline": getattr(e, "title", "").strip(),
            "source": source, "link": getattr(e, "link", ""),
        })
    return rows


def merge(collected: pd.DataFrame, buffer: list[dict]) -> pd.DataFrame:
    if not buffer:
        return collected
    new = pd.DataFrame(buffer)
    out = pd.concat([collected, new], ignore_index=True) if len(collected) else new
    out = out.dropna(subset=["headline"])
    out = out[out["headline"] != ""].drop_duplicates(subset=["ticker", "headline", "datetime"])
    return out[COLS]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tickers = {t: c for t, c in TICKERS.items() if (not ONLY or t in ONLY)}
    windows = month_windows(START)
    done = set(json.loads(PROGRESS.read_text())) if PROGRESS.exists() else set()
    collected = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=COLS)
    buffer: list[dict] = []

    total = len(tickers) * len(windows)
    print(f"Historical news from {START}: {len(tickers)} tickers x {len(windows)} months = {total} queries")
    print(f"Resuming: {len(done)} windows already done, {len(collected):,} headlines on disk")

    n = 0
    for ticker, company in tickers.items():
        for w_start, w_end in windows:
            n += 1
            key = f"{ticker}|{w_start}"
            if key in done:
                continue
            try:
                rows = fetch_window(company, w_start, w_end)
                for r in rows:
                    r["ticker"] = ticker
                buffer.extend(rows)
                print(f"  [{n}/{total}] {ticker:8s} {w_start[:7]}: {len(rows):>3} headlines")
                done.add(key)
            except Exception as e:
                print(f"  [{n}/{total}] {ticker:8s} {w_start[:7]}: ERROR {e} (retry next run)")
                time.sleep(DELAY_S * 2)
                continue
            time.sleep(DELAY_S)
            if len(buffer) >= 300:  # periodic checkpoint
                collected = merge(collected, buffer)
                collected.to_parquet(OUT, index=False)
                PROGRESS.write_text(json.dumps(sorted(done)))
                buffer = []

    collected = merge(collected, buffer)
    collected.to_parquet(OUT, index=False)
    PROGRESS.write_text(json.dumps(sorted(done)))
    if len(collected):
        print(f"\nDONE. {len(collected):,} headlines, {collected['ticker'].nunique()} tickers, "
              f"{collected['datetime'].min()} -> {collected['datetime'].max()}")


if __name__ == "__main__":
    main()
