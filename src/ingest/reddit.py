"""Ingest recent Reddit posts mentioning FTSE companies (read-only PRAW).

Requires a free Reddit "script" app. Put credentials in a .env file at the project
root (copy from .env.example):
    REDDIT_CLIENT_ID=...
    REDDIT_CLIENT_SECRET=...
    REDDIT_USER_AGENT=SentiFolio/0.1 by u/your_username

The free Reddit API returns recent posts only (it does not expose deep history); this
script is the live/recent social-media source. Historical depth for the backtest comes
from a pre-collected dataset (separate ingestion step).

Run from the project root:
    .venv/Scripts/python.exe -m src.ingest.reddit
"""
from __future__ import annotations
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import praw

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from config import TICKERS, SUBREDDITS, RAW_DIR

load_dotenv(ROOT / ".env")
POSTS_PER_QUERY = 100


def get_reddit() -> praw.Reddit:
    cid = os.getenv("REDDIT_CLIENT_ID")
    sec = os.getenv("REDDIT_CLIENT_SECRET")
    ua = os.getenv("REDDIT_USER_AGENT")
    if not all([cid, sec, ua]) or "your_client_id" in (cid or ""):
        raise SystemExit("Missing Reddit credentials. Copy .env.example to .env and fill in your values.")
    reddit = praw.Reddit(client_id=cid, client_secret=sec, user_agent=ua, check_for_async=False)
    reddit.read_only = True
    return reddit


def fetch_company(reddit: praw.Reddit, ticker: str, company: str) -> list[dict]:
    multi = reddit.subreddit("+".join(SUBREDDITS))
    rows = []
    for s in multi.search(f'"{company}"', sort="new", time_filter="year", limit=POSTS_PER_QUERY):
        rows.append({
            "datetime": datetime.fromtimestamp(s.created_utc, tz=timezone.utc),
            "ticker": ticker,
            "company": company,
            "subreddit": str(s.subreddit),
            "title": s.title or "",
            "text": s.selftext or "",
            "score": int(s.score),
            "num_comments": int(s.num_comments),
            "id": s.id,
        })
    return rows


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    reddit = get_reddit()
    print(f"Searching r/{'+'.join(SUBREDDITS)} for {len(TICKERS)} companies ...")
    all_rows: list[dict] = []
    for ticker, company in TICKERS.items():
        try:
            rows = fetch_company(reddit, ticker, company)
        except Exception as e:  # keep going if one query fails
            print(f"  {ticker}: ERROR {e}")
            rows = []
        all_rows.extend(rows)
        print(f"  {ticker:8s} {company:22s} {len(rows):>3} posts")
        time.sleep(1.0)

    df = pd.DataFrame(all_rows)
    if len(df):
        df = df.drop_duplicates(subset=["id"]).sort_values("datetime", ascending=False).reset_index(drop=True)
    out_path = RAW_DIR / "reddit.parquet"
    df.to_parquet(out_path, index=False)
    n_tickers = df["ticker"].nunique() if len(df) else 0
    print(f"\nSaved {len(df):,} posts ({n_tickers} tickers) -> {out_path}")
    if len(df):
        print("\nMost recent sample:")
        print(df[["datetime", "ticker", "subreddit", "title", "score"]].head(6).to_string(index=False))


if __name__ == "__main__":
    main()
