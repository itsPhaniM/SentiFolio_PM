# SentiFolio

**Sentiment-aware, explainable machine learning for UK (FTSE) stock portfolio construction with honest backtesting.**

Does news and social-media sentiment actually improve UK equity portfolios, or just add noise? SentiFolio scores financial text with FinBERT, combines it with technical indicators in an interpretable LightGBM model (explained via SHAP), builds mean-variance and risk-parity portfolios, and evaluates them with a transaction-cost-aware walk-forward backtest and the Deflated Sharpe Ratio. A controlled price-only vs. price+sentiment ablation isolates sentiment's value.

## Setup
Requires Python 3.12 (3.14 is too new for the ML wheels).

```bash
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # install incrementally as phases are reached
```

## Usage
```bash
# Week 1: fetch FTSE price data -> data/raw/prices.parquet
.venv/Scripts/python.exe -m src.ingest.prices

# Week 2: fetch recent news headlines -> data/raw/news.parquet  (PRIMARY sentiment source)
.venv/Scripts/python.exe -m src.ingest.news

# Week 2: historical news for the backtest (date-windowed Google News, resumable)
#   -> data/raw/news_history.parquet
.venv/Scripts/python.exe -m src.ingest.news_history

# Week 3: FinBERT sentiment scoring (GPU, resumable)
#   -> data/processed/news_history_scored.parquet
.venv/Scripts/python.exe -m src.features.sentiment

# Optional: Reddit posts -> data/raw/reddit.parquet  (needs approved API creds in .env)
.venv/Scripts/python.exe -m src.ingest.reddit
```

**Sentiment source:** news headlines are the primary source (consistent with the
literature). A Reddit scraper (`src/ingest/reddit.py`) is included but optional —
Reddit closed self-service API access in Nov 2025, so live social-media data is scoped
as future work pending developer approval.

## Project layout
```
config.py            # tickers, date range, paths
src/ingest/          # data collection (prices, sentiment)
data/                # raw + processed data (git-ignored)
```

## Status
🚧 In development — Week 3. Prices ✅ · news (live + historical) ✅ · FinBERT sentiment scoring ✅ (86k headlines, GPU). Next: feature engineering (sentiment + technical indicators).
