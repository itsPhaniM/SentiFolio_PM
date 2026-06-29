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
```

## Project layout
```
config.py            # tickers, date range, paths
src/ingest/          # data collection (prices, sentiment)
data/                # raw + processed data (git-ignored)
```

## Status
🚧 In development — Week 1: data ingestion.
