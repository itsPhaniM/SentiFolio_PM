# SentiFolio

**Sentiment-aware, explainable machine learning for UK (FTSE 100) stock portfolio construction with honest backtesting.**

Does news sentiment actually improve UK equity portfolios, or just add noise? SentiFolio scores financial news with FinBERT, combines it with technical indicators in an interpretable LightGBM model (explained via SHAP), builds mean-variance and risk-parity portfolios, and evaluates them with a transaction-cost-aware walk-forward backtest and the Deflated Sharpe Ratio. A controlled price-only vs. price+sentiment ablation isolates sentiment's value.

## Research questions

- **Q1** — Does adding news sentiment improve risk-adjusted returns over a price-only model, after realistic trading costs?
- **Q2** — Which features, price-based or sentiment-based, most influence the forecasts (via SHAP)?
- **Q3** — Do the model's portfolios beat passive benchmarks on a cost-aware, walk-forward basis?

## Key findings

The project is framed as a hypothesis test, and a well-checked "no" is a valid result.

- **Q1 — No robust gain from sentiment.** In cross-validation and in the backtest, adding sentiment shifts Sharpe by only a few hundredths and in no consistent direction (≈ −0.05, +0.07, 0.00 across the three allocators).
- **Q2 — Price features dominate.** The five most important SHAP features are all price-based (volatility, momentum, moving-average gaps); the strongest sentiment feature (3-day average sentiment) ranks only sixth.
- **Q3 — Passive is hard to beat.** The best active book earns a higher return (~23% CAGR) but not a higher Sharpe: buy-and-hold and equal-weight sit at ~1.27 and ~1.26, against ~1.18 for the best active strategy.
- **Regime nuance (the standout).** Splitting the walk-forward window by market volatility, sentiment clearly *helps* in high-volatility periods (≈ +0.24 to +0.44 Sharpe across allocators) but *hurts* in calm ones (≈ −0.22 to −0.73). The flat aggregate hides two opposite effects.

## Data

- **15 FTSE 100 names** across diverse sectors (energy, banks, pharma, mining, consumer, telecom, defence).
- **~32,000 daily OHLCV rows** (2018–2026, via yfinance).
- **~86,000 historical news headlines** (2020–2026, date-windowed Google News RSS), scored with FinBERT (~67% neutral / 18% positive / 14% negative).

## Setup

Requires Python 3.12 (3.14 is too new for the ML wheels).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt          # core pipeline
.\.venv\Scripts\python.exe -m pip install -r requirements-serve.txt    # API + dashboard
```

## Usage — full pipeline

Run from the project root, in order. Each stage reads the previous stage's output and prints a summary. A GPU is used automatically for FinBERT if available (falls back to CPU).

```powershell
# --- 1. Data ingestion (pulls live data; skip if data/ is already populated) ---
.\.venv\Scripts\python.exe -m src.ingest.prices          # -> data/raw/prices.parquet
.\.venv\Scripts\python.exe -m src.ingest.news            # -> data/raw/news.parquet (recent)
.\.venv\Scripts\python.exe -m src.ingest.news_history    # -> data/raw/news_history.parquet (resumable)

# --- 2. Sentiment scoring (FinBERT, GPU, resumable) ---
.\.venv\Scripts\python.exe -m src.features.sentiment     # -> data/processed/news_history_scored.parquet

# --- 3. Feature engineering (technical + sentiment) ---
.\.venv\Scripts\python.exe -m src.features.build_features # -> data/processed/features.parquet

# --- 4. Modelling ---
.\.venv\Scripts\python.exe -m src.model.experiment       # horizon + hyperparameter sweep (fair-shot)
.\.venv\Scripts\python.exe -m src.model.train            # walk-forward CV + ablation + SHAP

# --- 5. Portfolio construction + evaluation ---
.\.venv\Scripts\python.exe -m src.portfolio.signals      # walk-forward OOS forecasts -> signals.parquet
.\.venv\Scripts\python.exe -m src.portfolio.backtest     # cost-aware backtest + Deflated Sharpe
.\.venv\Scripts\python.exe -m src.portfolio.regime       # high-volatility vs calm regime robustness

# --- 6. Robustness checks ---
.\.venv\Scripts\python.exe -m scripts.survivorship_check   # coverage audit of the universe
.\.venv\Scripts\python.exe -m scripts.sentiment_agg_check  # is the null robust to aggregation scheme?
```

The processed data and reports are git-ignored, so a fresh clone regenerates them by running the pipeline above.

**Sentiment source:** news headlines are the primary source (consistent with the literature). A Reddit scraper (`src/ingest/reddit.py`) is included but optional — Reddit closed self-service API access in Nov 2025, so live social-media data is scoped as future work pending developer approval.

## Serving

```powershell
# Interactive dashboard (portfolio signals, risk panel, forecasts, SHAP, backtest)
.\.venv\Scripts\python.exe -m streamlit run src\serve\dashboard.py

# REST API (docs at http://127.0.0.1:8000/docs)
.\.venv\Scripts\python.exe -m uvicorn src.serve.api:app --reload
```

API endpoints: `/health`, `/forecasts`, `/portfolio`, `/risk`, `/shap`, `/backtest`.

Containerised run:

```powershell
docker compose up
```

## Backtest design

A transparent, hand-rolled pandas simulation (chosen for auditability, in keeping with the explainability theme). No look-ahead: daily returns are applied to the *held* weights before rebalancing, so new weights only take effect the next day. Trading frictions are split into a **10bps broker commission** and a **5bps execution slippage**, both charged on turnover at each rebalance. Benchmarks are equal-weight and buy-and-hold; the Deflated Sharpe Ratio (Bailey & López de Prado, 2014) corrects for the number of strategy trials.

## Project structure

```
config.py                     # tickers, date range, paths
src/ingest/
  prices.py                   # FTSE OHLCV via yfinance
  news.py                     # recent headlines (Google News RSS)
  news_history.py             # date-windowed historical headlines (resumable)
  reddit.py                   # optional social source (API-gated; future work)
src/features/
  sentiment.py                # FinBERT scoring (GPU, resumable)
  build_features.py           # technical + daily sentiment features -> modelling table
src/model/
  experiment.py               # horizon + hyperparameter sweep
  train.py                    # walk-forward CV, sentiment ablation, SHAP
src/portfolio/
  signals.py                  # walk-forward out-of-sample forecasts
  backtest.py                 # portfolio construction + cost-aware backtest + Deflated Sharpe
  regime.py                   # high-volatility vs calm regime robustness
src/serve/
  inference.py                # shared inference helpers
  api.py                      # FastAPI service
  dashboard.py                # Streamlit dashboard
scripts/                      # survivorship + aggregation robustness checks
tests/test_smoke.py           # smoke tests
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
data/                         # raw + processed data (git-ignored)
reports/                      # generated figures and CSV summaries (git-ignored)
```

## Tech stack

Python 3.12 · yfinance · Google News RSS + feedparser · FinBERT (HuggingFace Transformers + PyTorch) · LightGBM · SHAP · PyPortfolioOpt · pandas · FastAPI · Streamlit · Docker · GitHub Actions · Parquet storage.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Continuous integration runs the smoke tests on every push via GitHub Actions.

## Status

Complete — full pipeline implemented end to end: data ingestion, FinBERT sentiment, feature engineering, LightGBM modelling with walk-forward CV and SHAP, portfolio construction, cost-aware backtest with the Deflated Sharpe Ratio, regime and survivorship robustness checks, and a deployed API + dashboard.
