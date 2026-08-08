"""FastAPI service exposing SentiFolio forecasts, portfolio, SHAP and backtest results.

Run locally:
    .venv/Scripts/python.exe -m uvicorn src.serve.api:app --reload
Then browse the interactive docs at http://127.0.0.1:8000/docs
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from src.serve import inference

Arm = Literal["price_only", "price+sentiment"]

app = FastAPI(
    title="SentiFolio API",
    version="1.0",
    description="Explainable, sentiment-aware FTSE portfolio recommendations "
                "(LightGBM + SHAP, walk-forward backtested).",
)

# Allow the React dev server (Vite, default port 5173) to call the API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/forecasts")
def forecasts(arm: Arm = "price_only") -> dict:
    """20-day return forecasts for the FTSE universe, ranked best-to-worst."""
    return inference.latest_forecasts(arm)


@app.get("/portfolio")
def portfolio(arm: Arm = "price_only") -> dict:
    """Recommended top-K equal-weight portfolio from the latest forecasts."""
    return inference.recommend_portfolio(arm)


@app.get("/risk")
def risk(arm: Arm = "price_only") -> dict:
    """Risk profile of the recommended portfolio: annualised volatility, per-holding
    risk contributions, and the strategy's realised backtest risk metrics."""
    return inference.portfolio_risk(arm)


@app.get("/shap")
def shap() -> dict:
    """Mean |SHAP| feature importances (which features drive the forecasts)."""
    return inference.shap_importances()


@app.get("/backtest")
def backtest() -> dict:
    """Saved walk-forward backtest metrics for every strategy."""
    return inference.backtest_metrics()


@app.get("/equity")
def equity() -> dict:
    """Daily equity curves (growth of 1.0) for every strategy."""
    return inference.equity_curves()


@app.get("/regime")
def regime() -> dict:
    """Sharpe by market regime (high-volatility vs calm) and the sentiment ablation
    delta in each regime — the project's standout finding."""
    return inference.regime_summary()
