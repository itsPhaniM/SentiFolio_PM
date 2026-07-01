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


@app.get("/shap")
def shap() -> dict:
    """Mean |SHAP| feature importances (which features drive the forecasts)."""
    return inference.shap_importances()


@app.get("/backtest")
def backtest() -> dict:
    """Saved walk-forward backtest metrics for every strategy."""
    return inference.backtest_metrics()
