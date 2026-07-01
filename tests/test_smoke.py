"""CI smoke tests: the serving layer imports and the config is sane.

These deliberately avoid loading data/model artifacts (gitignored, absent in CI) —
they check that the package wires together and the FastAPI app builds its routes.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def test_config_universe():
    from config import TICKERS
    assert len(TICKERS) == 15
    assert all(t.endswith(".L") for t in TICKERS)


def test_api_imports_and_routes():
    from src.serve.api import app
    paths = {r.path for r in app.routes}
    assert {"/health", "/forecasts", "/portfolio", "/shap", "/backtest"} <= paths


def test_inference_arms():
    from src.serve import inference
    assert "price_only" in inference.ARMS_AVAILABLE
    assert "price+sentiment" in inference.ARMS_AVAILABLE
