"""Streamlit dashboard for SentiFolio: portfolio, forecasts, SHAP and backtest.

Run locally:
    .venv/Scripts/python.exe -m streamlit run src/serve/dashboard.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from src.serve import inference
from config import ROOT as PROJECT_ROOT

st.set_page_config(page_title="SentiFolio", layout="wide")
st.title("SentiFolio — Explainable, Sentiment-Aware FTSE Portfolios")
st.caption("LightGBM forecasts + SHAP explainability, walk-forward backtested with realistic costs.")

arm = st.sidebar.selectbox("Model arm", list(inference.ARMS_AVAILABLE))
st.sidebar.markdown(
    "**price_only** uses technical features; **price+sentiment** adds the FinBERT "
    "sentiment features. The ablation tests whether sentiment adds risk-adjusted value."
)

rec = inference.recommend_portfolio(arm)
st.subheader(f"Recommended portfolio — as of {rec['as_of']} ({rec['strategy']}, {rec['n_holdings']} holdings)")
st.dataframe(pd.DataFrame(rec["holdings"]), use_container_width=True, hide_index=True)

st.subheader(f"Return forecasts — {inference.HORIZON}-day horizon")
st.dataframe(pd.DataFrame(inference.latest_forecasts(arm)["forecasts"]),
             use_container_width=True, hide_index=True)

left, right = st.columns(2)
with left:
    st.subheader("SHAP feature importance")
    st.dataframe(pd.DataFrame(inference.shap_importances()["features"]),
                 use_container_width=True, hide_index=True)
    shap_png = PROJECT_ROOT / "reports" / "week4" / "shap_feature_importance.png"
    if shap_png.exists():
        st.image(str(shap_png))
with right:
    st.subheader("Walk-forward backtest")
    st.dataframe(pd.DataFrame(inference.backtest_metrics()["strategies"]),
                 use_container_width=True, hide_index=True)
    eq_png = PROJECT_ROOT / "reports" / "week5" / "equity_curves.png"
    if eq_png.exists():
        st.image(str(eq_png))
