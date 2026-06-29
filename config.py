"""Central configuration for the SentiFolio project."""
from __future__ import annotations
from pathlib import Path

# ---- Paths ----
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# ---- FTSE universe ----
# Yahoo Finance tickers use the ".L" suffix for the London Stock Exchange.
# ~15 large, liquid, well-discussed FTSE 100 names across diverse sectors
# (chosen so there is enough news/social-media chatter for the sentiment side).
TICKERS = {
    "BP.L": "BP",
    "SHEL.L": "Shell",
    "HSBA.L": "HSBC",
    "BARC.L": "Barclays",
    "LLOY.L": "Lloyds Banking Group",
    "NWG.L": "NatWest",
    "VOD.L": "Vodafone",
    "GSK.L": "GSK",
    "AZN.L": "AstraZeneca",
    "ULVR.L": "Unilever",
    "RIO.L": "Rio Tinto",
    "GLEN.L": "Glencore",
    "TSCO.L": "Tesco",
    "RR.L": "Rolls-Royce",
    "BA.L": "BAE Systems",
}

# ---- Backtest window ----
START_DATE = "2018-01-01"
END_DATE = None  # None = up to today
