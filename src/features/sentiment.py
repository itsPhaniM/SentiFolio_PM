"""Score news headlines with FinBERT (ProsusAI/finbert).

GPU-accelerated and resumable. Reads data/raw/news_history.parquet by default and writes
data/processed/news_history_scored.parquet, adding per-headline sentiment:
  sent_pos, sent_neg, sent_neu  - softmax probabilities
  sent_score                    - continuous score in [-1, 1]  (P(pos) - P(neg))
  sent_label                    - argmax label (positive/negative/neutral)

Run from the project root:
    .venv/Scripts/python.exe -m src.features.sentiment

Score the live-news file instead:
    SENT_INPUT=news.parquet  ... -m src.features.sentiment
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from config import RAW_DIR, PROCESSED_DIR

MODEL = "ProsusAI/finbert"
IN_NAME = os.getenv("SENT_INPUT", "news_history.parquet")
BATCH = int(os.getenv("SENT_BATCH", "256"))
MAX_LEN = 64               # headlines are short
CHECKPOINT_BATCHES = 20    # write progress every ~20 batches


def score_batch(texts, tok, model, device, idx, id2label):
    enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
    pos_i, neg_i, neu_i = idx
    out = []
    for p in probs:
        out.append({
            "sent_pos": float(p[pos_i]), "sent_neg": float(p[neg_i]), "sent_neu": float(p[neu_i]),
            "sent_score": float(p[pos_i] - p[neg_i]),
            "sent_label": id2label[int(p.argmax())].lower(),
        })
    return out


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    in_path = RAW_DIR / IN_NAME
    out_path = PROCESSED_DIR / in_path.name.replace(".parquet", "_scored.parquet")

    df = pd.read_parquet(in_path)
    df = df.dropna(subset=["headline"])
    df = df[df["headline"] != ""].sort_values(["ticker", "datetime", "headline"]).reset_index(drop=True)

    scored_so_far = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    done = len(scored_so_far)
    print(f"Input: {len(df):,} headlines | already scored: {done:,} | model: {MODEL}")
    if done >= len(df):
        print("Nothing to do - already complete.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()
    id2label = model.config.id2label
    lab2idx = {v.lower(): k for k, v in id2label.items()}
    idx = (lab2idx["positive"], lab2idx["negative"], lab2idx["neutral"])

    remaining = df.iloc[done:].reset_index(drop=True)
    new_chunks = []
    n = len(remaining)
    for b, i in enumerate(range(0, n, BATCH)):
        batch_df = remaining.iloc[i:i + BATCH].reset_index(drop=True)
        sc = pd.DataFrame(score_batch(batch_df["headline"].tolist(), tok, model, device, idx, id2label))
        new_chunks.append(pd.concat([batch_df, sc], axis=1))
        if b % CHECKPOINT_BATCHES == 0 or i + BATCH >= n:
            combined = pd.concat([scored_so_far] + new_chunks, ignore_index=True)
            combined.to_parquet(out_path, index=False)
            print(f"  scored {min(i + BATCH, n):,}/{n:,} new ({done + min(i + BATCH, n):,} total)")

    final = pd.read_parquet(out_path)
    print(f"\nDONE. {len(final):,} headlines scored -> {out_path}")
    print(final["sent_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
