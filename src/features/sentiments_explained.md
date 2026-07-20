# Sentiment Features Explained

Why every sentiment column in this project exists, and why it makes sense.

This file documents the columns produced by `sentiment.py` (per-headline scores) and the
sentiment half of `build_features.py` (daily per-stock features). The price/technical
features are documented in `build_features.py` itself.

---

## The problem that forces all of this

The model needs **one row per stock per trading day**, because that is how often a
portfolio decision is made. But on any given day one company might have six headlines
and another might have none.

So the headlines *have* to be compressed into a fixed set of numbers. That is not a
stylistic choice — it is the shape of the modelling table demanding it.

**Worked example.** Shell, 3 March, six headlines scored by FinBERT:

    +0.9   +0.2   -0.1   0.0   +0.4   -0.7

The model has one slot per column. What do you tell it about that day? There are only a
few sensible answers, and each one became a feature.

---

## Stage 1 — per-headline columns (`sentiment.py`)

FinBERT reads each headline and returns three probabilities that sum to 1.

### `sent_pos`, `sent_neg`, `sent_neu`
**What:** the model's probability that the headline is positive / negative / neutral.
**Why we kept all three:** they are the raw output. Keeping them means every later
decision (the score, the label) can be recomputed or revised without re-running FinBERT
over the whole corpus, which takes GPU time.
**Why it makes sense:** it is the cheapest possible insurance. Store the raw evidence,
derive everything else.

### `sent_score`  =  `P(positive) - P(negative)`
**What:** one signed number between -1 and +1.
**Why we picked it:** the model needs a single continuous value it can average, smooth
and compare. A three-way label cannot be averaged sensibly.
**Why it makes sense:** it preserves *confidence*, which the label throws away. A
headline the model is 51% sure is positive and one it is 96% sure is positive are very
different pieces of evidence, and this keeps that difference.
**Why `P(neu)` is dropped:** subtracting positive from negative already places a
confidently-neutral headline near zero, so it lands harmlessly in a daily average
instead of dragging it in either direction.

### `sent_label`
**What:** whichever of the three classes has the highest probability.
**Why we picked it:** it is human-readable, and it is what `sent_pos_ratio` counts.
**Why it makes sense:** useful for reporting and sanity checks (for example, the
roughly 67% neutral / 18% positive / 14% negative split), but deliberately *not* the
main modelling input, because it discards confidence.

> Implementation note: the code reads the class order from `model.config.id2label`
> rather than assuming it. FinBERT's output order is `0=positive, 1=negative,
> 2=neutral` — not alphabetical. Hardcoding the positions would silently swap classes
> and corrupt every downstream result.

---

## Stage 2 — daily per-stock features (`build_features.py`)

These answer "what happened in the news for this stock today?" Each one answers a
different question about that day's bag of scores.

### `sent_mean`
**What:** the plain average of the day's `sent_score` values. For the Shell example
above: `+0.12`.
**Why we picked it:** it is the most direct answer to *"what was the mood today?"* If
you compress a set of numbers into one, the average is the obvious first choice.
**Why it makes sense:** this is the feature that actually tests the project's core
hypothesis — that positive news precedes positive returns. Everything else is support.
**Honest note:** it is equal-weighted, so a Reuters scoop and a trade-magazine mention
count the same. This is the assumption the supervisor questioned, which is why
`scripts/sentiment_agg_check.py` re-tests the result under volume-weighted and
event-based schemes.

### `sent_vol`  (news volume — *not* volatility)
**What:** how many headlines the stock got that day. Shell example: `6`.
**Why we picked it:** the *amount* of attention may matter regardless of tone. A stock
suddenly generating twenty stories is in an unusual state even if the tone is neutral.
**Why it makes sense:** it tests a genuinely different hypothesis from `sent_mean`. If
sentiment helps through attention rather than direction, this is the feature that would
show it — and without it, that channel would go untested.
**Naming warning:** the `vol` here means *volume*, not volatility. `vol_20` (a price
feature) is volatility. Easy to confuse when reading SHAP output.

### `sent_disp`  (dispersion)
**What:** the standard deviation of the day's scores — how much the headlines disagreed.
Shell example: about `0.55`, i.e. quite mixed.
**Why we picked it:** a day where every story is mildly positive is not the same as a
day averaging to mildly positive because one story was glowing and another scathing.
The mean alone cannot tell those apart.
**Why it makes sense:** disagreement is a recognised proxy for uncertainty, and
uncertainty tends to precede larger price moves. Days with only one headline get `0`,
since a single value has no spread.

### `sent_pos_ratio`
**What:** the share of the day's headlines labelled positive. Shell example: `0.5`.
**Why we picked it:** it measures *breadth* rather than intensity — how many stories
were positive, not how strongly.
**Why it makes sense:** three mildly positive stories and one wildly positive story can
produce the same mean, but they are different situations. This separates them.

### `sent_mean_3d`, `sent_mean_7d`
**What:** `sent_mean` averaged over the last 3 and 7 days.
**Why we picked it:** this came from a problem observed in the data, not from theory.
Only around 64% of stock-days have any news at all, and when they do the median is under
four headlines. A single day's mean is therefore often missing or based on one story.
**Why it makes sense:** rolling windows fill the quiet days and stabilise a noisy
signal. It is also the more realistic hypothesis — market reaction to news builds over
days rather than resetting every midnight.
**Result note:** `sent_mean_3d` is consistently the strongest sentiment feature in the
SHAP ranking, which supports the smoothing decision.

### `sent_vol_7d`
**What:** total headline count over the last 7 days.
**Why we picked it:** the same smoothing logic applied to attention rather than tone.
**Why it makes sense:** it captures a sustained news cycle rather than a one-day spike,
which is the more plausible version of "this company is in the news right now."

### Days with no news
Filled with `0` for all four base features: neutral tone, zero volume, zero dispersion,
zero positive ratio.
**Why it makes sense:** "no news" genuinely is neutral. The rolling windows then carry
recent context across the gap.
**Honest limitation:** because prices start in 2018 but news only starts in 2020, and
coverage is around 64% overall, a meaningful share of rows feed the model a *default*
rather than a measured value. This is part of why the sentiment signal is weak, and it
is a limitation rather than a bug.

---

## Why seven sentiment features and not seventy

1. **Only 15 stocks.** The model ranks 15 names each day. That is a narrow
   cross-section; more features would fit noise rather than signal.
2. **The signal is weak** (rank IC around 0.037). The `deeper` configuration in
   `experiment.py` — more model capacity — produced the *worst* sentiment result in the
   grid, which is direct evidence that added complexity hurts here.
3. **SHAP has to stay readable.** Research question 2 asks which features drive the
   forecasts. Thirteen bars answer that; two hundred would not.
4. **Every feature must be defensible.** Each one above has a reason. A large
   auto-generated feature set would not.
5. **The ablation has to stay clean.** Six price features and seven sentiment features
   form two separable blocks, so the experiment can switch sentiment off without
   changing anything else.

The four base features are deliberately the standard way to summarise any group of
numbers — count, average, spread, proportion — so no obvious channel is left untested.
That coverage is what makes the eventual null result credible: sentiment was given a
fair chance through every reasonable route.

---

## Where judgment genuinely came in

Being honest about this is stronger than pretending everything was derived:

- **Window lengths (3 and 7 days)** are conventional round numbers, not proven optima.
- **Equal weighting** in `sent_mean` was an assumption — which is exactly why it was
  later tested against volume-weighted and event-based alternatives rather than
  defended.
- **The forecast horizon** was not assumed either; `experiment.py` sweeps 1, 5, 10 and
  20 days precisely so the choice is measured rather than guessed.

The principle: where a choice was arbitrary, it was tested rather than justified after
the fact.
