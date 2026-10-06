---
tags: [practical, method]
---

# Tooling - Python vs R vs Stata

Checked directly rather than assumed, for the statistical analysis step
(fitting [[Truncated Power-Law Fitting|truncated power laws]], producing the
figures behind [[Novelty-Seeking Phase Transition]]).

| | Python | R | Stata |
|---|---|---|---|
| power-law fitting | `powerlaw` — full CSN(2009) MLE + KS test | `poweRlaw` — equally rigorous | no native discrete power-law MLE |
| runs the cross-check sim | yes, already is | would need a rewrite | would need a rewrite |
| video output | `matplotlib.animation`, no extra install | heavier dependency chain | not a video tool |
| cost | free | free | commercial license |

> [!note] R is not technically worse
> R's `poweRlaw` package is just as rigorous an implementation of the same
> Clauset–Shalizi–Newman estimator. What decided it was the pipeline: the
> cross-check simulation ([[Running the Code]]) is already Python, so
> simulate → fit → plot → animate runs in one process, one language, nothing
> shelled out.

See [[References]] for the `powerlaw` package citation (Alstott, Bullmore &
Plenz, 2014).
