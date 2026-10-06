---
tags: [drive-rule, finding]
---

# Drive Rule - Novelty-Seeking

`drive = :novelty` — a continuous dial, `novelty_weight ∈ [0,1]`, from
crowd-avoidance behavior up to an active preference for undeveloped
theories, plus a small chance a saturating theory spawns new ones beside it.

Below `novelty_weight = 0.5` it behaves like [[Drive Rule - Avoid Crowded]];
at and above it, it actively prefers undeveloped theories. This single
parameter sweep produced the project's central finding — see
[[Novelty-Seeking Phase Transition]].

## Numbers (L=24, truncated power-law fit)

| novelty_weight | τ | mean ⟨s⟩ |
|---|---|---|
| 0.0 | 0.918 | 269.8 |
| 0.3 | 0.934 | 265.1 |
| 0.5 | 0.497 | 705.1 |
| 0.7 | 0.478 | 686.9 |
| 1.0 | 0.548 | 663.1 |

> [!abstract] The one-line result
> Below the transition, τ averages 0.91 ± 0.02; at or above it, τ drops to
> 0.52 ± 0.03, and mean avalanche size roughly 2.5×s. Full numbers and the
> statistical test in [[Novelty-Seeking Phase Transition]].

Philosophically: see [[Social Epistemology of Credit]] and
[[Aggregation Failure]] for what this step function is evidence *for*.
