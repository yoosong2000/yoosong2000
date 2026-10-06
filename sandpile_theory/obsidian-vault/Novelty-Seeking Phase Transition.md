---
tags: [finding, soc]
---

# Novelty-Seeking Phase Transition

The project's central result: sweeping [[Drive Rule - Novelty-Seeking|novelty_weight]]
from 0 to 1 produces a **sharp step**, not a slope, in the critical exponent.

## Setup
11 runs, grid size L=24, 40,000 measured steps each after a 15,000-step
warm-up discarded as transient. Each point fit with a
[[Truncated Power-Law Fitting|truncated power law]] against a lognormal
alternative.

## The numbers

| novelty_weight | τ (truncated power-law) | mean ⟨s⟩ | activity |
|---|---|---|---|
| 0.0 | 0.918 | 269.8 | 0.055 |
| 0.1 | 0.884 | 282.9 | 0.053 |
| 0.2 | 0.890 | 280.8 | 0.053 |
| 0.3 | 0.934 | 265.1 | 0.055 |
| 0.4 | 0.914 | 281.4 | 0.053 |
| **0.5** | **0.497** | **705.1** | **0.031** |
| 0.6 | 0.517 | 670.9 | 0.032 |
| 0.7 | 0.478 | 686.9 | 0.031 |
| 0.8 | 0.570 | 665.3 | 0.032 |
| 0.9 | 0.519 | 687.3 | 0.031 |
| 1.0 | 0.548 | 663.1 | 0.033 |

Below the transition: **τ = 0.91 ± 0.02**. At or above it: **τ = 0.52 ± 0.03**
— a roughly 2.5× jump in mean avalanche size, a near-halving of activity,
right at `novelty_weight = 0.5`. Truncated power law beats lognormal
decisively on both sides (R > 20, p < 10⁻¹⁰⁰), so this is not "critical vs.
not critical" crudely — it's a large, real shift in *how* critical, hinging
on a single incentive.

> [!quote] Reading the result
> A population of scientists individually and rationally avoiding crowded
> theories ([[Drive Rule - Avoid Crowded]]) does not produce smooth,
> scale-free research cascades. It produces rarer, more extreme ones. What
> pulls the field back toward scale-free behavior is specifically the
> preference for *novel* theories, not mere crowd-avoidance — two things
> that sound alike and do opposite things to the field.

See [[Social Epistemology of Credit]] and [[Aggregation Failure]] for why
this is a philosophically loaded result, not just a modeling curiosity, and
[[Research Questions]] for what would need to be true of real research
bursts to test it empirically.
