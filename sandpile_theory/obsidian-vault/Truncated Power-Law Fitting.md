---
tags: [method, statistics]
---

# Truncated Power-Law Fitting

How avalanche-size tails are actually fit in this project, using the Python
`powerlaw` package's implementation of Clauset, Shalizi & Newman (2009) —
see [[References]].

## Why not just fit a power law?

A *finite* lattice always cuts the avalanche tail off at some `s_max` set by
`L`. An **untruncated** power law loses to a lognormal in a goodness-of-fit
test on literally every drive rule tested here — but that's expected, not a
failure of [[Self-Organized Criticality]]: nothing in the pure power-law
family can express a cutoff a finite system must have.

> [!important] The honest test
> **Truncated power law vs. lognormal** — both families can express a
> cutoff, so the comparison is fair. `P(s) ~ s^-τ · e^(-s/s_max)` beating
> lognormal (log-likelihood ratio R > 0, p < 0.05) is the real SOC
> signature on a finite grid.

## Result across every drive rule

Truncated power law beats lognormal decisively everywhere tested — R
ranging 21–36, p < 10⁻¹⁰⁰ — on [[Drive Rule - Uniform]],
[[Drive Rule - Avoid Crowded]], [[Drive Rule - Matthew Effect]],
[[Drive Rule - Frontier]], and every point of the
[[Novelty-Seeking Phase Transition|novelty sweep]]. What changes across
rules and across the transition isn't *whether* the field is critical in
this narrow statistical sense — it's the value of τ itself, by a large and
philosophically interesting margin.

This correction was applied after an earlier by-hand MLE/log-bin-slope
estimate overstated how directly "power law beats lognormal" could be
claimed — kept in the project history as a methodological note, not hidden.
