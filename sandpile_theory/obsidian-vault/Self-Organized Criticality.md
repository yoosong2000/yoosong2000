---
tags: [concept, soc, physics]
---

# Self-Organized Criticality

An ordinary critical point has to be tuned to — a magnet is scale-free only
at exactly its critical temperature, a single point in parameter space.
**Self-organized criticality** (Bak, Tang & Wiesenfeld, 1987) is the claim
that some systems reach that same scale-free state *without* tuning, because
the critical state is where the dynamics settle on their own.

## The four ingredients
1. **Slow drive** — stress added gradually, one unit at a time.
2. **A threshold** — nothing happens locally until a cell holds too much,
   then it relaxes and pushes the excess outward.
3. **Fast relaxation** — the triggered cascade finishes completely before
   the next input arrives. This separation of timescales is what makes "an
   avalanche" a well-defined, countable thing.
4. **Conservation with a boundary leak** — nothing is created or destroyed
   inside the system; it only ever leaves at the edges.

That last ingredient does the real work: below critical density, input piles
up faster than it leaves, so density *rises*; above it, avalanches become so
easy to trigger that too much leaves at once, so density *falls*. The fixed
point of that feedback is exact marginal stability.

## The signature

```
P(s) ~ s^-τ · G(s / L^D)
```

Avalanche sizes distributed across every scale up to the system size `L`,
with a cutoff that moves with `L`. A power law measured at one grid size
alone proves little — see [[Finite-Size Scaling]] for this project's own
check of that claim, [[Truncated Power-Law Fitting]] for how τ itself is
fit properly, and [[Novelty-Seeking Phase Transition]] for where that check
actually mattered.

> [!warning] Not every power law is SOC
> Lognormal distributions, preferential attachment, and plain bad binning
> all produce something that looks power-law-shaped on a bad plot. Clauset,
> Shalizi & Newman (2009) — see [[References]] — is the standard fix.

Applied in: [[Bak-Tang-Wiesenfeld Sandpile]], [[Theory-Scientist Mapping]]
