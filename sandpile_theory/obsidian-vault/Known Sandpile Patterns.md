---
tags: [concept, soc, physics, reference]
---

# Known Sandpile Patterns

The stylized facts [[Bak-Tang-Wiesenfeld Sandpile|BTW]] is actually famous for —
the checklist any variant, including [[Theory Sandpile]] itself, should be run
against.

1. **Self-organization to a critical slope.** From any initial condition the
   density converges to a stationary value with no tuning. For 2D BTW the
   exact stationary mean height is **17/8 = 2.125**. Reproducing this number
   is the cheapest correctness test there is — see
   [[Drive Rule - Uniform]]'s 2.085 on a finite L=24 grid.
2. **Power-law avalanche statistics** in size, duration, area, and radius
   (see [[Avalanche Features]]), each cut off at the system size, with
   [[Finite-Size Scaling|finite-size scaling]] collapse.
3. **Punctuated equilibrium.** Long quiet stretches broken by bursts spanning
   all scales — most grains do nothing; occasionally one grain moves the
   whole system. See [[Kuhns Punctuated Structure]] for the philosophical
   reading of this.
4. **The Abelian property** (Dhar, 1990). The final stable configuration and
   the number of topplings at each site are independent of the order in
   which unstable sites are relaxed — only the parallel-update duration
   depends on ordering. Covered in [[Bak-Tang-Wiesenfeld Sandpile]].
5. **Exact combinatorics.** The recurrent configurations form an abelian
   group; their number equals the number of spanning trees of the graph
   (matrix–tree theorem). Connects sandpiles to loop-erased random walks,
   the Tutte polynomial, and rotor-routing — the deepest result in the
   subject, and not something this project's simulation touches directly.
6. **Conservation matters.** Bulk conservation plus boundary dissipation
   (see [[Self-Organized Criticality]]'s four ingredients) is the standard
   recipe. Non-conservative variants, like the Olami–Feder–Christensen
   earthquake model, are far more delicate and their criticality is still
   argued about.
7. **Deterministic fractal patterns.** Drop N grains on one site of an empty
   infinite lattice and relax: a strikingly self-similar pattern with a
   proven scaling limit (Pegden–Smart). A different phenomenon from the
   avalanche statistics this project measures — often what people have
   actually seen in BTW pictures online.
8. **1/f noise — the claim that did not survive.** BTW's original motivating
   claim was 1/f power spectra; Jensen, Christensen & Fogedby (1989) showed
   the sandpile actually gives 1/f². The most-cited wrong part of the
   original 1987 paper. See [[References]].
9. **Multiscaling in 2D BTW.** The deterministic 2D sandpile does not obey
   clean simple finite-size scaling; naive fits give τ ≈ 1.2, while the
   "waves of toppling" decomposition gives an exponent of exactly 1 for
   waves. Published values disagree because the underlying assumption
   differs — directly relevant to how [[Truncated Power-Law Fitting|this
   project fits τ]], and why that note insists on checking a lognormal
   alternative rather than trusting one estimator.
10. **Universality classes exist and are narrow.** BTW, the stochastic Manna
    model (τ ≈ 1.27), and the Oslo ricepile model are not the same model.
    Real rice piles are only critical for *elongated* grains (Frette et al.,
    1996) — round grains give characteristic-size events. The empirical
    lesson: mechanism details decide whether you get SOC at all, which is
    exactly what [[Novelty-Seeking Phase Transition]] finds for this
    project's own mechanism.
11. **Claimed real-world instances.** Earthquakes (Gutenberg–Richter),
    solar flares, forest fires, neuronal avalanches (Beggs & Plenz, 2003 —
    τ = 3/2, mean field), rainfall, extinction records. All contested to
    different degrees — see [[Philosophy of Science - Overview#Where it has nothing to say]]
    for the same caution applied to this project's own science-of-science
    claim.

> [!warning] A power law is not proof
> Pattern 9 and the honest result in [[Truncated Power-Law Fitting]] are the
> same caution twice: a visually power-law-shaped plot is not SOC by itself.
> Clauset, Shalizi & Newman (2009) — [[References]] — is the fix in both
> places.

See also [[Self-Organized Criticality]] for the mechanism behind all eleven,
and [[Theory-Scientist Mapping]] for which of these this project actually
checked (1, 2, 4, 9) versus only inherited by citation (3, 5, 6, 7, 8, 10, 11).
