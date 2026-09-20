# A sandpile model of theory development

An Agents.jl implementation of the Bak–Tang–Wiesenfeld (BTW) sandpile, re-read
so that **cells are theories and grains are scientists**, plus the machinery to
check whether the result is actually critical.

```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. run_demo.jl            # L=32, ~1–2 min
julia --project=. run_demo.jl 16 5000 20000   # smaller/faster

pip install numpy pandas matplotlib powerlaw
python3 analyze.py                       # ANALYSIS.md, figures/, results_*.csv
python3 make_video.py :novelty 0.7       # theory_field.gif — the field evolving over time
```

Files:

| file | what it is |
|---|---|
| `TheorySandpile.jl` | the model + the analysis functions, all five drive rules (`:uniform`, `:avoid_crowded`, `:matthew`, `:frontier`, `:novelty`) |
| `run_demo.jl` | control run, the drive-rule comparison, the novelty_weight sweep, finite-size scaling |
| `validate_reference.py` | the same rules in dependency-free Python, used as a cross-check — also what `analyze.py` and `make_video.py` run against |
| `analyze.py` | rigorous statistical analysis (power-law / truncated-power-law / lognormal fits via the `powerlaw` package) → `ANALYSIS.md`, `figures/*.png`, `results_*.csv` |
| `make_video.py` | renders an animated GIF of the field evolving step by step |
| `ANALYSIS.md` | the generated report — read this for the numbers, not the summary below |
| `NOVELTY_RESULTS.md` | the original by-hand novelty-seeking writeup; superseded in rigor by `ANALYSIS.md` but kept for the narrative |

---

## 1. Self-organized criticality, in one page

A **critical point** is normally something you have to *tune* to. Heat a magnet
and it is ordered below `T_c` and disordered above; exactly at `T_c` — and only
there — correlations become scale-free, fluctuations occur at every size, and
the response to a small perturbation has no characteristic scale. That is a
measure-zero point in parameter space. If criticality explained anything in
nature you would have to explain who keeps turning the knob.

**Self-organized criticality** is the claim that a class of slowly driven,
dissipative, spatially extended systems *walks to that point by itself* and
stays there, with no tuning. The critical state is an **attractor of the
dynamics** rather than a special value of a parameter.

The mechanism, stated generally:

1. **Slow drive.** Energy/stress/grains are added slowly.
2. **A threshold.** Nothing happens locally until a local variable exceeds
   `z_c`; then it relaxes and pushes the excess to its neighbours.
3. **Fast relaxation.** The cascade finishes before the next input arrives —
   a *separation of timescales*. This is what makes "avalanche" a well-defined
   object at all.
4. **Bulk conservation + boundary dissipation.** Inside, relaxation only moves
   stuff around; it leaves only at the edges. Stationarity then requires
   inflow = outflow, which forces the system to a state where a typical input
   travels the whole system size to escape — and that is the critical state.

The negative feedback is the point: below critical density, input piles up
faster than it leaves (density rises); above it, avalanches are so easy to
trigger that too much leaves (density falls). The fixed point is exactly
marginal stability.

The **observable signature** is a power law with no characteristic scale:

```
P(s) ~ s^(-τ) · G(s / L^D)
```

avalanche sizes distributed as a power law, cut off only by the system size
`L`. The cutoff must *move with `L`* — a power law measured at a single system
size proves very little, which is why `run_demo.jl` does finite-size scaling.

**A caution worth internalising:** SOC is a much narrower claim than "I found a
power law." Power laws come out of lognormals, of preferential attachment, of
random multiplicative processes, of bad binning, and of plotting too little
data on log axes. Clauset–Shalizi–Newman (2009) is the standard antidote.

---

## 2. The BTW sandpile

On an `L × L` grid each cell holds `z(i,j)` grains.

- **Drive:** add one grain at a random cell.
- **Topple:** any cell with `z ≥ 4` gives one grain to each of its four
  neighbours: `z → z−4`, `z(neighbour) → z(neighbour)+1`.
- **Dissipate:** grains pushed off the boundary are gone.
- Repeat toppling until every cell is stable; only then add the next grain.

That is the whole model. It was introduced in Bak, Tang & Wiesenfeld (1987) as
an explanation of `1/f` noise, and it is the canonical SOC model.

### Known patterns / stylized facts

These are the things the sandpile is actually famous for, and the checklist to
run any variant against:

1. **Self-organization to a critical slope.** From any initial condition the
   density converges to a stationary value with no tuning. For 2D BTW the exact
   stationary mean height is `17/8 = 2.125`. Reproducing this number is the
   cheapest correctness test there is.
2. **Power-law avalanche statistics** in size `s` (number of topplings),
   duration `T`, area (distinct sites), and radius, each cut off at the system
   size, with finite-size-scaling collapse.
3. **Punctuated equilibrium.** Long quiet stretches broken by bursts spanning
   all scales. Most grains do nothing; occasionally one grain moves the whole
   system. The trigger's size tells you nothing about the event's size.
4. **The Abelian property (Dhar 1990).** The final stable configuration and the
   number of topplings at each site are *independent of the order* in which you
   relax unstable sites. Only the parallel-update duration depends on ordering.
   This makes the model exactly solvable in a way almost no other SOC model is.
5. **Exact combinatorics.** The recurrent configurations form an abelian group;
   their number equals the number of spanning trees of the graph (matrix–tree
   theorem). This is the deepest thing in the subject and connects sandpiles to
   loop-erased random walks, the Tutte polynomial, and rotor-routing.
6. **Conservation matters.** Bulk conservation plus boundary dissipation is the
   standard recipe. Non-conservative variants (OFC earthquake model) are much
   more delicate and their criticality is still argued about.
7. **Deterministic fractal patterns.** Drop `N` grains on one site of an empty
   infinite lattice and relax: you get a strikingly self-similar pattern with a
   proven scaling limit (Pegden–Smart). Different phenomenon from the avalanche
   statistics; often what people have seen in pictures.
8. **`1/f` noise — the claim that did not survive.** BTW's motivating claim was
   `1/f` power spectra; Jensen, Christensen & Fogedby (1989) showed the sandpile
   actually gives `1/f²`. Worth knowing, because it is the most-cited *wrong*
   part of the original paper.
9. **Multiscaling in 2D BTW.** The deterministic 2D sandpile does **not** obey
   clean simple finite-size scaling; naive fits give `τ ≈ 1.2`, the "waves of
   toppling" decomposition gives an exponent of exactly 1 for waves, and
   published values disagree because the underlying assumption differs. The
   *stochastic* Manna model is much better behaved (`τ ≈ 1.27`, `D ≈ 2.75`) and
   defines its own universality class. Mean-field / branching-process values
   (exact above `d_c = 4`) are `τ = 3/2`, `α = 2`.
10. **Universality classes exist and are narrow.** BTW, Manna, and Oslo are not
    the same model. Real rice piles are only critical for *elongated* grains
    (Frette et al. 1996) — round grains give characteristic-size events. This is
    the empirical lesson: mechanism details decide whether you get SOC at all.
11. **Claimed real-world instances.** Earthquakes (Gutenberg–Richter), solar
    flares, forest fires, neuronal avalanches (Beggs & Plenz 2003 — `τ = 3/2`,
    mean field), rainfall, extinction records. All contested to different
    degrees.

---

## 3. The mapping to theory space

| sandpile | this model |
|---|---|
| cell | a theory |
| lattice edge | two theories close enough that work on one carries into the other |
| lattice distance | conceptual distance; a far theory is reachable only through the intervening ones |
| grain | a scientist working on that theory |
| `z_c = 4` | how many scientists a theory can absorb before it is worked out |
| toppling | the theory saturates and its late arrivals spill into adjacent / auxiliary / bridging theories |
| avalanche | a research cascade — one extra worker sets off a chain of theories being taken up in sequence |
| boundary dissipation | scientists leaving the field |
| critical slope | the field's stationary crowding level |

Two additions the sand does not have:

- **`visits[pos]`** — how many people have *ever* worked on a theory.
  Monotone: theory development is irreversible in a way that sand height is not.
  "Developed" = `visits > 0`.
- **Priority credit** — the first `credit_slots` arrivals at a theory bank
  `credit_decay^k` each; everyone after them banks nothing. Credit goes to
  whoever gets there first, and often only to the first. This is what makes a
  crowded theory unattractive without any coordination.

Which scientists move when a theory topples: the **latest arrivals**. The early
ones have the claim on it and stay; the latecomers, who would get nothing, move
to adjacent theories. That is exactly BTW's arithmetic (four out of the cell)
with a reason attached to *which* four.

### The four entry rules

The interesting knob is how a *new* scientist picks a topic — the thing your
question was really about. `drive =`

- `:uniform` — anywhere. This is literally BTW, and it is the control.
- `:avoid_crowded` — compares `sample` theories and takes the least worked one.
  Your "if it is a low-hanging fruit because plenty of people are on it, a new
  scientist is motivated to work on something new."
- `:matthew` — the opposite. Joins the most crowded of `sample` theories. Hot
  topics attract people; success breeds success (Merton's Matthew effect).
- `:frontier` — looks for an undeveloped theory adjacent to a developed one.
  Works the edge of the known — the "adjacent possible".

---

## 4. What comes out

Numbers below are from `validate_reference.py` (identical rules, `L = 24`,
20k warm-up + 60k measured steps). **Julia could not be installed in the
container these were produced in — the egress policy blocks the Julia binary
host — so the Julia code in this directory has been reviewed but not
executed. Run `run_demo.jl` and check the control block first.**

```
drive=:uniform         density 2.073   activity 0.42   <s>  56   max s 1034   τ_fit 1.02
drive=:avoid_crowded   density 2.352   activity 0.055  <s> 270   max s 2233   τ_fit 1.51
drive=:matthew         density 2.028   activity 0.92   <s>  34   max s  705   τ_fit 1.00
drive=:frontier        density 2.076   activity 0.42   <s>  56   max s 1094   τ_fit 1.00
```

Reading them:

- **`:uniform` reproduces BTW.** Density 2.073 against the exact `2.125` (the
  gap is finite-size: `L = 24` with open boundaries), and a clean power law over
  two and a half decades. The credit bookkeeping rides along without disturbing
  the sandpile — which is the point of keeping `:uniform` as a control.
- **`:matthew` suppresses large cascades.** Activity goes to 0.92 — almost every
  new scientist triggers *something* — but the mean and the maximum event both
  shrink. Piling onto whatever is already hot keeps burning the pile down
  locally, so it never accumulates the stress a system-spanning cascade needs.
  Rich-get-richer makes science *busier and less consequential*.
- **`:avoid_crowded` breaks scale invariance.** Only 5.5% of steps do anything,
  but when something happens it is huge (`<s> = 270`, five times `:uniform`),
  and the binned distribution develops a **bump at large `s`** instead of a
  clean cutoff. That bump is a *characteristic scale*: the system loads evenly,
  stays subcritical for a long time, then sweeps. Perfectly rational
  novelty-seeking pushes the field **away from SOC** toward quasi-periodic
  system-wide upheaval. (Suggestive, not settled — only ~2.4k events survive the
  cut. Rerun with more steps before believing it.)
- **Finite-size scaling holds.** Going `L = 12 → 24`: `<s>` goes `16.5 → 56.4`
  (ratio 3.4, against `L²`'s 4 — the exact BTW result is `<s> ∝ L²`, since in
  the stationary state every added grain must random-walk to the boundary),
  `max s` goes `219 → 1034`, and `τ_fit` stays at `0.99 → 1.02`. A fixed
  exponent with a cutoff that moves with `L` is the actual evidence of
  criticality; the single-`L` power law is not.
- **`:frontier` degenerates.** On a finite lattice every theory eventually gets
  worked (`576/576`), so after the transient there is no frontier and the rule
  falls back to uniform. This is a modelling artifact and the clearest thing to
  fix: theory space should *grow*, not saturate.

The one prediction worth stating in advance: **make the theory graph
small-world** (rewire a few edges so distant theories become adjacent) and the
exponent should move toward the mean-field value `τ = 3/2`, because the
avalanche becomes a branching process on a locally tree-like graph. The
"bridging theories are required" assumption is not decoration — it is the thing
that puts the model in a non-mean-field universality class.

---

## 5. Is this a legitimate model of science?

Honestly: it is a legitimate *toy*, in the same sense BTW is a toy of
earthquakes. What it can support is a conditional claim — "if theory space is
locally connected and credit is winner-take-all, then research effort
self-organizes to a marginal state and progress arrives in scale-free bursts."
It cannot support "science is critical" without data.

**What carries over well**

- Slow drive / fast relaxation is real: people enter the field slowly compared
  to how fast a hot result reshuffles who works on what.
- Local conservation is defensible: over the short run the workforce is roughly
  fixed and redistributes.
- Thresholded local capacity from the priority rule is the strongest part of the
  analogy — Kitcher's and Strevens's work is precisely about the priority rule
  producing a division of cognitive labor without central planning.
- "Punctuated equilibrium of paradigms" is exactly Kuhn's picture, and SOC gives
  it a mechanism rather than a metaphor.

**What breaks, in descending order of severity**

1. **The lattice.** Theory space is not a 2D grid; it is a growing,
   heterogeneous, high-dimensional network with hubs. Dimension controls the
   exponents, so the grid quietly fixes the answer. *Fix: swap `GridSpace` for
   `GraphSpace` over a real citation/concept network.*
2. **No growth.** Real theory space expands — new theories are created,
   frequently by combining existing ones (the adjacent possible, recombinant
   growth). Here the space is fixed and saturates, which is what kills the
   `:frontier` rule. *Fix: add nodes at the frontier when a theory topples.*
3. **Conservation of scientists.** The scientific workforce has grown roughly
   exponentially for a century. A conservative model of a non-conservative
   system is a real problem, since conservation is one of the load-bearing
   ingredients for SOC.
4. **All theories are interchangeable.** No difficulty, no fertility, no
   truth-value, no wrongness. Nothing can be abandoned. A theory here is a slot,
   not a claim.
5. **Toppling is not what scientists do.** A saturated theory does not eject its
   workers into the four topics nearest it; people choose, with foresight and
   incomplete information.
6. **Credit is not a fixed number of slots.** Citations are famously
   heavy-tailed and path-dependent (Matthew effect), not a decaying sequence of
   `credit_slots` prizes.

**How I would actually test it.** The observable this model predicts is a
scale-free distribution of *research cascade sizes*. That is measurable: take a
concept/citation network, define an event as a burst of new entrants into a
concept and its neighbours, and check whether burst size is power-law with a
cutoff that grows with the field's size. If bursts have a characteristic
scale, the `:avoid_crowded` regime is the better description and the SOC story
is wrong.

---

## 6. References

**Sandpiles and SOC — the core**

- P. Bak, C. Tang, K. Wiesenfeld, *Self-organized criticality: an explanation of
  1/f noise*, Phys. Rev. Lett. **59**, 381 (1987). The original.
- P. Bak, C. Tang, K. Wiesenfeld, *Self-organized criticality*, Phys. Rev. A
  **38**, 364 (1988). The long version; read this one.
- D. Dhar, *Self-organized critical state of sandpile automaton models*, Phys.
  Rev. Lett. **64**, 1613 (1990). Abelian property, group structure, exact
  results. The single most important follow-up.
- D. Dhar, *Theoretical studies of self-organized criticality*, Physica A
  **369**, 29 (2006). Best review of the exact theory.
- H. J. Jensen, *Self-Organized Criticality*, Cambridge (1998). Compact and
  readable.
- G. Pruessner, *Self-Organised Criticality: Theory, Models and
  Characterisation*, Cambridge (2012). The reference work, and the right place
  for how to measure exponents without fooling yourself.
- K. Christensen, N. Moloney, *Complexity and Criticality*, Imperial College
  Press (2005). Best textbook if you want to work the problems.
- P. Bak, *How Nature Works* (1996). The popular account — read it for the
  program, not for the evidence.

**Essential corrections and neighbouring models**

- H. J. Jensen, K. Christensen, H. C. Fogedby, *1/f noise, distribution of
  lifetimes, and a pile of sand*, Phys. Rev. B **40**, 7425 (1989). The sandpile
  gives 1/f², not 1/f.
- S. S. Manna, *Two-state model of self-organized criticality*, J. Phys. A
  **24**, L363 (1991). The stochastic sandpile; its own universality class.
- K. Christensen, Á. Corral, V. Frette, J. Feder, T. Jøssang, *Tracer dispersion
  in a self-organized critical system*, Phys. Rev. Lett. **77**, 107 (1996). The
  Oslo ricepile model.
- V. Frette et al., *Avalanche dynamics in a pile of rice*, Nature **379**, 49
  (1996). SOC only for elongated grains — the experiment that shows how narrow
  the phenomenon is.
- Z. Olami, H. J. S. Feder, K. Christensen, PRL **68**, 1244 (1992). OFC
  earthquake model; non-conservative.
- B. Drossel, F. Schwabl, PRL **69**, 1629 (1992). Forest-fire model.
- P. Bak, K. Sneppen, PRL **71**, 4083 (1993). Punctuated equilibrium in
  evolution — the closest SOC model in spirit to "theories competing".
- J. M. Beggs, D. Plenz, *Neuronal avalanches in neocortical circuits*,
  J. Neurosci. **23**, 11167 (2003). The best-supported empirical case.
- W. Pegden, C. Smart, *Convergence of the Abelian sandpile*, Duke Math. J.
  **162**, 627 (2013). The fractal patterns, made rigorous.

**Before you claim a power law**

- A. Clauset, C. R. Shalizi, M. E. J. Newman, *Power-law distributions in
  empirical data*, SIAM Review **51**, 661 (2009). Non-negotiable.
- R. Frigg, *Self-organised criticality — what it is and what it isn't*, Stud.
  Hist. Phil. Sci. **34**, 613 (2003). The philosophical critique.
- N. W. Watkins, G. Pruessner, S. C. Chapman, N. B. Crosby, H. J. Jensen,
  *25 years of self-organized criticality: concepts and controversies*, Space
  Sci. Rev. **198**, 3 (2016). What survived.

**Science as the thing being modelled**

- T. S. Kuhn, *The Structure of Scientific Revolutions* (1962). The punctuated
  picture you are formalising.
- R. K. Merton, *Priorities in scientific discovery*, Am. Sociol. Rev. **22**,
  635 (1957); *The Matthew effect in science*, Science **159**, 56 (1968). The
  credit assumptions in your model, stated by the person who found them.
- P. Kitcher, *The division of cognitive labor*, J. Philosophy **87**, 5 (1990).
  The canonical formal treatment of scientists distributing across research
  programs under credit incentives. **Start here.**
- M. Strevens, *The role of the priority rule in science*, J. Philosophy **100**,
  55 (2003). Why winner-take-all credit is (arguably) epistemically efficient.
- M. Weisberg, R. Muldoon, *Epistemic landscapes and the division of cognitive
  labor*, Phil. Sci. **76**, 225 (2009). An ABM on a topic landscape — the
  closest existing model to yours. Read with its critique: J. M. Alexander,
  J. Himmelreich, C. Thompson, Phil. Sci. **82**, 424 (2015).
- K. Zollman, *The epistemic benefit of transient diversity*, Erkenntnis **72**,
  17 (2010). Why crowding onto the current best option is bad for the group.
- S. Fortunato et al., *Science of science*, Science **359**, eaao0185 (2018).
  The empirical field; your source of data.
- J. G. Foster, A. Rzhetsky, J. A. Evans, *Tradition and innovation in
  scientists' research strategies*, Am. Sociol. Rev. **80**, 875 (2015).
  Measures the explore/exploit tradeoff your `:avoid_crowded` rule encodes, and
  finds scientists are far more conservative than the reward structure implies.
- F. Tria, V. Loreto, V. D. P. Servedio, S. H. Strogatz, *The dynamics of
  correlated novelties*, Sci. Rep. **4**, 5890 (2014). The adjacent possible,
  formalised — the right way to make your theory space grow.
- G. Iacopini, S. Milojević, V. Latora, *Network dynamics of innovation
  processes*, Phys. Rev. Lett. **120**, 048301 (2018). Discovery as a walk on a
  growing network of ideas. This is the model to merge with yours.
- J. S. G. Chu, J. A. Evans, *Slowed canonical progress in large fields of
  science*, PNAS **118**, e2021636118 (2021). Empirical: crowding a field
  entrenches it rather than advancing it — directly relevant to `:matthew`.
- M. Park, E. Leahey, R. J. Funk, *Papers and patents are becoming less
  disruptive over time*, Nature **613**, 138 (2023).

**Agents.jl**

- G. Datseris, A. R. Vahdati, T. C. DuBois, *Agents.jl: a performant and
  feature-full agent-based modeling software of minimal code complexity*,
  SIMULATION **100**, 1019 (2024). Docs: <https://juliadynamics.github.io/Agents.jl/>
