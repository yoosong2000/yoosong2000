---
tags: [concept, soc, model]
---

# Bak-Tang-Wiesenfeld Sandpile

The canonical [[Self-Organized Criticality]] model (1987). Unmodified rule:

1. A square grid, each cell holding some number of grains.
2. Add one grain to a random cell.
3. Any cell holding **four or more** grains sheds one grain to each of its
   four neighbors.
4. Grains that fall off the edge of the grid are gone.
5. Repeat the shedding until every cell is stable, then add the next grain.

Nothing about this needs sand — [[Theory-Scientist Mapping]] reads the same
five rules as a model of who works on what.

## Known reference numbers
- 2D stationary mean height: exactly **17/8 = 2.125** grains/cell (this
  model's own BTW control lands at 2.073–2.085 on a finite L=24 grid,
  consistent with the known finite-size gap — see [[Finite-Size Scaling]]).
- Avalanche exponent τ ≈ 1 under naive MLE/log-bin fits; τ_tpl ≈ 0.87 under
  the rigorous truncated-power-law fit — see [[Truncated Power-Law Fitting]].

This is one model among a known family of stylized results — see
[[Known Sandpile Patterns]] for the full list BTW is actually famous for,
and which of them this project checked directly versus only inherited by
citation.

> [!note] The Abelian property (Dhar, 1990)
> The final stable configuration does not depend on the order in which
> unstable cells are relaxed — only the parallel-update *duration* does.
> This is what makes BTW exactly solvable in a way almost no other SOC model
> is.

Reinterpreted in [[Theory-Scientist Mapping]] · Reference implementation
details in [[Running the Code]]
