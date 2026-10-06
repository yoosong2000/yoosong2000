---
tags: [finding, method, soc]
---

# Finite-Size Scaling

A power law measured at a single grid size proves very little — the actual
signature of [[Self-Organized Criticality]] is a cutoff that moves
*with* the system size `L`:

```
P(s) ~ s^-τ · G(s / L^D)
```

This project checked that directly on [[Drive Rule - Uniform|the uniform
control]], using the dependency-free Python reference implementation
(see [[Running the Code]]).

## The check (L=12 vs. L=24, `:uniform`)

| L | density | mean ⟨s⟩ | max s | τ (log-bin slope) |
|---|---|---|---|---|
| 12 | 2.083 | 16.5 | 219 | 0.991 |
| 24 | 2.073 | 56.4 | 1034 | 1.021 |

Mean avalanche size goes **16.5 → 56.4** as `L` doubles — a ratio of 3.4,
against the exact BTW prediction of `⟨s⟩ ∝ L²` (ratio 4 for a doubling),
since in the stationary state every added grain must random-walk to the
boundary before it can leave. The gap from 4 is the expected finite-size
correction at these small grid sizes. `max s` grows even faster (219 → 1034,
a 4.7× jump), and τ stays essentially flat (0.99 → 1.02) across both sizes.

> [!important] Why this is the actual evidence of criticality
> A fixed exponent with a cutoff that moves with `L` is what distinguishes a
> genuine scale-free process from a single lucky-looking histogram. See
> [[Known Sandpile Patterns]] #2 and #9, and
> [[Truncated Power-Law Fitting]] for the stricter version of this same
> caution applied to the [[Novelty-Seeking Phase Transition|novelty
> sweep]]'s τ values.

## What's verified vs. not

The L=12/L=24 numbers above come from an actual run of
`validate_reference.py`. `run_demo.jl` additionally sweeps L=8, 16, 32 on
the Julia side — written and reviewed, consistent with the above, but not
executed in this project's own development environment (see the
[[Running the Code|execution caveat]] that applies to all the Julia code
here).

## Not yet done

The [[Novelty-Seeking Phase Transition]] was only checked at a single grid
size, L=24. Whether the sharp step at `novelty_weight = 0.5` survives at
L=12 and L=48 — whether the transition point itself drifts with `L`, the
way a true phase boundary should stay fixed while `L` only changes the
sharpness of the step — is the most direct follow-up this project's central
finding is still missing. See [[Research Questions]].
