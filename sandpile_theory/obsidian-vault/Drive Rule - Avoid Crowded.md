---
tags: [drive-rule, finding]
---

# Drive Rule - Avoid Crowded

`drive = :avoid_crowded` — samples a handful of theories, joins the
least-worked one.

Individually rational: credit is already gone on a crowded theory under the
[[Theory-Scientist Mapping|priority rule]], so there's no reason to join one.
But this is the rule that **breaks** scale-free behavior — see
[[Novelty-Seeking Phase Transition]] for the numbers, and
[[Social Epistemology of Credit]] for why that's philosophically interesting
rather than just a quirk of this particular model.

## Numbers (L=24)
- stationary density: 2.337
- activity: only 0.056 of steps trigger a cascade — far quieter than [[Drive Rule - Uniform]]
- mean avalanche size ⟨s⟩: 265.2 — nearly 5× uniform's, when something does happen
- τ (truncated power-law fit): 0.946

> [!warning] The characteristic-scale bump
> The avalanche-size distribution under this rule develops a visible bump at
> large `s` instead of a clean cutoff — the signature of a creeping-back
> characteristic scale, not pure scale-free behavior. Rare, catastrophic
> events replace frequent, graded ones.

Contrast with [[Drive Rule - Novelty-Seeking]], which looks similar in a
one-line description and does the opposite to the field.
