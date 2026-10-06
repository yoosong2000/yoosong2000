---
tags: [concept, model]
---

# Theory-Scientist Mapping

The reinterpretation at the center of this project: the unmodified
[[Bak-Tang-Wiesenfeld Sandpile]] rule, read as a model of research effort.

| sandpile | this model |
|---|---|
| cell | a theory |
| grid edge | two theories close enough that work on one carries into the other |
| grain | a scientist currently working on that theory |
| z_c = 4 | how many scientists a theory absorbs before it's worked out |
| toppling | the theory saturates; its newest arrivals spill into adjacent, bridging theories |
| avalanche | a research cascade — one more worker sets off a chain of theories taken up in sequence |
| boundary loss | scientists who leave the field |

## Two additions the sand never needed

**Development is irreversible.** A count of every arrival a theory has
*ever* had, not just its current occupancy — unlike sand height, this number
never goes down. A theory is "developed" the moment that count passes zero.

**Priority credit.** The first few scientists to arrive at a theory bank
`decay^k` credit each; everyone after them banks nothing. This is what gives
a theory a real capacity — not a physical limit, but the fact that staying
past the first few arrivals stops paying. When a theory saturates, it's the
*latest* arrivals who move on; the earliest keep their claim and stay.

> [!tip] Why this matters
> The priority rule is the single assumption the [[Philosophy of Science - Overview|philosophical]]
> reading leans on hardest — see [[Social Epistemology of Credit]].

Who moves when a theory saturates, and where, is governed by one of five
rules — see the drive-rule notes linked from [[Theory Sandpile]].
