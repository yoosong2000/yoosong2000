# What philosophical questions this model is good for

The sandpile is strongest exactly where a philosophical question can be reduced to a
claim about *aggregation* — what collective pattern follows from a stated micro-rule —
and weakest wherever the question needs semantic content (truth, meaning, justification)
that the model has no representation for. Four clusters where it does real work, not
just supplies a metaphor, followed by where it has nothing to say.

## 1. Social epistemology of the division of cognitive labor

This is the model's home literature (Kitcher 1990, Strevens 2003, Zollman 2010,
Weisberg & Muldoon 2009), and it's a well-posed philosophical question: given that
credit is scarce and winner-take-all, what allocation rule for scientists-to-topics
serves the community's epistemic good, and does individual rationality get you there
automatically?

The sandpile doesn't just illustrate this question, it produces a counter-example
inside it: `:avoid_crowded` is the individually rational response to winner-take-all
credit, and it's the rule that breaks scale-free search and produces rarer, more
catastrophic cascades instead. That's a formal instance of the "invisible hand"
optimism in Polanyi's *Republic of Science* and Kitcher's own early argument failing on
its own terms — rational self-interest under this reward structure doesn't converge on
good collective search, it needs an additional ingredient (novelty-seeking) to do that.
A philosophy paper arguing against naive invisible-hand pictures of science usually has
nothing but intuition pumps for this claim. The model gives it a mechanism and a number
(§ below).

## 2. Aggregation failure / methodological individualism in philosophy of social science

More general than (1): this is a clean case where summing individually optimal choices
does not yield an optimal collective outcome, in an *epistemic* domain rather than the
usual resource-allocation one (commons problems, Condorcet paradoxes). The value of
running it as a model rather than arguing it in prose is that you can ask exactly *how
much* worse — mean avalanche size roughly 2.5× larger, activity dropping by nearly
half, once novelty-seeking replaces pure crowd-avoidance (see `ANALYSIS.md`) — rather
than just asserting a qualitative mismatch.

## 3. Kuhn's punctuated structure, given a mechanism instead of a metaphor

Kuhn's picture — long normal science, punctuated by revolutions — has always had an
explanatory gap: why punctuation, and why does revolution-size vary the way it seems to
historically? SOC answers a narrower but sharper question: punctuation and graduated
change aren't two different kinds of process needing two different explanations (an
incrementalist story for normal science, a special story for revolutions) — they're the
same distribution read at different scales. That's a real philosophical move: it
dissolves an apparent dichotomy (gradualist vs. revolutionary change) into one mechanism
with one exponent.

Worth being honest about the limit here too: this explains the *shape* of the size
distribution, not why any particular revolution happened, which is the part historians
of science actually argue about.

## 4. Emergence as a formally tractable case

Philosophers arguing about emergence usually have either toy cellular automata with no
interpretive content, or richly interpreted social phenomena with no formal handle. SOC
is unusual in giving both at once: a macro-regularity (the critical exponent,
universality class membership) that is fully bottom-up, substrate-independent across
very different microscopic rules (`:uniform`, `:frontier`, and `:novelty≥0.5` all land
in a similar τ≈0.5–0.9 band despite different mechanisms), yet not predictable by just
inspecting one agent's rule. That's a clean, checkable instance of "weak emergence" to
reason from, instead of the usual hand-wave.

## Where it has nothing to say

Anything that needs a truth predicate. Realism/anti-realism, verisimilitude, whether
science converges on true theories rather than merely crowded ones — the model has no
concept of a theory being right, only developed. Individual belief-updating and
confirmation theory are also out of scope; there's no evidence, no credence, nothing
resembling Bayesian updating, just attention allocation. And incommensurability in Kuhn
and Feyerabend's strong sense (conceptual, not just positional, distance between
paradigms) is at most gestured at by the bridging-theory requirement — that captures
*structural* distance in a graph, not meaning change.

## The one-sentence version

It's a formal object for testing whether a *proposed reward structure* for a research
community is epistemically self-defeating, which is a live, contested question in
social epistemology rather than a settled one.

## Supporting numbers

From `ANALYSIS.md` (L=24, truncated-power-law MLE, Clauset–Shalizi–Newman 2009 method):

| regime | τ (truncated power-law) | mean avalanche size ⟨s⟩ | activity |
|---|---|---|---|
| `novelty_weight < 0.5` (crowd-avoidance) | 0.91 ± 0.02 | 265–285 | ~0.053 |
| `novelty_weight ≥ 0.5` (novelty-seeking) | 0.52 ± 0.03 | 660–705 | ~0.031 |

Truncated power law beats the lognormal alternative decisively on both sides
(log-likelihood ratio R > 20, p < 10⁻¹⁰⁰), so the difference is not "critical vs. not
critical" in the crude sense — it is a real, large shift in *how* critical, hinging on
a single incentive: whether scientists are rewarded for novelty or merely for avoiding
crowds.

## See also

- `README.md` §5, "Is this a legitimate model of science?" — the science-of-science
  framing this page's §1–2 builds on, including the honest limitations (fixed 2D
  lattice, no theory-space growth, conserved population).
- `tutorial.html` — the interactive walkthrough of the mechanism itself (also published
  live at <https://claude.ai/artifact/3dHFieeRjFspChFA39mG9j>), useful context before
  these questions, since they all presuppose the credit/toppling rules it explains.
- Full bibliography in `README.md` §6.
