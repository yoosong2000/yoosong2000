---
tags: [finding, method]
---

# Avalanche Features

Every cascade in [[Theory Sandpile]] is logged as four numbers, each
meaningful on its own.

| feature | counts | reads as |
|---|---|---|
| **size (s)** | number of topples | total follow-on work the cascade triggered |
| **duration (T)** | parallel-update rounds | how many bridging-theory hops deep it went |
| **area (a)** | distinct theories touched | breadth, vs. size's willingness to revisit a hot theory repeatedly |
| **radius (r)** | max distance from the seed | a direct test of "a distant theory needs its bridges built first" (see [[Theory-Scientist Mapping]]) |

Two features tracked but not yet analyzed in this project: the
**size–radius relation** gives a fractal dimension (`area ~ radius^D`) —
whether avalanches are compact blobs or spindly filaments — and the
**duration–size relation** (`T ~ s^(1/D)` in SOC theory) is a second,
independent scaling law that should hold together with τ for "critical" to
mean more than a size-distribution artifact. Natural extension to
[[Running the Code|analyze.py]].

Size is the feature fitted in [[Truncated Power-Law Fitting]] and reported
throughout [[Novelty-Seeking Phase Transition]].
