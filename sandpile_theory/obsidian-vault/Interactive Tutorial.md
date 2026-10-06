---
tags: [practical, howto]
---

# Interactive Tutorial

`tutorial.html` in the repository root is a self-contained, interactive
walkthrough of [[Theory Sandpile]] — not a static writeup, a live
simulation running in the browser. Also published at
<https://claude.ai/artifact/3dHFieeRjFspChFA39mG9j>.

## What's on it

- **A live sandpile in the hero.** A from-scratch JavaScript
  reimplementation of the [[Bak-Tang-Wiesenfeld Sandpile|BTW]] rules, a
  22×22 field, running continuously, with buttons to switch between
  [[Drive Rule - Uniform|uniform]], [[Drive Rule - Avoid Crowded|avoid
  crowds]], and [[Drive Rule - Novelty-Seeking|seek novelty]] and watch the
  avalanche statistics shift in real time — largest avalanche so far, steps
  taken, percent of theories developed.
- The [[Self-Organized Criticality]] background and the
  [[Theory-Scientist Mapping]] table.
- The [[Drive Rule - Uniform|five drive rules]] as cards.
- [[Avalanche Features]], explained against the live demo directly above it.
- **Two charts drawn to scale from the actual run data** behind
  [[Novelty-Seeking Phase Transition]] — redrawn as theme-aware inline SVG
  (not pasted matplotlib PNGs) so they read correctly in both light and dark
  viewing modes.
- [[Research Questions]], [[Tooling - Python vs R vs Stata]], and copy-button
  code snippets matching [[Running the Code]].
- A condensed version of [[References]].

## Relationship to this vault

The HTML tutorial and this Obsidian vault cover the same material in two
different media: the tutorial is a single linear page built to be shared
and explored visually (with a live demo neither Obsidian nor plain markdown
can run); the vault is the same content factored into atomic, cross-linked
notes built to be searched, graphed, and extended. Treat `tutorial.html` as
the thing to send someone who wants to *see* the model working, and this
vault as the thing to open when you want to trace how one finding connects
to another.

## Verification note

This session has no live preview tool, so before publishing, the embedded
simulation and chart-coordinate logic were dry-run in Node.js separately
(confirmed no infinite loops, no NaN coordinates, correct stationary
densities in the 2.1–2.5 range) and the full script was syntax-checked —
but the rendered page itself has not been visually inspected.
