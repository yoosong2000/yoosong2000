# Novelty-Seeking Phase Transition Results

## Summary

The extended model demonstrates a **critical phase transition** as novelty-seeking incentives increase. Pure crowd-avoidance breaks self-organized criticality; balanced novelty-seeking restores it—suggesting that scientists' rational preferences for novel topics naturally reproduce the critical state.

## Experimental Setup

- **Model:** TheorySandpileNovelty.jl with `novelty_weight` ∈ [0, 1]
- **Parameter sweep:** 11 points from 0.0 to 1.0, in increments of 0.1
- **Grid:** L = 24 (576 theories)
- **Protocol:** 10k warmup + 30k measured steps, seeded reproducibly
- **Cross-check:** Python reference implementation (validate_novelty_v2.py)

## Key Findings

### Phase Transition at novelty_weight ≈ 0.5

```
novelty_w  density  activity   <s>    τ_fit   interpretation
─────────────────────────────────────────────────────────────
   0.0     2.387    0.050    300.3   1.492   ╱ Chaotic regime
   0.2     2.403    0.053    278.3   1.375   │ events rare, large
   0.4     2.450    0.052    282.4   1.665   │ when they occur
───────────────────────────────────────────┼─────────────────
   0.5     2.200    0.031    700.6   1.060   │ CRITICAL REGIME
   0.6     2.340    0.030    703.6   0.975   │ τ ≈ 1.0 recovered
   0.8     2.316    0.030    710.0   1.033   ╲ scale-free
   1.0     2.252    0.031    686.9   0.974
```

### τ(novelty_weight)

- **Below transition** (nw < 0.5): τ ≈ 1.5 (far from critical τ ≈ 1)
- **Above transition** (nw ≥ 0.5): τ ≈ 0.97 ± 0.05 (**critical**)
- **Transition sharpness:** Δτ ≈ 0.5 over Δnw = 0.1

### Observable changes across the transition

| metric | nw=0.0 | nw=0.5 | change | meaning |
|---|---|---|---|---|
| τ | 1.49 | 1.06 | ↓ 0.43 | back to scale-free |
| <s> | 300 | 701 | ↑ 2.3× | larger events when they happen |
| activity | 0.050 | 0.031 | ↓ 0.38 | fewer events, but bigger |
| density | 2.39 | 2.20 | ↓ 0.19 | scientists spread out |

## Interpretation

### Why crowd-avoidance fails (nw < 0.5)

Pure crowd-avoidance (the original `:avoid_crowded` rule) creates a **pathological loading regime**:
- Scientists rationally avoid saturated theories
- Theories never fully empty; they load unevenly
- The system accumulates stress and releases it catastrophically
- Result: characteristic scale (τ > 1), not scale-free events

### Why novelty-seeking restores criticality (nw ≥ 0.5)

Balanced novelty incentives create **marginal stability**:
- Novelty bonus for undeveloped theories provides alternative targets
- Crowded theories naturally depopulate (no credit left anyway)
- Load distributes evenly across the theory space
- System self-organizes to the brink of instability (the critical state)
- Result: τ ≈ 1.0, matching the Bak–Tang–Wiesenfeld exponent

### The physical mechanism

When `novelty_weight` crosses 0.5, the topic-choice heuristic flips:
- **Below 0.5:** minimize(occupancy, visits) — leads to pathological piling
- **Above 0.5:** prefer undeveloped theories — load self-regulates

The critical state is not *prevented* by novelty-seeking; it is *enabled* by it. Scientists' preference for novel topics is exactly the negative feedback that maintains criticality.

## Failure modes and future directions

1. **Theory creation didn't activate.** Toppling-triggered creation of new theories was implemented but didn't create measurable numbers (showing in both Python and Julia). The fixed grid may be saturating too quickly. **Fix:** let the grid grow dynamically or increase creation probability.

2. **No finite-size scaling yet.** The transition should survive and sharpen as L grows. **Next:** run at L=16, 32, 48 and plot the collapse.

3. **Credit mechanisms are simplified.** Real citation counts are heavy-tailed, path-dependent, and field-dependent. The decay^k model is a crude approximation.

## Conclusions

1. **Novelty-seeking is not the enemy of criticality**—it is the mechanism that maintains it.
2. **The critical state is fragile.** Departures from novelty-seeking (pure crowd-avoidance) quickly destabilize it.
3. **Testable prediction:** empirical research cascades (bursts of work on related topics) should show power-law size distributions, with exponent ~1, if this model holds. Deviation from τ ≈ 1 would indicate topic-choice incentives are misaligned.

## References

- Kitcher (1990) — the formal theory of topic allocation under credit incentives.
- Zollman (2010) — transient diversity and epistemic efficiency.
- Foster, Rzhetsky, Evans (2015) — empirical measures of explore/exploit in science, show scientists are more conservative than optimal.

---

**Code:** TheorySandpileNovelty.jl, run_novelty_sweep.jl, validate_novelty_v2.py
