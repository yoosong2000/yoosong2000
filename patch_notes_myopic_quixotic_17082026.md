# Patch notes — `myopic_quixotic_17082026.jl`

Audit of the ratio measures, the belief/action export path, and the empirical
critical-mass measure; plus the patches applied.

Patched file: `myopic_quixotic_17082026_patched.jl`
Every change is wrapped in `#=== PATCH n (KIND) ===#` … `#=== end PATCH n ===#`.
Grep `PATCH` to walk them in file order.

---

## 1. Definitional check — `ratioBeliefs` / `ratioActs`

The intended semantics were confirmed correct:

- `ratioBeliefs[2]` = % of agents whose credence on arm 2 exceeds their credence on arm 1
- `ratioBeliefs[1]` = the complement
- `ratioActs[k]` = % of agents who *pulled* arm k this round

`ratioActs` does not need the pairwise `>` structure that beliefs need: an action is a
single categorical choice per agent, so `count(==(arm), actions)` is the right
operationalization, and `ratioActs[1] + ratioActs[2]` should always be exactly 100.

**Known edge case (not patched, by design):** strict `>` / `<` means an exact tie counts
toward *neither* ratio. Under `Credence0 = "Agnostic2"` both arms start with identical
α = β priors, so an arm untouched by an agent's neighborhood stays pinned at exactly 0.5
(α and β decay by the same γ). Ties are therefore reachable, though rare once evidence
accrues.

---

## 2. Bugs found and fixed, in the order they were discovered

### Round 1 — the ratio matrices were never filled

`stackratioact` / `stackratiobelief` were pre-allocated `undef` but the per-round
stacking lines were commented out; only the `_arm_2` vectors were populated. The
DataFrame built from them therefore exported `ratioact_arm_1` and `ratiobelief_arm_1`
as **uninitialized memory**. The summary pipeline was unaffected (it used a separate
end-of-run block), so no published numbers were wrong — but the intermediate CSVs were.

Also flagged: redundant recomputation of `ratioBeliefs` inside the `for arm` loop, and a
dead `for arm2` loop in the end-of-run block whose `any(...)` generator owned its own
local `arm2`. That `any(... .> ...)` form also technically means "beats at least one
rival," which only coincides with "beats arm 1" at `armsN = 2`.

### Round 2 — plot/CSV mismatches

| Issue | Detail |
|---|---|
| Verification slice off by one | `numeric_data[:, (end-4-armsN*2):(end-1-armsN*2)]` evaluated to cols 26:29 — the file labelled `stackratioact` actually held `[isCycle, ratioAct1, ratioAct2, ratioBelief1]` |
| BC plot collapsed to the diagonal | `plot!(BCupdate2D, stackbeliefA[...], stackbeliefA[...])` — both axes were arm 1, while the start/end scatter markers used `colA`/`colB` correctly |
| `DataFrame(numeric_data, :auto)` | exports `x1…x34`, forcing hand-done index arithmetic downstream — exactly where the slice bug came from |

### Round 3 — regressions from the restructuring

| Issue | Detail |
|---|---|
| `procNumb == 1 & simulation<2` | In Julia `&` binds **tighter** than `==`, so this parsed as the chained comparison `procNumb == (1 & simulation) < 2`, i.e. `procNumb == isodd(simulation)`. It fired on every **odd** simulation. |
| `tipping_round` trapped in the guard | The guard's matching `end` had moved to after the dashboard, so critical mass was computed only on worker 1 for a couple of sims and never returned from `RunRounds` |
| `target_dir` scope | Defined inside the `csvprint` guard, used afterwards at `cd(target_dir)` and `savefig(..., joinpath(target_dir, ...))` → `UndefVarError` once the quota was exhausted |
| `stackingSSE` never written | Allocated `undef`, exported to the SSE csv and plotted as the Global SSE figure. `SSErecord` was the one actually receiving data — two SSE plots that disagreed by construction. Line 2395 also drew `ylims` from `maximum(stackingSSE)`. |
| `belief_cols` shadowing | Bound as `Vector{Symbol}`, then rebound as a `UnitRange` in the same scope; worked only because the Symbol version was consumed first |
| `playAlot` results width | Allocated its own matrix at `13 + 2*armsN`; widening only the `DoIt` reshape would have thrown `DimensionMismatch` on the first simulation |

**Resolved along the way:** `playSimulation` now returns `stackratioact[end,:]` /
`stackratiobelief[end,:]`, so the summary CSV and the per-round CSV finally use one
definition of the ratios. This was the single largest source of mismatch.

---

## 3. Persistence-gated critical mass

### The problem with the old measure

```julia
threshold = 0.50
maintain  = 50                      # declared, never used
took_over = arm2_ratio[end] >= threshold
```

Three defects:

1. **No persistence gate.** A group oscillating between arm 1 and arm 2 until the final
   round scored identically to one that genuinely locked in.
2. **50% is not a majority.** With `popSize = 8`, `>= 0.50` counts a 4-4 tie as a takeover.
3. **Single terminal threshold** for all runs, regardless of how the simulation ended.

### The replacement

`majority_profile(series, thr, window)` turns a per-round ratio series into a spell
profile; `critical_mass(...)` applies it with thresholds conditioned on the terminal state.

Returned fields:

| field | meaning |
|---|---|
| `onset` | first round of the **first** spell lasting ≥ `window` (may later be lost) |
| `durable` | first round of the **terminal** spell, only if that spell lasted ≥ `window` — this is *the* critical round |
| `spells` | number of separate times the series rose to/above threshold (oscillation count) |
| `longest` | longest spell |
| `frac_above` | fraction of all rounds above threshold |

The three axes requested:

1. **Majority in action** — `tipping_act`, from `stackratioact[:, 2]`
2. **Majority in belief** — `tipping_bel`, from `stackratiobelief[:, 2]`
3. **Threshold by back-tracking the terminal state** — consensus in `roundRecord` is
   defined on beliefs, so on a converged run (`arm 1/2 consensus`) the belief bar is
   unanimity (100%); on a polarized run it stays at simple majority. Actions keep the
   majority bar either way, because ε-exploration means action unanimity is never the
   right target.

```julia
maj_thr = 100.0 * (fld(popSize, 2) + 1) / popSize      # 62.5% at popSize = 8
thr_bel = (consensus1 || consensus2) ? 100.0 : maj_thr
thr_act = maj_thr
```

Derived flags: `took_over` (= `durable` exists), `oscillating` (= no `durable` **and**
`spells >= 2`), and `foreclose_round` (round arm 1 durably locked arm 2 out, on arm-1
consensus runs).

### Indexing verification

Series position *r* **is** round *r*, since both matrices are filled as
`[roundCounter, :]`. Values drop straight into `vline!` on an axis of `rounds = 1:numRuns`
with no offset. Verified against hand-checked cases (`window = 5`, `popSize = 8`):

| case | onset | durable | spells |
|---|---|---|---|
| clean takeover from r6 | 6 | 6 | 1 |
| terminal spell only 4 rounds | — | — | 1 |
| oscillates, ends below | 1 | — | 2 |
| oscillates, ends locked at r9 | 1 | 9 | 2 |
| above throughout | 1 | 1 | 1 |
| 4-of-8 tie every round | — | — | 0 |

Rows 3 vs 4 are the distinction that was missing: both oscillate, only one gets a critical
round. Row 3 still reports `onset = 1` (arm 2 *did* hold a 5-round majority once) with
`oscillating = 1`, separating "never got traction" from "got traction and lost it."
Row 6 is the case the old code scored as a takeover.

---

## 4. Patch index

| # | Kind | Change |
|---|---|---|
| 1 | NEW | `CRIT_WINDOW`, `majority_profile`, `critical_mass` inserted before `RunRounds` |
| 2 | MOVED | `lastResult` + `CM` hoisted out of the export guard — computed for every run |
| 3 | FIXED | `procNumb == 1 & simulation<2` → `&&` |
| 4 | FIXED | `target_dir` hoisted out of the `csvprint` guard |
| 5 | REMOVED | old single-round tipping test |
| 6 | NEW | 7 critical-mass columns on the per-round `df` |
| 7 | CHANGED | `RunRounds` returns `CM` |
| 8 | CHANGED | `playSimulation` receives `CM` |
| 9 | CHANGED | 6 critical-mass fields appended to the outcome row |
| 10 | CHANGED | `DoIt` reshape widened to `13 + 2*armsN + 6` |
| 11 | NEW | extraction + `takeoverRate` / `oscillationRate` / conditional means |
| 12 / 12b | NEW | 7 aggregates and their names on the final summary row |
| 13 | FIXED | `stackingSSE` now written (`cumsum`, per the axis label) |
| 14 | RENAMED | `belief_cols` → `belief_colrange` |
| 15 | FIXED | `playAlot` results width widened to match PATCH 9 |

New columns land **after** `ratioBelief2`, so the existing `14 + i` /
`14 + armsN + i` parsing in `DoIt` is untouched.

### New summary columns

`takeoverRate`, `oscillationRate`, `meanTippingAct`, `meanTippingBelief`,
`meanOnsetAct`, `meanSpellsAct`, `critWindow`

Means are conditioned on the run having tipped — runs that never tipped store 0, and
averaging those in would drag the mean toward zero rather than leaving it undefined.

### New per-round columns

`TippingRound`, `TookOver`, `TippingBelief`, `OnsetAct`, `OnsetBelief`,
`ForecloseRound`, `Oscillating`, `SpellsAct`, `LongestAct`

---

## 5. Left unchanged, deliberately

**`roundRecord` can still contradict the ratio columns.** It breaks ties with
`rand(max_indices)`; `ratioBeliefs` uses strict `>` / `<` and counts a tie toward neither.
A row can read `RoundResult = "arm 2 consensus"` alongside `ratiobelief_arm_2 = 87.5`.
This is a definitional choice rather than a bug — but note `thr_bel = 100.0` on converged
runs inherits it, so a tie-broken "consensus" could leave `tipping_bel` missing on a run
labelled converged. To lock them together, `roundRecord` should return `"X"` on ties
instead of sampling.

**`profileB`** (allocated `Matrix{Any}(undef, numRuns, profile_width)`) is never read or
written — pure waste on every run. Deleting it touches nothing else.

**`relative_progress`** is declared at the top of `RunRounds` as
`Vector{Vector{Float64}}` and later reassigned to a plain vector; the declaration is dead.

---

## 6. Verification performed, and its limits

- **Logic replication** — `majority_profile` re-implemented independently and run against
  the seven cases in §3. All passed.
- **Block balance** — `RunRounds` checked end-to-end, balanced at 0 once inline
  `if …; …; end` and `x = if … end` forms are accounted for.
- **Width consistency** — both `results` declarations confirmed to read
  `13 + 2*armsN + 6`.
- **Column-order consistency** — `belief_cols` ordering (`for j in 1:armsN for i in
  1:popSize`) confirmed to match `vec(roundbeliefs')`, i.e. agent-minor within arm-major.

**Not performed:** the patched file has not been executed. Julia was unavailable in the
analysis environment, so this is structural and logical verification, not a runtime test.
A smoke run at `numRuns = 10`, `times = 2` is worth doing before a full sweep.
