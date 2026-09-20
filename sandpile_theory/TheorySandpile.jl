"""
    TheorySandpile

A Bak–Tang–Wiesenfeld sandpile re-read as a model of theory development,
written with Agents.jl.

    cell   -> a theory
    edge   -> two theories close enough that work on one carries over to the other
    grain  -> a scientist working on a theory
    z_c    -> how many scientists a theory can absorb before it is worked out
    topple -> the theory saturates; its late arrivals spill into adjacent
              (auxiliary / bridging) theories
    avalanche -> a research cascade: one more worker sets off a chain of
              theories being taken up in sequence

The lattice geometry carries the "you cannot jump to a distant theory" claim:
a scientist reaches a far-away theory only by the intervening theories being
developed first, one topple at a time.

Everything is standard Abelian BTW except three additions:

  * `visits[pos]` — how many scientists have *ever* worked on a theory.
    Irreversible: theories accumulate development, sand does not.
  * priority credit — the first `credit_slots` arrivals at a theory earn
    `credit_decay^k` each; everyone after them earn nothing.  Credit in
    science goes to whoever gets there first.
  * a novelty bonus and probabilistic theory creation (only active under the
    `:novelty` drive) — see `choose_theory` and `topple!`.

Five driving rules let you ask what the *entry rule of new scientists* does
to the critical state:

  * `:uniform`        — classic BTW driving: a scientist starts anywhere.
  * `:avoid_crowded`  — compares `sample` theories, takes the least worked one
                        (priority credit makes crowded theories worthless).
  * `:matthew`        — the opposite: joins the most crowded of `sample`
                        theories.  Hot topics attract more people.
  * `:frontier`       — looks for an undeveloped theory adjacent to a developed
                        one, i.e. works the edge of what is known.
  * `:novelty`        — continuous version of `:avoid_crowded`/`:frontier`,
                        tuned by `novelty_weight` ∈ [0,1]. Below 0.5 it behaves
                        like `:avoid_crowded`; at and above 0.5 it actively
                        prefers undeveloped theories and lets toppling spawn
                        new ones nearby (`theory_creation`, `created_theories`).
                        See NOVELTY_RESULTS.md for the phase transition this
                        produces at novelty_weight ≈ 0.5.
"""
module TheorySandpile

using Agents
using Random
using Printf
using Statistics

export Scientist, theory_sandpile, run_field!, warmup!,
       occupancy, occupancy_matrix, density, n_developed, activity,
       mean_credit, gini_credit,
       logbin, mle_exponent, loglog_slope, ascii_loglog, summarize

# The four orthogonal neighbours.  A topple sheds exactly one scientist per
# direction, so `length(NEIGHBOR_OFFSETS)` is the shed size and the toppling
# threshold must be at least that big.
const NEIGHBOR_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1))
const SHED = length(NEIGHBOR_OFFSETS)
const DRIVES = (:uniform, :avoid_crowded, :matthew, :frontier, :novelty)

"""
    Scientist

One researcher.  `arrived` is a global clock stamp of when they started on
their current theory, which is what decides who has priority on it.
"""
@agent struct Scientist(GridAgent{2})
    credit::Float64   # priority credit banked so far
    hops::Int         # how many theories they have moved through
    arrived::Int      # arrival clock stamp at the current theory
end

"""
    Field

Model-level state.  Reachable as `model.threshold`, `model.visits`, ... .
"""
mutable struct Field
    L::Int
    threshold::Int          # z_c: scientists a theory absorbs before toppling
    credit_slots::Int       # how many arrivals at a theory earn any credit
    credit_decay::Float64   # kth earner gets credit_decay^k
    drive::Symbol           # :uniform | :avoid_crowded | :matthew | :frontier | :novelty
    sample::Int             # how many theories a new scientist compares
    novelty_weight::Float64 # :novelty only — 0 = crowd-avoidance, 1 = strong novelty bias
    novelty_bonus::Float64  # :novelty only — extra credit for an undeveloped theory's first arrival
    theory_creation::Bool   # :novelty only — let toppling spawn new theories nearby
    visits::Matrix{Int}     # cumulative arrivals per theory, all time
    clock::Int
    record::Bool
    # one entry per model step while `record` is on
    sizes::Vector{Int}
    durations::Vector{Int}
    areas::Vector{Int}
    radii::Vector{Int}
    seeds::Vector{Tuple{Int,Int}}
    left_field::Int         # scientists dissipated at the boundary
    created_theories::Int   # :novelty only — theories spawned by toppling
end

# ---------------------------------------------------------------- construction

"""
    theory_sandpile(; L, threshold, credit_slots, credit_decay, drive, sample,
                      novelty_weight, novelty_bonus, theory_creation, seed)

Build the model.  `drive` picks the entry rule for new scientists — see the
module docstring for the five options. `novelty_weight`, `novelty_bonus`, and
`theory_creation` only take effect when `drive == :novelty`.
"""
function theory_sandpile(; L::Int = 32,
                           threshold::Int = 4,
                           credit_slots::Int = 3,
                           credit_decay::Float64 = 0.5,
                           drive::Symbol = :uniform,
                           sample::Int = 5,
                           novelty_weight::Float64 = 0.5,
                           novelty_bonus::Float64 = 1.0,
                           theory_creation::Bool = true,
                           seed::Int = 42)
    threshold ≥ SHED || throw(ArgumentError(
        "threshold must be ≥ $SHED, the number of neighbours a topple feeds"))
    drive in DRIVES || throw(ArgumentError("unknown drive rule $drive"))
    0 ≤ novelty_weight ≤ 1 || throw(ArgumentError("novelty_weight must be in [0,1]"))

    space = GridSpace((L, L); periodic = false, metric = :manhattan)
    field = Field(L, threshold, credit_slots, credit_decay, drive, sample,
                  novelty_weight, novelty_bonus, theory_creation,
                  zeros(Int, L, L), 0, false,
                  Int[], Int[], Int[], Int[], Tuple{Int,Int}[], 0, 0)

    return StandardABM(Scientist, space;
                       model_step! = field_step!,
                       properties = field,
                       rng = Xoshiro(seed))
end

# -------------------------------------------------------------------- queries

inside(model, p) = 1 ≤ p[1] ≤ model.L && 1 ≤ p[2] ≤ model.L

"How many scientists are working on the theory at `pos` right now."
occupancy(model, pos) = length(ids_in_position(pos, model))

"`occupancy` for the whole lattice, as a matrix (handy for a heatmap)."
occupancy_matrix(model) =
    [occupancy(model, (i, j)) for i in 1:model.L, j in 1:model.L]

"Scientists per theory — the sandpile's 'slope'.  Self-organises to ≈2.1."
density(model) = nagents(model) / model.L^2

"A theory counts as developed once anyone has ever worked on it."
n_developed(model) = count(>(0), model.visits)

"Fraction of recorded steps in which the new scientist set off any topple."
activity(model) = isempty(model.sizes) ? 0.0 : count(>(0), model.sizes) / length(model.sizes)

mean_credit(model) = nagents(model) == 0 ? 0.0 : mean(a.credit for a in allagents(model))

"Gini coefficient of banked credit among the scientists currently in the field."
function gini_credit(model)
    xs = sort!([a.credit for a in allagents(model)])
    n = length(xs)
    (n == 0 || sum(xs) == 0) && return 0.0
    return (2 * sum(i * x for (i, x) in enumerate(xs))) / (n * sum(xs)) - (n + 1) / n
end

# ----------------------------------------------------------------- the rules

"""
    arrive!(agent, pos, model; move = true)

Put `agent` to work on the theory at `pos`, awarding priority credit if the
theory still has an unclaimed credit slot (plus a novelty bonus for its very
first arrival under the `:novelty` drive), and time-stamping the arrival.
"""
function arrive!(agent, pos, model; move::Bool = true)
    v = model.visits[pos[1], pos[2]]
    if v < model.credit_slots
        credit_earned = model.credit_decay^v
        if model.drive === :novelty && v == 0 && model.novelty_weight > 0
            credit_earned += model.novelty_weight * model.novelty_bonus
        end
        agent.credit += credit_earned
    end
    model.visits[pos[1], pos[2]] = v + 1
    model.clock += 1
    agent.arrived = model.clock
    move && move_agent!(agent, pos, model)
    return agent
end

"Neighbouring theories of `pos` that are inside the field."
function field_neighbors(model, pos)
    qs = Tuple{Int,Int}[]
    for d in NEIGHBOR_OFFSETS
        q = (pos[1] + d[1], pos[2] + d[2])
        inside(model, q) && push!(qs, q)
    end
    return qs
end

"Neighbouring theories of `pos` that are inside the field and undeveloped."
function empty_neighbors(model, pos)
    qs = Tuple{Int,Int}[]
    for d in NEIGHBOR_OFFSETS
        q = (pos[1] + d[1], pos[2] + d[2])
        inside(model, q) && model.visits[q[1], q[2]] == 0 && push!(qs, q)
    end
    return qs
end

"Where does the next scientist start?  See `theory_sandpile` for the rules."
function choose_theory(model)
    model.drive === :uniform && return random_position(model)

    cands = [random_position(model) for _ in 1:model.sample]
    crowd(p) = (occupancy(model, p), model.visits[p[1], p[2]])

    if model.drive === :avoid_crowded
        return argmin(crowd, cands)
    elseif model.drive === :matthew
        return argmax(crowd, cands)
    elseif model.drive === :frontier
        for p in cands
            model.visits[p[1], p[2]] == 0 || continue
            any(q -> model.visits[q[1], q[2]] > 0, field_neighbors(model, p)) && return p
        end
        return first(cands)   # no frontier in sight: fall back to uniform
    else # :novelty — continuous interpolation, see module docstring
        if model.novelty_weight < 0.5
            return argmin(crowd, cands)
        else
            score(p) = (model.visits[p[1], p[2]] == 0 ? 0 : 1, occupancy(model, p))
            return argmin(score, cands)
        end
    end
end

"""
    topple!(model, pos)

The theory at `pos` has absorbed all the workers it can. Its `SHED` most
recent arrivals move on, one into each adjacent theory. The earlier arrivals
keep their claim and stay. A scientist pushed off the edge of the field has
left research altogether.

Under `:novelty` with `theory_creation` on, a saturating theory may also seed
brand-new theories in its still-undeveloped neighbours — the "adjacent
possible" — with probability `0.2 * novelty_weight` per empty neighbour.
"""
function topple!(model, pos)
    # copy: the space is mutated below, so we must not hold its internal vector
    ids = sort!(collect(ids_in_position(pos, model)); by = id -> model[id].arrived)
    dirs = shuffle(abmrng(model), collect(NEIGHBOR_OFFSETS))
    for k in 1:SHED
        agent = model[ids[end - SHED + k]]
        agent.hops += 1
        q = (pos[1] + dirs[k][1], pos[2] + dirs[k][2])
        if inside(model, q)
            arrive!(agent, q, model)
        else
            model.left_field += 1
            remove_agent!(agent, model)
        end
    end

    if model.drive === :novelty && model.theory_creation && model.novelty_weight > 0.2
        for q in empty_neighbors(model, pos)
            if rand(abmrng(model)) < 0.2 * model.novelty_weight
                model.visits[q[1], q[2]] = 1
                model.created_theories += 1
            end
        end
    end
    return nothing
end

"""
    relax!(model, seed) -> (size, duration, area, radius)

Run the cascade started at `seed` to completion.  Rounds are synchronous, so
`duration` is the number of generations of the cascade; `size` counts topples,
`area` counts distinct theories that toppled at least once.

By the Abelian property (Dhar 1990) the final configuration does not depend on
the order in which unstable theories are handled — only `duration` does.
"""
function relax!(model, seed::Tuple{Int,Int})
    ntopples = duration = radius = 0
    toppled = Set{Tuple{Int,Int}}()
    queue = Tuple{Int,Int}[seed]

    while !isempty(queue)
        fired = false
        nxt = Tuple{Int,Int}[]
        for p in queue
            occupancy(model, p) < model.threshold && continue
            topple!(model, p)
            fired = true
            ntopples += 1
            push!(toppled, p)
            radius = max(radius, abs(p[1] - seed[1]) + abs(p[2] - seed[2]))
            push!(nxt, p)                     # may still be over threshold
            append!(nxt, field_neighbors(model, p))
        end
        fired && (duration += 1)
        queue = unique(nxt)
    end
    return ntopples, duration, length(toppled), radius
end

"""
    field_step!(model)

One unit of slow time: exactly one new scientist enters the field, and the
resulting cascade is run to completion before anyone else arrives.  That
separation of timescales — slow drive, fast relaxation — is what makes the
system self-organise to the critical state rather than merely sit near it.
"""
function field_step!(model)
    pos = choose_theory(model)
    agent = add_agent!(pos, model, 0.0, 0, 0)
    arrive!(agent, pos, model; move = false)

    s, d, a, r = relax!(model, pos)
    if model.record
        push!(model.sizes, s)
        push!(model.durations, d)
        push!(model.areas, a)
        push!(model.radii, r)
        push!(model.seeds, pos)
    end
    return nothing
end

# ------------------------------------------------------------------- driving

"Step the model without recording, to burn in the critical state."
function warmup!(model, n::Int)
    model.record = false
    step!(model, n)
    return model
end

"""
    run_field!(model; warmup = 20_000, measure = 60_000)

Burn in, clear the statistics, then measure.  Avalanche distributions taken
before the pile reaches its stationary slope are not the SOC distributions —
they still carry the transient.
"""
function run_field!(model; warmup::Int = 20_000, measure::Int = 60_000)
    warmup!(model, warmup)
    empty!(model.sizes); empty!(model.durations)
    empty!(model.areas); empty!(model.radii); empty!(model.seeds)
    model.record = true
    step!(model, measure)
    model.record = false
    return model
end

# ------------------------------------------------------------------ analysis

"""
    logbin(xs; base = 1.6) -> Vector{Tuple{Float64,Float64}}

Logarithmically binned, width-normalised pdf.  Linear binning of a power law
is dominated by empty bins in the tail and will fool you about the exponent.
"""
function logbin(xs::AbstractVector{<:Real}; base::Real = 1.6)
    ys = filter(>(0), xs)
    isempty(ys) && return Tuple{Float64,Float64}[]
    hi = maximum(ys)
    edges = Float64[1.0]
    while last(edges) < hi
        push!(edges, max(last(edges) * base, last(edges) + 1))
    end
    counts = zeros(Int, length(edges) - 1)
    for y in ys
        k = searchsortedlast(edges, float(y))
        k = clamp(k, 1, length(counts))
        counts[k] += 1
    end
    n = length(ys)
    out = Tuple{Float64,Float64}[]
    for i in eachindex(counts)
        counts[i] == 0 && continue
        w = edges[i+1] - edges[i]
        push!(out, (sqrt(edges[i] * edges[i+1]), counts[i] / (w * n)))
    end
    return out
end

"""
    mle_exponent(xs, xmin) -> (alpha, n)

Discrete power-law MLE of Clauset, Shalizi & Newman (2009), eq. 3.7.  Fit only
the tail: `xmin` matters more than the estimator does.  Report it, and never
claim a power law from the MLE alone — check the binned plot too.
"""
function mle_exponent(xs::AbstractVector{<:Real}, xmin::Real)
    tail = filter(≥(xmin), xs)
    n = length(tail)
    n < 20 && return (NaN, n)
    s = sum(log(x / (xmin - 0.5)) for x in tail)
    return (1 + n / s, n)
end

"Least-squares slope of the log-binned pdf over `lo ≤ s ≤ hi`."
function loglog_slope(binned, lo::Real, hi::Real)
    pts = [p for p in binned if lo ≤ p[1] ≤ hi]
    length(pts) < 3 && return NaN
    lx = [log(p[1]) for p in pts]
    ly = [log(p[2]) for p in pts]
    mx, my = mean(lx), mean(ly)
    return sum((lx .- mx) .* (ly .- my)) / sum((lx .- mx) .^ 2)
end

"Terminal log-log plot, so the distribution is visible without a plotting stack."
function ascii_loglog(binned; width::Int = 34, label::AbstractString = "P(s)")
    isempty(binned) && return
    lo = log10(minimum(p[2] for p in binned))
    hi = log10(maximum(p[2] for p in binned))
    span = max(hi - lo, eps())
    println("  log-binned $label:")
    for (x, y) in binned
        bar = round(Int, width * (log10(y) - lo) / span)
        @printf("    %10.1f  %9.3e  %s\n", x, y, "#"^max(bar, 1))
    end
end

"""
    summarize(model)

Print the observables that decide whether this run looks critical.
"""
function summarize(model)
    s = model.sizes
    active = filter(>(0), s)
    binned = logbin(s)
    tau_mle, n = mle_exponent(s, 4)
    tau_fit = -loglog_slope(binned, 4, model.L^2 / 4)

    label = model.drive === :novelty ?
        "drive=:novelty (novelty_weight=$(model.novelty_weight))" :
        "drive=:$(model.drive)"
    println("=== ", label, "  L=", model.L, "  z_c=", model.threshold, " ===")
    @printf("  stationary density   : %.3f scientists/theory  (2D BTW ref ≈ 2.125)\n", density(model))
    @printf("  active steps         : %d/%d = %.3f\n", length(active), length(s), activity(model))
    if !isempty(active)
        @printf("  <s>                  : %.1f   max s = %d\n", mean(active), maximum(active))
        @printf("  <T>                  : %.1f   max T = %d\n",
                mean(filter(>(0), model.durations)), maximum(model.durations))
    end
    @printf("  tau (MLE, s ≥ 4)     : %.3f  (n=%d)\n", tau_mle, n)
    @printf("  tau (log-bin slope)  : %.3f\n", tau_fit)
    @printf("  theories ever worked : %d/%d\n", n_developed(model), model.L^2)
    @printf("  scientists in field  : %d   (left the field: %d)\n", nagents(model), model.left_field)
    @printf("  credit: mean %.2f, Gini %.3f\n", mean_credit(model), gini_credit(model))
    model.drive === :novelty && @printf("  theories created     : %d\n", model.created_theories)
    ascii_loglog(binned)
    return nothing
end

end # module
