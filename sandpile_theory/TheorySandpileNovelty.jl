"""
    TheorySandpileNovelty

Extended model with novelty-seeking incentives and theory creation.

Additions to the base model:
  * `novelty_weight` — interpolate between pure crowding-avoidance (0) and
    strong novelty bias (1). At 1, bonus credit rewards new theories.
  * `novelty_bonus` — extra credit awarded for working on undeveloped theories.
  * Theory creation — when a theory reaches saturation, neighboring empty
    spaces can spawn new theories, expanding the research landscape.
  * Comparative experiments — sweep novelty_weight to measure how it affects
    the critical exponent, density, and activity.
"""
module TheorySandpileNovelty

using Agents
using Random
using Printf
using Statistics

export ScientistNovelty, theory_sandpile_novelty, run_field!, warmup!,
       occupancy, density, n_developed, activity, mean_credit,
       logbin, mle_exponent, loglog_slope, ascii_loglog, summarize

const NEIGHBOR_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1))
const SHED = length(NEIGHBOR_OFFSETS)

@agent struct ScientistNovelty(GridAgent{2})
    credit::Float64
    hops::Int
    arrived::Int
end

mutable struct FieldNovelty
    L::Int
    threshold::Int
    credit_slots::Int
    credit_decay::Float64
    novelty_weight::Float64      # 0 = pure crowd-avoidance, 1 = strong novelty bias
    novelty_bonus::Float64        # extra credit for undeveloped theories
    sample::Int
    visits::Matrix{Int}
    clock::Int
    record::Bool
    sizes::Vector{Int}
    durations::Vector{Int}
    areas::Vector{Int}
    radii::Vector{Int}
    created_theories::Int
    left_field::Int
end

function theory_sandpile_novelty(; L::Int = 32,
                                   threshold::Int = 4,
                                   credit_slots::Int = 3,
                                   credit_decay::Float64 = 0.5,
                                   novelty_weight::Float64 = 0.5,
                                   novelty_bonus::Float64 = 1.0,
                                   sample::Int = 5,
                                   seed::Int = 42)
    threshold ≥ SHED || throw(ArgumentError("threshold must be ≥ $SHED"))
    0 ≤ novelty_weight ≤ 1 || throw(ArgumentError("novelty_weight must be in [0,1]"))

    space = GridSpace((L, L); periodic = false, metric = :manhattan)
    field = FieldNovelty(L, threshold, credit_slots, credit_decay,
                        novelty_weight, novelty_bonus, sample,
                        zeros(Int, L, L), 0, false,
                        Int[], Int[], Int[], Int[], 0, 0)

    return StandardABM(ScientistNovelty, space;
                       model_step! = field_step!,
                       properties = field,
                       rng = Xoshiro(seed))
end

inside(model, p) = 1 ≤ p[1] ≤ model.L && 1 ≤ p[2] ≤ model.L

occupancy(model, pos) = length(ids_in_position(pos, model))

density(model) = nagents(model) / model.L^2

n_developed(model) = count(>(0), model.visits)

activity(model) = isempty(model.sizes) ? 0.0 : count(>(0), model.sizes) / length(model.sizes)

mean_credit(model) = nagents(model) == 0 ? 0.0 : mean(a.credit for a in allagents(model))

function arrive!(agent, pos, model; move::Bool = true)
    v = model.visits[pos[1], pos[2]]
    credit_earned = 0.0
    
    if v < model.credit_slots
        credit_earned = model.credit_decay^v
        # Novelty bonus: extra credit for undeveloped theories
        if v == 0 && model.novelty_weight > 0
            credit_earned += model.novelty_weight * model.novelty_bonus
        end
    end
    
    agent.credit += credit_earned
    model.visits[pos[1], pos[2]] = v + 1
    model.clock += 1
    agent.arrived = model.clock
    move && move_agent!(agent, pos, model)
    return agent
end

function field_neighbors(model, pos)
    qs = Tuple{Int,Int}[]
    for d in NEIGHBOR_OFFSETS
        q = (pos[1] + d[1], pos[2] + d[2])
        inside(model, q) && push!(qs, q)
    end
    return qs
end

function empty_neighbors(model, pos)
    qs = Tuple{Int,Int}[]
    for d in NEIGHBOR_OFFSETS
        q = (pos[1] + d[1], pos[2] + d[2])
        inside(model, q) && model.visits[q[1], q[2]] == 0 && push!(qs, q)
    end
    return qs
end

function choose_theory(model)
    cands = [random_position(model) for _ in 1:model.sample]
    
    # Score: (occupancy, visits)
    # novelty_weight interpolates: at 0, pure crowd-avoidance; at 1, novelty bias
    score(p) = begin
        occ = occupancy(model, p)
        vis = model.visits[p[1], p[2]]
        # At novelty_weight=0: prefer low (occ, vis)
        # At novelty_weight=1: prefer high vis (newly developed), then low occ
        if model.novelty_weight < 0.5
            # Crowd-avoidance dominates
            return (occ, vis)
        else
            # Novelty dominates: prefer undeveloped, then avoid crowded
            return (vis == 0 ? 0 : 100, occ)
        end
    end
    
    return argmin(score, cands)
end

function topple!(model, pos)
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
    
    # Theory creation: spawn new theories in empty neighbor slots
    # (only if novelty weight is high enough to justify exploration)
    if model.novelty_weight > 0.3
        empty = empty_neighbors(model, pos)
        # Probabilistically create new theories proportional to novelty_weight
        for p in empty
            if rand(abmrng(model)) < model.novelty_weight * 0.1  # 10% per empty slot
                model.visits[p[1], p[2]] = 1  # Mark as minimally developed
                model.created_theories += 1
            end
        end
    end
    
    return nothing
end

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
            push!(nxt, p)
            append!(nxt, field_neighbors(model, p))
        end
        fired && (duration += 1)
        queue = unique(nxt)
    end
    return ntopples, duration, length(toppled), radius
end

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
    end
    return nothing
end

function warmup!(model, n::Int)
    model.record = false
    step!(model, n)
    return model
end

function run_field!(model; warmup::Int = 20_000, measure::Int = 60_000)
    warmup!(model, warmup)
    empty!(model.sizes); empty!(model.durations)
    empty!(model.areas); empty!(model.radii)
    model.record = true
    step!(model, measure)
    model.record = false
    return model
end

# ------------------------------------------------------------------ analysis

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

function mle_exponent(xs::AbstractVector{<:Real}, xmin::Real)
    tail = filter(≥(xmin), xs)
    n = length(tail)
    n < 20 && return (NaN, n)
    s = sum(log(x / (xmin - 0.5)) for x in tail)
    return (1 + n / s, n)
end

function loglog_slope(binned, lo::Real, hi::Real)
    pts = [p for p in binned if lo ≤ p[1] ≤ hi]
    length(pts) < 3 && return NaN
    lx = [log(p[1]) for p in pts]
    ly = [log(p[2]) for p in pts]
    mx, my = mean(lx), mean(ly)
    return sum((lx .- mx) .* (ly .- my)) / sum((lx .- mx) .^ 2)
end

function ascii_loglog(binned; width::Int = 28, label::AbstractString = "P(s)")
    isempty(binned) && return
    lo = log10(minimum(p[2] for p in binned))
    hi = log10(maximum(p[2] for p in binned))
    span = max(hi - lo, eps())
    for (x, y) in binned[1:min(10, end)]
        bar = round(Int, width * (log10(y) - lo) / span)
        @printf("      %9.1f  %9.3e  %s\n", x, y, "#"^max(bar, 1))
    end
end

function summarize(model)
    s = model.sizes
    active = filter(>(0), s)
    binned = logbin(s)
    tau_mle, n = mle_exponent(s, 4)
    tau_fit = -loglog_slope(binned, 4, model.L^2 / 4)

    @printf("  novelty_w=%.2f: ρ=%.3f  act=%.3f  <s>=%6.1f  τ=%.3f  theories_created=%d\n",
            model.novelty_weight, density(model), activity(model),
            isempty(active) ? 0 : mean(active), tau_fit, model.created_theories)
end

end # module
