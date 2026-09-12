#!/usr/bin/env julia
# Usage:  julia --project=. run_demo.jl [L] [warmup] [measure]
#
# First time:  julia --project=. -e 'using Pkg; Pkg.instantiate()'

include("TheorySandpile.jl")
using .TheorySandpile
using Printf
using Random
using Statistics

const L       = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 32
const WARMUP  = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 20_000
const MEASURE = length(ARGS) ≥ 3 ? parse(Int, ARGS[3]) : 60_000

# ---------------------------------------------------------------------------
# 1. Classic BTW, as a plain array, as a control.
#    The agent model must reproduce these numbers; if it does not, the extra
#    machinery (credit, entry rules) has broken the sandpile rather than
#    extended it.
# ---------------------------------------------------------------------------
function btw_reference(L; warmup = 20_000, measure = 60_000, seed = 7)
    rng = Random.Xoshiro(seed)
    z = zeros(Int, L, L)
    sizes = Int[]
    stack = Tuple{Int,Int}[]
    for step in 1:(warmup + measure)
        i, j = rand(rng, 1:L), rand(rng, 1:L)
        z[i, j] += 1
        s = 0
        push!(stack, (i, j))
        while !isempty(stack)
            (a, b) = pop!(stack)
            z[a, b] < 4 && continue
            z[a, b] -= 4
            s += 1
            for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
                p, q = a + di, b + dj
                (1 ≤ p ≤ L && 1 ≤ q ≤ L) || continue      # else: off the edge
                z[p, q] += 1
                z[p, q] ≥ 4 && push!(stack, (p, q))
            end
            z[a, b] ≥ 4 && push!(stack, (a, b))
        end
        step > warmup && push!(sizes, s)
    end
    return z, sizes
end

println("\n############  control: plain BTW sandpile  ############\n")
zref, sref = btw_reference(L; warmup = WARMUP, measure = MEASURE)
act = filter(>(0), sref)
binned = logbin(sref)
@printf("  stationary density   : %.3f grains/cell  (2D BTW ref ≈ 2.125)\n", mean(zref))
@printf("  active steps         : %.3f\n", length(act) / length(sref))
@printf("  <s> = %.1f   max s = %d\n", mean(act), maximum(act))
@printf("  tau (log-bin slope)  : %.3f\n",
        -loglog_slope(binned, 4, L^2 / 4))
ascii_loglog(binned)

# ---------------------------------------------------------------------------
# 2. The theory model, under each rule for how new scientists pick a topic.
#    :uniform is the control — it is BTW with extra bookkeeping, so it should
#    land on the control numbers above.  The other three are the experiment.
# ---------------------------------------------------------------------------
println("\n############  theory sandpile (Agents.jl)  ############")
models = Dict{Symbol,Any}()
for drive in (:uniform, :avoid_crowded, :matthew, :frontier)
    model = theory_sandpile(; L = L, drive = drive, seed = 1)
    run_field!(model; warmup = WARMUP, measure = MEASURE)
    println()
    summarize(model)
    models[drive] = model
end

# ---------------------------------------------------------------------------
# 3. Finite-size scaling.  A power law with a cutoff that moves with L is the
#    real signature of criticality; a power law at one L alone proves nothing.
# ---------------------------------------------------------------------------
println("\n############  finite-size scaling (drive=:uniform)  ############\n")
@printf("  %4s %10s %10s %12s\n", "L", "<s>", "max s", "tau_fit")
for l in (8, 16, 32)
    m = theory_sandpile(; L = l, drive = :uniform, seed = 3)
    run_field!(m; warmup = 5_000 * (l ÷ 8), measure = 30_000)
    a = filter(>(0), m.sizes)
    @printf("  %4d %10.1f %10d %12.3f\n", l, mean(a), maximum(a),
            -loglog_slope(logbin(m.sizes), 4, l^2 / 4))
end

# ---------------------------------------------------------------------------
# 4. Idiomatic Agents.jl data collection, if you would rather have a DataFrame
#    than the vectors the model keeps itself.
# ---------------------------------------------------------------------------
#
#   using Agents
#   m = theory_sandpile(; L = 32)
#   warmup!(m, 20_000)
#   mdata = [density, n_developed, nagents]
#   _, mdf = run!(m, 5_000; mdata)
#
# and per-scientist data with
#
#   adata = [(:credit, mean), (:hops, maximum)]
#   adf, _ = run!(m, 5_000; adata)

println("\ndone.")
