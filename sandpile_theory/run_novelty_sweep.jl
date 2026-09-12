#!/usr/bin/env julia
# Comparative experiments: how does novelty-seeking weight affect the critical state?

include("TheorySandpileNovelty.jl")
using .TheorySandpileNovelty
using Printf

const L       = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 32
const WARMUP  = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 10_000
const MEASURE = length(ARGS) ≥ 3 ? parse(Int, ARGS[3]) : 30_000

println("\n############  Novelty-seeking sweep (L=$L)  ############\n")
println("novelty_w  density  activity   <s>    τ_fit  theories")
println("─" ^ 58)

for nw in 0.0:0.1:1.0
    model = theory_sandpile_novelty(; L = L, novelty_weight = nw, seed = Int(100 + nw*100))
    run_field!(model; warmup = WARMUP, measure = MEASURE)
    summarize(model)
end

println("\n" * "─" ^ 58)
println("Summary: Watch how τ changes as novelty_weight → 1")
println("- At 0.0: pure crowd-avoidance (breaks SOC)")
println("- At 0.5: balanced incentives")
println("- At 1.0: strong novelty bias (should recover SOC if theory creation works)")
