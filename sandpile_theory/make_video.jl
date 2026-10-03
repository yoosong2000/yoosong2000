#!/usr/bin/env julia
"""
Render an MP4 of the theory-sandpile field evolving, using Agents.jl's native
Makie-based visualization (`abmvideo`).

UNTESTED: this environment cannot install Julia (the egress policy blocks
julialang's binary host, and GitHub only ships source releases), so this
script is reviewed but not run. It mirrors `make_video.py`'s two-panel idea
(occupancy + cumulative development) but produces a real MP4 directly via
Makie's `FFMPEG_jll` — a Julia-managed portable ffmpeg binary — so it needs
no system package install, unlike the Python version's Pillow-GIF fallback.

CAVEAT ON TOP OF "untested": Agents.jl's `abmplot`/`abmvideo` keyword names
and calling convention changed between v5 and v6 (the move to `StandardABM`
storing its own stepping functions changed what needs to be passed
explicitly). This is written against the documented v6 API (matching this
repo's `Project.toml` Agents = "6" pin), but if it errors on
`agent_step!`/`model_step!` argument handling, run `? abmvideo` and
`? abmplot` in the Julia REPL first — that will show the exact signature for
whatever Agents.jl version actually resolves, which is the one thing this
review couldn't confirm without a working Julia install.

    julia --project=. -e 'using Pkg; Pkg.add(["CairoMakie"]); Pkg.instantiate()'
    julia --project=. make_video.jl                     # drive=:novelty, nw=0.7
    julia --project=. make_video.jl :uniform
    julia --project=. make_video.jl :novelty 0.3         # chaotic regime, for contrast
"""
include("TheorySandpile.jl")
using .TheorySandpile
using Agents
using CairoMakie   # headless, no GPU/display needed — the right choice for a batch render

const DRIVE = length(ARGS) ≥ 1 ? Symbol(ARGS[1]) : :novelty
const NW    = length(ARGS) ≥ 2 ? parse(Float64, ARGS[2]) : 0.7
const L     = length(ARGS) ≥ 3 ? parse(Int, ARGS[3]) : 28
const WARMUP = 6000
const FRAMES = 400
const OUT = "theory_field_$(DRIVE)_$(NW).mp4"

model = theory_sandpile(; L = L, drive = DRIVE, novelty_weight = NW, seed = 7)
warmup!(model, WARMUP)
model.record = true   # so avalanche stats are visible during the recorded window too

# Agents.jl calls this once per agent per step; all of this model's dynamics
# live in `field_step!` (registered as `model_step!` in `theory_sandpile`), so
# the per-agent hook is a deliberate no-op.
agent_step!(agent, model) = nothing

# Colour each scientist by banked credit — makes the priority-rule "early
# arrivals keep the claim" dynamic visible as color, not just position.
agent_color(a) = a.credit
agent_size(a) = 6

# `occupancy_matrix` is already exported by TheorySandpile.jl; Agents.jl's
# `heatarray` accepts a function of the model, so this draws the occupancy
# heatmap live underneath the agent markers — the same left panel as the
# Python version, but with agents drawn as individual points on top of it.
fig, ax, abmobs = abmplot(model;
    agent_step!, model_step! = TheorySandpile.field_step!,
    agent_color, agent_size,
    heatarray = TheorySandpile.occupancy_matrix,
    heatkwargs = (colorrange = (0, 4), colormap = :inferno),
    figure = (; size = (700, 650)),
    title = "Theory sandpile — drive=$(DRIVE)" *
            (DRIVE === :novelty ? "  (novelty_weight=$NW)" : ""))

abmvideo(OUT, model, agent_step!, TheorySandpile.field_step!;
         agent_color, agent_size,
         heatarray = TheorySandpile.occupancy_matrix,
         heatkwargs = (colorrange = (0, 4), colormap = :inferno),
         framerate = 12, frames = FRAMES,
         title = "Theory sandpile — drive=$(DRIVE)" *
                 (DRIVE === :novelty ? "  (novelty_weight=$NW)" : ""))

println("wrote $OUT  ($FRAMES frames)")
@show density(model) n_developed(model) mean_credit(model)
