---
tags: [practical, howto]
---

# Running the Code

Reference implementation: Agents.jl, cross-checked against a dependency-free
Python port and analyzed with the `powerlaw` package. See
[[Tooling - Python vs R vs Stata]] for why Python was chosen for the analysis
step specifically.

## Install Julia

```bash
curl -fsSL https://install.julialang.org | sh
```

## Get the code

```bash
git clone https://github.com/yoosong2000/yoosong2000
cd yoosong2000/sandpile_theory
```

## Install dependencies once

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Run it

```bash
# BTW control, all five drive rules, novelty sweep, finite-size scaling
julia --project=. run_demo.jl            # L=32, ~1-2 min
julia --project=. run_demo.jl 16 5000 20000   # smaller grid, faster

# a video of the field evolving (real MP4, via Makie's bundled ffmpeg)
julia --project=. make_video.jl                # drive=:novelty, novelty_weight=0.7
julia --project=. make_video.jl :uniform
julia --project=. make_video.jl :novelty 0.3   # chaotic regime, for contrast
```

```bash
# statistical analysis and the figures referenced throughout this vault
pip install numpy pandas matplotlib powerlaw
python3 analyze.py
```

> [!warning] `make_video.jl` is reviewed, not run
> Agents.jl's `abmplot`/`abmvideo` keyword signature changed between v5 and
> v6. If it errors on `agent_step!`/`model_step!` arguments, run `?abmvideo`
> in the Julia REPL first — the fix is almost certainly a small keyword-name
> tweak, not a rewrite. `run_demo.jl` is lower risk: it's a closer port of
> the already-validated Python reference.

See [[Avalanche Features]] and [[Novelty-Seeking Phase Transition]] for what
the numbers these scripts print actually mean.
