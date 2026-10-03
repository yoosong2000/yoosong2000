"""Render an animation of the theory-sandpile field evolving over time.

Two panels, side by side, updated every N steps of the simulation:
  left  — occupancy (how many scientists are on each theory right now)
  right — cumulative development (how many scientists have EVER worked there)

An avalanche shows up as a burst of colour spreading outward from the seed
theory in the left panel; the right panel only ever fills in, never empties —
visualising "theories developed" as an irreversible frontier advancing across
the field.

No ffmpeg is required: this writes an animated GIF via Pillow, which any
browser or image viewer plays back like a short video.

    python3 make_video.py                       # drive=:novelty, novelty_weight=0.7
    python3 make_video.py :uniform               # classic BTW driving
    python3 make_video.py :novelty 0.3           # chaotic regime (see NOVELTY_RESULTS.md)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from validate_reference import Model


def collect_frames(drive=":novelty", novelty_weight=0.7, L=28, warmup=6000,
                    n_steps=400, seed=7, stride=1):
    """Run the model and snapshot the grid every `stride` steps.

    Returns (occ_frames, dev_frames, seed_positions) as numpy arrays /
    lists, each of shape (n_frames, L, L).
    """
    m = Model(L=L, drive=drive, novelty_weight=novelty_weight, seed=seed)
    for _ in range(warmup):
        m.step(record=False)

    occ_frames, dev_frames, seeds, sizes = [], [], [], []
    for _ in range(n_steps):
        p, s = m.step(record=True)
        occ_frames.append(np.array(m.occupancy_grid(), dtype=float))
        dev_frames.append(np.array(m.visits_grid(), dtype=float))
        seeds.append(p)
        sizes.append(s)
    return occ_frames[::stride], dev_frames[::stride], seeds[::stride], sizes[::stride], m


def render(drive=":novelty", novelty_weight=0.7, L=28, warmup=6000,
           n_steps=400, seed=7, out="theory_field.gif", fps=12):
    occ, dev, seeds, sizes, m = collect_frames(
        drive=drive, novelty_weight=novelty_weight, L=L, warmup=warmup,
        n_steps=n_steps, seed=seed)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    label = f"{drive}" + (f"  (novelty_weight={novelty_weight})" if drive == ":novelty" else "")
    fig.suptitle(f"Theory sandpile — drive={label}", fontsize=12)

    im1 = ax1.imshow(occ[0], cmap="inferno", vmin=0, vmax=4, origin="lower")
    ax1.set_title("occupancy (scientists on theory now)")
    ax1.set_xticks([]); ax1.set_yticks([])
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    dev_max = max(int(d.max()) for d in dev) or 1
    im2 = ax2.imshow(dev[0], cmap="viridis", vmin=0, vmax=dev_max, origin="lower")
    ax2.set_title("cumulative development (ever worked)")
    ax2.set_xticks([]); ax2.set_yticks([])
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    marker1, = ax1.plot([], [], "o", color="cyan", markersize=8, markeredgecolor="white")
    step_text = fig.text(0.5, 0.02, "", ha="center", fontsize=10)

    def update(i):
        im1.set_data(occ[i])
        im2.set_data(dev[i])
        sy, sx = seeds[i][0] - 1, seeds[i][1] - 1
        marker1.set_data([sx], [sy])
        step_text.set_text(f"step {i}/{len(occ)}   avalanche size s={sizes[i]}   "
                            f"new-scientist arrival marked in cyan")
        return im1, im2, marker1, step_text

    anim = FuncAnimation(fig, update, frames=len(occ), interval=1000 / fps, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"wrote {out}  ({len(occ)} frames @ {fps} fps = {len(occ)/fps:.1f}s)")
    print(f"  L={L}  drive={label}  final density={np.mean(occ[-1]):.3f}"
          f"  theories developed={int(np.sum(dev[-1] > 0))}/{L*L}")


if __name__ == "__main__":
    drive = sys.argv[1] if len(sys.argv) > 1 else ":novelty"
    nw = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
    render(drive=drive, novelty_weight=nw)
