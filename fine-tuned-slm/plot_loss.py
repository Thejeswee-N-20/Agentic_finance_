"""Render the training curves for the QLoRA run.

Loss and learning rate are different measures on different scales, so they get
their own panels rather than a shared twin axis.
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = Path(__file__).with_name("train.log")
OUT = Path(__file__).with_name("loss_curve.png")

SERIES_1 = "#2a78d6"  # blue
SERIES_2 = "#eb6834"  # orange
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"

TOTAL_STEPS = 228
STEPS_PER_EPOCH = TOTAL_STEPS / 3.0


def main():
    log = LOG.read_text()
    pat = re.compile(r"'loss': ([0-9.]+).*?'learning_rate': ([0-9.e-]+).*?'epoch': ([0-9.]+)")
    rows = pat.findall(log)
    if not rows:
        raise SystemExit("no loss rows in train.log")

    losses = [float(r[0]) for r in rows]
    lrs = [float(r[1]) for r in rows]
    steps = list(range(5, 5 * len(rows) + 1, 5))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 6), dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
    )
    fig.patch.set_facecolor("white")

    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        # epoch boundaries as recessive reference lines
        for e in (1, 2):
            ax.axvline(e * STEPS_PER_EPOCH, color=GRID, linewidth=1,
                       linestyle=(0, (4, 4)), zorder=0)

    ax1.plot(steps, losses, "-", color=SERIES_1, linewidth=2, marker="o", markersize=3.5,
             markerfacecolor=SERIES_1, markeredgecolor="white", markeredgewidth=0.6)
    ax1.set_ylabel("training loss", color=INK_MUTED, fontsize=10)
    ax1.set_title(
        "Qwen2.5-3B-Instruct QLoRA on FinGPT forecaster-dow30 — 228 steps, 2.97 epochs",
        color=INK, fontsize=12, pad=12, loc="left",
    )

    # direct-label only the endpoints, not every point
    ax1.annotate(f"{losses[0]:.3f}", (steps[0], losses[0]), textcoords="offset points",
                 xytext=(6, 6), fontsize=9, color=INK)
    ax1.annotate(f"{losses[-1]:.3f}", (steps[-1], losses[-1]), textcoords="offset points",
                 xytext=(-8, -14), fontsize=9, color=INK, ha="right")
    for e in (1, 2):
        ax1.annotate(f"epoch {e}", (e * STEPS_PER_EPOCH, max(losses)),
                     textcoords="offset points", xytext=(4, -4), fontsize=8,
                     color=INK_MUTED, va="top")

    ax2.plot(steps, lrs, "-", color=SERIES_2, linewidth=2)
    ax2.set_ylabel("learning rate", color=INK_MUTED, fontsize=10)
    ax2.set_xlabel("optimizer step", color=INK_MUTED, fontsize=10)
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax2.yaxis.get_offset_text().set_color(INK_MUTED)
    ax2.annotate("cosine schedule, 3% warmup",
                 (steps[len(steps) // 3], lrs[len(steps) // 3]),
                 textcoords="offset points", xytext=(8, 8), fontsize=9, color=INK_MUTED)

    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}  ({len(rows)} points, loss {losses[0]:.3f} -> {losses[-1]:.3f})")


if __name__ == "__main__":
    main()
