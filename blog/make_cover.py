"""Builds the dev.to cover image (1000 x 420, dev.to's recommended ratio).

Run from the repository root:  python blog/make_cover.py   (needs matplotlib)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "images" / "cover.png"

BG = "#0f1a2b"
INK = "#f0efec"
INK2 = "#a9b4c4"
NCM = "#3987e5"
AR = "#eb6834"

LEVELS = [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]  # official TinyStories level lengths

fig = plt.figure(figsize=(10, 4.2), dpi=200)
fig.patch.set_facecolor(BG)
ax = fig.add_axes((0, 0, 1, 1))
ax.set_xlim(0, 1000)
ax.set_ylim(0, 420)
ax.axis("off")

# Left: title block
ax.text(56, 318, "NEXT-CONCEPT MODELING", color=NCM, fontsize=13, fontweight="bold", family="DejaVu Sans")
ax.text(56, 250, "Writing text\ncoarse to fine", color=INK, fontsize=31, fontweight="bold",
        family="DejaVu Sans", va="center", linespacing=1.1)
ax.text(56, 152, "11 parallel steps instead of 128.\nI tested Stanford's NCM against a classic LLM.",
        color=INK2, fontsize=12.5, family="DejaVu Sans", va="center", linespacing=1.5)

# Right: the NCM hierarchy, one row per level, widening coarse to fine
x0, x1 = 520, 950
top, row_h, gap = 360, 14, 6
for i, n in enumerate(LEVELS):
    y = top - i * (row_h + gap)
    width = (x1 - x0) * n / LEVELS[-1]
    cx = (x0 + x1) / 2
    left = cx - width / 2
    cells = min(n, 32)  # draw at most 32 cells so the finest rows stay legible
    cell_w = width / cells
    alpha = 0.35 + 0.65 * i / (len(LEVELS) - 1)
    for c in range(cells):
        ax.add_patch(FancyBboxPatch((left + c * cell_w + 1, y), max(cell_w - 2, 1.5), row_h,
                                    boxstyle="round,pad=0,rounding_size=2", linewidth=0,
                                    facecolor=NCM, alpha=alpha))
ax.text(x0 - 12, top + row_h / 2, "L0", color=INK2, fontsize=9, ha="right", va="center")
ax.text(x0 - 12, top - 11 * (row_h + gap) + row_h / 2, "L11", color=INK2, fontsize=9, ha="right", va="center")

# Below it: the autoregressive token chain, one token per step
y_ar = 58
n_tok = 40
tok_w = (x1 - x0) / n_tok
for t in range(n_tok):
    ax.add_patch(FancyBboxPatch((x0 + t * tok_w + 1, y_ar), tok_w - 2, row_h,
                                boxstyle="round,pad=0,rounding_size=2", linewidth=0,
                                facecolor=AR, alpha=0.9 if t < 8 else 0.25))
ax.text(x0 - 12, y_ar + row_h / 2, "LLM", color=INK2, fontsize=9, ha="right", va="center")
ax.text(x1, y_ar - 16, "one token per step", color=INK2, fontsize=9, ha="right", va="center")
ax.text(x1, top + row_h + 16, "one whole level per step", color=INK2, fontsize=9, ha="right", va="center")

fig.savefig(OUT, dpi=200, facecolor=BG)
print("wrote", OUT)
