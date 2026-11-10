"""Builds the blog charts from the recorded results in ncm-research-playground/results.

Run from the repository root:  python blog/make_charts.py   (needs matplotlib)
Numbers that are not in a results file (the 4-sample quality run and the paper's
reported values) are written out below with their source.
"""

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "ncm-research-playground" / "results"
OUT = ROOT / "blog" / "images"
OUT.mkdir(parents=True, exist_ok=True)

NCM = "#2a78d6"   # NCM, everywhere
AR = "#eb6834"    # autoregressive baseline, everywhere
REF = "#8a8986"   # paper reference values
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 14,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "axes.axisbelow": True,
})


def save(fig, name, note, title=None):
    if title:
        fig.suptitle(title, x=0.01, y=0.98, ha="left", va="top", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, note, fontsize=9, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95 if title else 1))
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("wrote", OUT / name)


def label_bars(ax, bars, fmt):
    for b in bars:
        ax.annotate(fmt(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    fontsize=11, color=INK)


# Benchmark summary (benchmark 58a26ec7, 10 timed runs per cell)
summary = {}
for path in (RESULTS / "benchmarks").glob("*-summary.csv"):
    for row in csv.DictReader(path.open()):
        summary[(row["model"], int(row["batch_size"]))] = row
HW = "Apple M5 Pro, CPU, fp32"

# 1. Sequential forward passes
fig, ax = plt.subplots(figsize=(8, 4.2))
names = ["NCM (TinyStories)", "TinyStories-33M\n(autoregressive)"]
vals = [int(summary[("ncm-tinystories", 1)]["passes"]), int(summary[("ar-tinystories-33m", 1)]["passes"])]
bars = ax.barh(names, vals, color=[NCM, AR], height=0.5)
for b, v in zip(bars, vals):
    ax.annotate(f"{v} passes", (v, b.get_y() + b.get_height() / 2), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=12, color=INK)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
ax.set_xlim(0, 150)
ax.set_xlabel("sequential forward passes to write 128 tokens")
save(fig, "01-sequential-passes.png", "Counted from the recorded runs. NCM's level 0 is a lookup, not a transformer pass.", title="NCM needs 11 sequential steps where a classic LLM needs 128")

# 2. End-to-end latency, median with p95
fig, ax = plt.subplots(figsize=(8, 4.6))
x = [0, 1]
w = 0.36
for i, (model, color, label) in enumerate([("ncm-tinystories", NCM, "NCM"), ("ar-tinystories-33m", AR, "TinyStories-33M")]):
    med = [float(summary[(model, b)]["total_s_median"]) * 1000 for b in (1, 4)]
    p95 = [float(summary[(model, b)]["total_s_p95"]) * 1000 for b in (1, 4)]
    pos = [p + (i - 0.5) * w for p in x]
    bars = ax.bar(pos, med, w * 0.92, color=color, label=label)
    ax.errorbar(pos, med, yerr=[[0, 0], [p - m for p, m in zip(p95, med)]], fmt="none",
                ecolor=INK2, elinewidth=1.2, capsize=4)
    for b, top in zip(bars, p95):
        ax.annotate(f"{b.get_height():.0f} ms", (b.get_x() + b.get_width() / 2, top), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=11, color=INK)
ax.set_xticks(x, ["batch 1", "batch 4"])
ax.set_ylabel("time to finish the whole text (ms)")
ax.set_ylim(0, 1600)
ax.legend(loc="upper left")
save(fig, "02-latency.png", f"Median of 10 timed runs after 2 warm ups; whisker shows p95. {HW}.", title="End to end, NCM is 1.42x faster at batch 1 and 1.79x at batch 4")

# 3. Time to first visible output (log scale)
fig, ax = plt.subplots(figsize=(8, 4.6))
for i, (model, color, label) in enumerate([("ncm-tinystories", NCM, "NCM (whole text at once)"), ("ar-tinystories-33m", AR, "TinyStories-33M (first token)")]):
    med = [float(summary[(model, b)]["first_output_s_median"]) * 1000 for b in (1, 4)]
    pos = [p + (i - 0.5) * w for p in x]
    bars = ax.bar(pos, med, w * 0.92, color=color, label=label)
    label_bars(ax, bars, lambda v: f"{v:.1f} ms" if v < 100 else f"{v:.0f} ms")
ax.set_yscale("log")
ax.set_ylim(1, 5000)
ax.set_xticks(x, ["batch 1", "batch 4"])
ax.set_ylabel("time until the reader sees something (ms, log scale)")
ax.legend(loc="upper left")
save(fig, "03-first-output.png", f"Median of 10 timed runs. NCM has nothing to show until the last level is done. {HW}.", title="A classic LLM shows its first word about 80x sooner")

# 4. Throughput
fig, ax = plt.subplots(figsize=(8, 4.6))
for i, (model, color, label) in enumerate([("ncm-tinystories", NCM, "NCM"), ("ar-tinystories-33m", AR, "TinyStories-33M")]):
    med = [float(summary[(model, b)]["words_per_s_median"]) for b in (1, 4)]
    pos = [p + (i - 0.5) * w for p in x]
    bars = ax.bar(pos, med, w * 0.92, color=color, label=label)
    label_bars(ax, bars, lambda v: f"{v:.0f}")
ax.set_xticks(x, ["batch 1", "batch 4"])
ax.set_ylabel("words per second")
ax.set_ylim(0, 700)
ax.legend(loc="upper left")
save(fig, "04-throughput.png", f"Median words per second over 10 timed runs. {HW}.", title="NCM's throughput advantage grows with batch size")

# 5. Reproduction of the paper's TinyStories result
merge_log = (RESULTS / "official_eval" / "tinystories" / "merge.log").read_text()
seeds = [(m.group(1), float(m.group(2))) for m in re.finditer(r"^\s+(\d+)\s+([\d.]+)\s+[\d.]+\s*$", merge_log, re.M)]
mean = re.search(r"mean\s+([\d.]+)\s+[\d.]+\s+\(\+/- ([\d.]+)", merge_log)
mean_v, std_v = float(mean.group(1)), float(mean.group(2))
PAPER_PPL = 15.41  # paper, Table 8
fig, ax = plt.subplots(figsize=(8, 4.6))
labels = [f"seed {s}" for s, _ in seeds] + ["mean"]
vals = [v for _, v in seeds] + [mean_v]
colors = [NCM] * len(seeds) + ["#1a4f91"]
bars = ax.bar(labels, vals, 0.6, color=colors)
ax.errorbar([len(seeds)], [mean_v], yerr=[std_v], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=4)
label_bars(ax, bars[:-1], lambda v: f"{v:.2f}")
ax.annotate(f"{mean_v:.2f}", (len(seeds), mean_v + std_v), xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=11, color=INK)
ax.axhline(PAPER_PPL, color=REF, linestyle="--", linewidth=1.6, label=f"paper reports {PAPER_PPL}")
ax.legend(loc="upper left")
ax.set_ylim(0, 21)
ax.set_ylabel("gen-PPL under GPT-2-large (lower is better)")
save(fig, "05-reproduction.png", "Official eval scripts, unmodified: 5 seeds x 256 stories. Mean 16.23 +/- 0.48; entropy 3.967 vs 3.98 reported.", title="The paper's TinyStories quality reproduces on a laptop CPU")

# 6. Quality head to head (one 4-sample comparison run, seed 0)
QUALITY = {"NCM": (13.48, 4.056), "TinyStories-33M": (9.74, 3.993)}  # docs/EXPERIMENTS.md, experiment 4
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.4))
for ax, idx, title, ylim, fmt in [(a1, 0, "Fluency: gen-PPL (lower is better)", 16, "{:.2f}"),
                                  (a2, 1, "Diversity: entropy (higher is more varied)", 5, "{:.3f}")]:
    vals = [QUALITY[k][idx] for k in QUALITY]
    bars = ax.bar(list(QUALITY), vals, 0.55, color=[NCM, AR])
    label_bars(ax, bars, lambda v, f=fmt: f.format(v))
    ax.set_ylim(0, ylim)
    ax.set_title(title, fontsize=12)
fig.suptitle("The classic LLM writes more fluent stories; NCM is slightly more varied",
             x=0.01, ha="left", fontsize=15, fontweight="bold", color=INK)
save(fig, "06-quality.png", "One comparison run: 4 samples each, seed 0, temperature 1, scored by GPT-2-large.")

# 7. Coarse to fine: how close each level gets to the final latent
probe = (RESULTS / "logs" / "probe_levels.log").read_text()
rows = re.findall(r"^\s+(\d+)\s+(\d+)\s+\(.*?\)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", probe, re.M)
lvls = [int(r[0]) for r in rows]
lens = [int(r[1]) for r in rows]
ms = [float(r[2]) for r in rows]
cos = [float(r[4]) for r in rows]
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(lvls, cos, color=NCM, linewidth=2, marker="o", markersize=7)
for l, n, c in zip(lvls, lens, cos):
    if l in (1, 3, 6, 9, 11):
        ax.annotate(f"{n} codes", (l, c), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=10, color=INK2)
ax.set_xticks(lvls)
ax.set_ylim(0, 1.05)
ax.set_xlabel("level (number of codes grows from 8 to 128)")
ax.set_ylabel("cosine similarity to the final latent")
save(fig, "07-coarse-to-fine.png", "From experiments/probe_levels.py on the official TinyStories checkpoint (2 samples).", title="Each level refines the sketch: the latent converges coarse to fine")

# 8. Time per level
fig, ax = plt.subplots(figsize=(8, 4.6))
bars = ax.bar(lvls, ms, 0.7, color=NCM)
ax.set_xticks(lvls, [f"L{l}\n{n}" for l, n in zip(lvls, lens)], fontsize=10)
ax.set_xlabel("level and its number of codes, all predicted in one parallel pass")
ax.set_ylabel("time for the level (ms)")
save(fig, "08-time-per-level.png", "From experiments/probe_levels.py, batch 2, first (cold) run, so absolute times are higher than the benchmark.", title="Finer levels cost more, but each is still a single pass")

# 9. What the paper claims: speedup over the strongest diffusion LM
PAPER_SPEEDUP = {"OpenWebText": 10, "LM1B": 21, "XSum": 31}  # paper abstract, arXiv:2610.14162
fig, ax = plt.subplots(figsize=(8, 4.2))
bars = ax.bar(list(PAPER_SPEEDUP), list(PAPER_SPEEDUP.values()), 0.55, color=REF)
label_bars(ax, bars, lambda v: f"{v:.0f}x")
ax.set_ylim(0, 36)
ax.set_ylabel("speedup over the strongest diffusion LM")
save(fig, "09-paper-speedups.png", "From the paper's abstract (Kumbong et al., 2026), GPU. Not measured here; the baseline is diffusion, not autoregressive.", title="What the paper reports: NCM vs diffusion language models")
