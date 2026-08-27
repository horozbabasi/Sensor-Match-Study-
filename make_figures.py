"""
Regenerate every figure in the paper from the released result files.

    python make_figures.py [path/to/release]      # default: ./release

Reads only the CSV files in <release>/results/ and writes fig1..fig8 to ./figures.
Nothing is retrained and no model is loaded: each figure is a plot of numbers that
are already in the archive.

Fig. 6 is deliberately absent. It shows validation accuracy at every epoch, and
per-epoch histories were never written to disk -- only the final per-run metrics
were. It is the one figure in the paper that cannot be rebuilt from the release.

Figure sizes below reproduce the images embedded in the manuscript exactly.
Requires: numpy, pandas, matplotlib.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # render to file; no interactive window needed
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
ROOT = sys.argv[1] if len(sys.argv) > 1 else "release"   # where the archive was unzipped
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})

# One colour per pretraining source, kept identical across every figure:
RED = "#b03a2e"   # SatlasPretrain
BLU = "#1f5f8b"   # ImageNet
GRN = "#1e8449"   # SatlasPretrain aerial (the sensor-matched checkpoint)
GRY = "grey"      # random initialization

# The six label budgets, in the order they appear on every x-axis.
K = ["5", "10", "25", "50", "100", "full"]
x = np.arange(6)

# --------------------------------------------------------------------------
# Load the per-run results and average over the five seeds
# --------------------------------------------------------------------------
# Each CSV holds one row per (configuration, label budget, seed) run.
eurosat = pd.read_csv(f"{ROOT}/results/eurosat_results.csv")
resisc = pd.read_csv(f"{ROOT}/results/resisc45_results.csv")

runs = pd.concat([eurosat, resisc])
runs = runs[runs.augment == False]        # main grid only; the augmentation tier is separate

grouped = runs.groupby(["dataset", "config_id", "k_shot"])
acc = grouped.test_acc.mean().unstack("k_shot")[K]      # Stage 2 (fine-tuned) accuracy
frozen = grouped.frozen_acc.mean().unstack("k_shot")[K]  # Stage 1 (frozen backbone) accuracy
ece = grouped.ece.mean().unstack("k_shot")[K]            # expected calibration error

E = acc.loc["eurosat"]     # shorthand: EuroSAT accuracy table
R = acc.loc["resisc45"]    # shorthand: RESISC45 accuracy table


def axis_setup(ax, ylabel):
    """Label budget on x, shared grid style, y-label only on the left panel."""
    ax.set_xticks(x)
    ax.set_xticklabels(K)
    ax.set_xlabel("labels per class")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def panel_letter(ax, text):
    """Bold (a)/(b) in the top-left corner, as the journal requires for multi-panel figures."""
    ax.text(0.012, 0.985, text, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")


# --------------------------------------------------------------------------
# Fig. 1 -- accuracy against label budget, every configuration, both datasets
# --------------------------------------------------------------------------
# Line style carries meaning: solid = 9-band or sensor-matched, dashed = 3-band
# or sensor-mismatched, dotted grey = trained from scratch.
fig, axs = plt.subplots(1, 2, figsize=(11.90, 4.485), dpi=300)

eurosat_lines = [
    ("swin_b_satlas-ms_ms9",      "Swin-B / Satlas-MS / 9b",  RED, "o", "-"),
    ("swin_b_satlas-rgb_rgb3",    "Swin-B / Satlas-RGB / 3b", RED, "o", "--"),
    ("swin_b_imagenet_ms9",       "Swin-B / ImageNet / 9b",   BLU, "o", "-"),
    ("swin_b_imagenet_rgb3",      "Swin-B / ImageNet / 3b",   BLU, "o", "--"),
    ("resnet50_satlas-ms_ms9",    "R50 / Satlas-MS / 9b",     RED, "s", "-"),
    ("resnet50_imagenet_ms9",     "R50 / ImageNet / 9b",      BLU, "s", "-"),
    ("resnet50_imagenet_rgb3",    "R50 / ImageNet / 3b",      BLU, "s", "--"),
    ("resnet50_satlas-rgb_rgb3",  "R50 / Satlas-RGB / 3b",    RED, "s", "--"),
    ("swin_b_scratch_ms9",        "Swin-B / Scratch",         GRY, "v", ":"),
    ("resnet50_scratch_ms9",      "R50 / Scratch",            GRY, "v", ":"),
]
for cid, label, colour, marker, style in eurosat_lines:
    axs[0].plot(x, E.loc[cid].values, marker=marker, color=colour,
                ls=style, label=label, ms=4)

resisc_lines = [
    ("swin_b_imagenet_rgb3",      "Swin-B / ImageNet",                BLU, "o", "-"),
    ("swin_b_satlas-aerial_rgb3", "Swin-B / Satlas-Aerial (matched)", GRN, "o", "-"),
    ("resnet50_imagenet_rgb3",    "R50 / ImageNet",                   BLU, "s", "-"),
    ("swin_b_satlas-rgb_rgb3",    "Swin-B / Satlas-S2 (mismatched)",  RED, "o", "--"),
    ("resnet50_satlas-rgb_rgb3",  "R50 / Satlas-S2 (mismatched)",     RED, "s", "--"),
    ("swin_b_scratch_rgb3",       "Swin-B / Scratch",                 GRY, "v", ":"),
    ("resnet50_scratch_rgb3",     "R50 / Scratch",                    GRY, "v", ":"),
]
for cid, label, colour, marker, style in resisc_lines:
    axs[1].plot(x, R.loc[cid].values, marker=marker, color=colour,
                ls=style, label=label, ms=4)

for i, ax in enumerate(axs):
    axis_setup(ax, "test accuracy" if i == 0 else "")
    ax.legend(fontsize=6.5, loc="lower right")
    panel_letter(ax, "(a)" if i == 0 else "(b)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig1.png", dpi=300)
plt.close(fig)

# --------------------------------------------------------------------------
# Fig. 2 -- the 2x2 factorial on EuroSAT, one panel per backbone
# --------------------------------------------------------------------------
# Four arms: Satlas/ImageNet crossed with 9-band/3-band. From those four means:
#   source effect   = average gain from Satlas over ImageNet
#   spectral effect = average gain from 9 bands over 3 bands
#   interaction     = how much the spectral gain depends on the source
fig, axs = plt.subplots(1, 2, figsize=(11.90, 4.29), dpi=300)

for ax, backbone, letter in zip(axs, ["resnet50", "swin_b"], ["(a)", "(b)"]):
    satlas_9, satlas_3, imnet_9, imnet_3 = [
        E.loc[f"{backbone}_{arm}"] for arm in
        ["satlas-ms_ms9", "satlas-rgb_rgb3", "imagenet_ms9", "imagenet_rgb3"]
    ]
    source = 0.5 * ((satlas_9 + satlas_3) - (imnet_9 + imnet_3))
    spectral = 0.5 * ((satlas_9 + imnet_9) - (satlas_3 + imnet_3))
    interaction = (satlas_9 - satlas_3) - (imnet_9 - imnet_3)

    ax.plot(x, source.values, "s-", color=BLU, label="source (Satlas − ImageNet)", ms=4)
    ax.plot(x, spectral.values, "o-", color=RED, label="spectral (9-band − 3-band)", ms=4)
    ax.plot(x, interaction.values, "^--", color=GRN, label="interaction", ms=4)
    ax.axhline(0, color="k", lw=0.8)          # zero line: no effect

    axis_setup(ax, "effect on accuracy" if letter == "(a)" else "")
    ax.legend(fontsize=7)
    panel_letter(ax, letter)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2.png", dpi=300)
plt.close(fig)

# --------------------------------------------------------------------------
# Fig. 3 -- the sensor-match result, with input configuration held constant
# --------------------------------------------------------------------------
# All four arms are Swin-B on identical 3-channel RGB input, so the only thing
# that varies is which checkpoint the weights came from.
fig, ax = plt.subplots(figsize=(7.10, 4.49), dpi=300)

for cid, label, colour, marker, style in [
    ("swin_b_imagenet_rgb3",      "ImageNet (out of domain)",           BLU, "o", "-"),
    ("swin_b_satlas-aerial_rgb3", "Satlas Aerial (modality-matched)",   GRN, "^", "-"),
    ("swin_b_satlas-rgb_rgb3",    "Satlas Sentinel-2 RGB (mismatched)", RED, "s", "--"),
    ("swin_b_scratch_rgb3",       "Scratch",                            GRY, "v", ":"),
]:
    ax.plot(x, R.loc[cid].values, marker=marker, color=colour,
            ls=style, label=label, lw=2, ms=5)

# Double-headed arrow at k=5 measuring the matched-vs-mismatched gap.
aerial_5 = R.loc["swin_b_satlas-aerial_rgb3", "5"]
s2_5 = R.loc["swin_b_satlas-rgb_rgb3", "5"]
ax.annotate("", xy=(0, aerial_5), xytext=(0, s2_5),
            arrowprops=dict(arrowstyle="<->", color=GRN, lw=1.5))
ax.text(0.12, (aerial_5 + s2_5) / 2, f"+{aerial_5 - s2_5:.3f}\nsensor match",
        color=GRN, fontsize=9, va="center")

axis_setup(ax, "test accuracy")
ax.legend(loc="center", bbox_to_anchor=(0.47, 0.40), fontsize=8.5, framealpha=0.95)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3.png", dpi=300)
plt.close(fig)

# --------------------------------------------------------------------------
# Fig. 4 -- SatlasPretrain minus ImageNet, paired arm by arm
# --------------------------------------------------------------------------
# Above zero means geospatial pretraining helped; below zero means it hurt.
fig, axs = plt.subplots(1, 2, figsize=(11.90, 4.485), dpi=300)

for satlas, imnet, label, colour, style in [
    ("resnet50_satlas-ms_ms9",   "resnet50_imagenet_ms9",   "R50 / 9-band",    BLU, "-"),
    ("resnet50_satlas-rgb_rgb3", "resnet50_imagenet_rgb3",  "R50 / 3-band",    BLU, "--"),
    ("swin_b_satlas-ms_ms9",     "swin_b_imagenet_ms9",     "Swin-B / 9-band", RED, "-"),
    ("swin_b_satlas-rgb_rgb3",   "swin_b_imagenet_rgb3",    "Swin-B / 3-band", RED, "--"),
]:
    axs[0].plot(x, (E.loc[satlas] - E.loc[imnet]).values,
                marker="s" if "R50" in label else "o",
                color=colour, ls=style, label=label, ms=4)

for satlas, imnet, label, colour, style in [
    ("resnet50_satlas-rgb_rgb3",  "resnet50_imagenet_rgb3", "R50 / S2-RGB (mismatched)",    BLU, "--"),
    ("swin_b_satlas-rgb_rgb3",    "swin_b_imagenet_rgb3",   "Swin-B / S2-RGB (mismatched)", RED, "--"),
    ("swin_b_satlas-aerial_rgb3", "swin_b_imagenet_rgb3",   "Swin-B / Aerial (matched)",    GRN, "-"),
]:
    axs[1].plot(x, (R.loc[satlas] - R.loc[imnet]).values,
                marker="o", color=colour, ls=style, label=label, ms=4)

for i, ax in enumerate(axs):
    ax.axhline(0, color="k", lw=0.8)
    axis_setup(ax, "accuracy: SatlasPretrain − ImageNet" if i == 0 else "")
    ax.legend(fontsize=7)
    panel_letter(ax, "(a)" if i == 0 else "(b)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig4.png", dpi=300)
plt.close(fig)

# --------------------------------------------------------------------------
# Fig. 5 -- expected calibration error against label budget
# --------------------------------------------------------------------------
# Lower is better: ECE is the gap between a model's confidence and how often
# it is actually right.
fig, axs = plt.subplots(1, 2, figsize=(11.90, 4.29), dpi=300)

for cid, label, colour, marker, style in [
    ("swin_b_satlas-ms_ms9",     "Swin-B / Satlas-MS / 9b", RED, "o", "-"),
    ("swin_b_imagenet_rgb3",     "Swin-B / ImageNet / 3b",  BLU, "o", "-"),
    ("resnet50_satlas-ms_ms9",   "R50 / Satlas-MS / 9b",    RED, "s", "--"),
    ("resnet50_imagenet_rgb3",   "R50 / ImageNet / 3b",     BLU, "s", "--"),
    ("resnet50_scratch_ms9",     "R50 / Scratch / 9b",      GRY, "v", ":"),
    ("swin_b_scratch_ms9",       "Swin-B / Scratch / 9b",   GRY, "v", ":"),
]:
    axs[0].plot(x, ece.loc["eurosat", cid].values, marker=marker,
                color=colour, ls=style, label=label, ms=4)

for cid, label, colour, marker, style in [
    ("swin_b_imagenet_rgb3",      "Swin-B / ImageNet",                BLU, "o", "-"),
    ("swin_b_satlas-aerial_rgb3", "Swin-B / Satlas-Aerial (matched)", GRN, "o", "-"),
    ("resnet50_imagenet_rgb3",    "R50 / ImageNet",                   BLU, "s", "-"),
    ("swin_b_satlas-rgb_rgb3",    "Swin-B / Satlas-S2 (mismatched)",  RED, "o", "--"),
    ("resnet50_satlas-rgb_rgb3",  "R50 / Satlas-S2 (mismatched)",     RED, "s", "--"),
    ("resnet50_scratch_rgb3",     "R50 / Scratch",                    GRY, "v", ":"),
    ("swin_b_scratch_rgb3",       "Swin-B / Scratch",                 GRY, "v", ":"),
]:
    axs[1].plot(x, ece.loc["resisc45", cid].values, marker=marker,
                color=colour, ls=style, label=label, ms=4)

for i, ax in enumerate(axs):
    axis_setup(ax, "expected calibration error" if i == 0 else "")
    ax.legend(fontsize=6.5)
    panel_letter(ax, "(a)" if i == 0 else "(b)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig5.png", dpi=300)
plt.close(fig)

# --------------------------------------------------------------------------
# Fig. 6 -- not reproducible here (see the module docstring)
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Fig. 7 -- what unfreezing the backbone is worth, on EuroSAT
# --------------------------------------------------------------------------
# Stage 2 accuracy minus Stage 1 accuracy. Negative means fine-tuning made the
# frozen representation worse.
fig, ax = plt.subplots(figsize=(7.90, 4.385), dpi=300)
gain = (acc - frozen).loc["eurosat"]

for cid, label, colour, marker, style in [
    ("resnet50_satlas-ms_ms9",   "R50 / Satlas-MS / 9b",    RED, "s", "-"),
    ("resnet50_satlas-rgb_rgb3", "R50 / Satlas-RGB / 3b",   RED, "s", "--"),
    ("resnet50_imagenet_ms9",    "R50 / ImageNet / 9b",     BLU, "s", "-"),
    ("resnet50_imagenet_rgb3",   "R50 / ImageNet / 3b",     BLU, "s", "--"),
    ("swin_b_satlas-ms_ms9",     "Swin-B / Satlas-MS / 9b", RED, "o", "-"),
    ("swin_b_imagenet_rgb3",     "Swin-B / ImageNet / 3b",  BLU, "o", "-"),
    ("resnet50_scratch_ms9",     "R50 / Scratch / 9b",      GRY, "v", ":"),
]:
    ax.plot(x, gain.loc[cid].values, marker=marker, color=colour,
            ls=style, label=label, ms=4)

ax.axhline(0, color="k", lw=0.8)
ax.text(0.02, -0.038, "fine-tuning hurts", color=RED, fontsize=9)
axis_setup(ax, "accuracy gain from fine-tuning (Stage 2 − Stage 1)")
ax.legend(fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}/fig7.png", dpi=300)
plt.close(fig)

# --------------------------------------------------------------------------
# Fig. 8 -- per-class F1 on RESISC45, full labels, seed 42
# --------------------------------------------------------------------------
# Classes are sorted by the matched aerial arm, so the hardest classes sit on
# the left. The scratch line shows how uneven performance is without pretraining.
per_class = pd.read_csv(f"{ROOT}/results/per_class_f1.csv")
subset = per_class[
    (per_class.dataset == "resisc45")
    & (per_class.k_shot.astype(str) == "full")
    & (per_class.seed == 42)
    & (per_class.stage == "stage2")
]
table = subset.pivot(index="class_name", columns="config_id", values="f1")[
    ["swin_b_satlas-aerial_rgb3", "swin_b_imagenet_rgb3", "swin_b_scratch_rgb3"]
].sort_values("swin_b_satlas-aerial_rgb3")

fig, ax = plt.subplots(figsize=(10.90, 4.48), dpi=300)
xx = np.arange(len(table))
ax.plot(xx, table["swin_b_satlas-aerial_rgb3"], "^-", color=GRN,
        label="Swin-B / Satlas-Aerial (matched)", ms=4)
ax.plot(xx, table["swin_b_imagenet_rgb3"], "o-", color=BLU,
        label="Swin-B / ImageNet", ms=4)
ax.plot(xx, table["swin_b_scratch_rgb3"], "v:", color=GRY,
        label="Swin-B / Scratch", ms=4)
ax.set_xticks(xx)
ax.set_xticklabels(table.index, rotation=90, fontsize=7)
ax.set_ylabel("per-class F1")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT}/fig8.png", dpi=300)
plt.close(fig)

print(f"Wrote fig1-fig5, fig7 and fig8 to {OUT}/ (Fig. 6 is not reproducible from the release).")
