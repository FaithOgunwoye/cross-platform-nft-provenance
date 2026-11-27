"""Camera-ready figures for the new experiments (E1 collection-aware scaling,
E2 hard attacks with example thumbnails).  Same style as make_figures.py.

Inputs:  results/large/sib_e1.csv, results/large/sib_e2.csv,
         results/large/examples/ref_nouns_5__*.png   (Nouns art is CC0)
Output:  results/figures/fig_scale_collection_aware.{pdf,png}
         results/figures/fig_hard_attacks.{pdf,png}
Run:     python src/make_camera_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))
from make_figures import C1, C2, C3, C4, COL_W, GRID, INK, INK2, SEQ, save, style

BASE = Path(__file__).parent.parent
LARGE = BASE / "results" / "large"


def fig_scale():
    """FPR on genuine NFTs vs registry size: visual similarity alone rises
    with every real sibling added; collection-aware screening stays flat."""
    e1 = pd.read_csv(LARGE / "sib_e1.csv")
    series = [("M7 Transf. pHash", "M7 TA-pHash", C1, "o"),
              ("M10 CLIP", "M10 CLIP", C2, "s"),
              ("M11 SSCD", "M11 SSCD", C3, "^"),
              ("Cascade T3=CLIP", "Cascade (T3 = CLIP)", C4, "D")]
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for sysname, label, col, mk in series:
        for aware, ls, face in ((False, "-", col), (True, "--", "white")):
            g = e1[(e1.system == sysname) & (e1.collection_aware == aware)].sort_values("index_size")
            x, y = g.index_size.values, g.FPR.values
            if not aware:
                ax.fill_between(x, g.FPR_lo, g.FPR_hi, color=col, alpha=0.13, lw=0)
            ax.plot(x, y, ls, color=col, lw=1.3, marker=mk, ms=3.6, mfc=face, mec=col,
                    mew=0.9, zorder=3 if not aware else 4)
    n_neg = int(e1[(e1.system == "M1 Raw SHA-256")].iloc[0][["FP", "TN"]].sum()
                - e1[(e1.system == "M1 Raw SHA-256")].iloc[0]["FP_wrong_src"])
    # direct labels at the right end (visual-only curves)
    xmax = e1.index_size.max()
    ends = {s: e1[(e1.system == s) & (~e1.collection_aware) & (e1.index_size == xmax)].FPR.iloc[0]
            for s, *_ in series}
    offs = {"M11 SSCD": 0, "M10 CLIP": -0.05, "Cascade T3=CLIP": 0.05, "M7 Transf. pHash": 0}
    end_labels = {"M11 SSCD": "SSCD", "M10 CLIP": "CLIP",
                  "Cascade T3=CLIP": "Cascade", "M7 Transf. pHash": "M7"}
    for sysname, label, col, _ in series:
        ax.annotate(end_labels[sysname], (xmax, ends[sysname]),
                    xytext=(4, offs[sysname] * 100), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.set_xlim(0, xmax * 1.02)
    ax.set_ylim(-0.02, 1.0)
    ax.set_xticks([100, 1000, 2000, 3000, xmax])
    ax.set_xticklabels(["100", "1k", "2k", "3k", f"{xmax/1000:.1f}k"])
    ax.set_xlabel("Registry entries", fontsize=9)
    ax.set_ylabel(f"FPR on {n_neg} genuine NFTs", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, color=GRID, lw=0.4); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    leg = [Line2D([], [], color=INK2, lw=1.3, ls="-", label="visual similarity only"),
           Line2D([], [], color=INK2, lw=1.3, ls="--", marker="o", mfc="white", ms=3.6,
                  label="collection-aware")]
    ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(0.09, 0.79),
              frameon=True, facecolor="white", edgecolor="none",
              fontsize=8, handlelength=2.2, labelspacing=0.6)
    fig.subplots_adjust(left=0.16, bottom=0.19, right=0.80, top=0.96)
    save(fig, "fig_scale_collection_aware")


def fig_hard(*, with_thumbnails=False):
    """Detection rate per attack family, with the attack shown on a CC0 Nouns
    token above each column."""
    e2 = pd.read_csv(LARGE / "sib_e2.csv")
    cols = [("compound", "Compound\n(2-3 edits)", "cmp1"), ("adaptive", "Adaptive\n(M7-evading)", "adapt"),
            ("img2img_0.3", "img2img\ns=0.3", "i2i30"), ("img2img_0.5", "img2img\ns=0.5", "i2i50"),
            ("img2img_0.7", "img2img\ns=0.7", "i2i70"), ("instruct_pix2pix", "Instruct-\nPix2Pix", "ip2p0")]
    rows = [("M7 Transf. pHash", "M7 TA-pHash"), ("M10 CLIP", "M10 CLIP"), ("M11 SSCD", "M11 SSCD"),
            ("Cascade T3=CLIP", "Cascade (CLIP)"), ("Cascade T3=SSCD", "Cascade (SSCD)")]
    M = np.array([[e2[(e2.system == r) & (e2.group == c)].rate.iloc[0] for c, *_ in cols] for r, _ in rows])
    N = [int(e2[(e2.system == rows[0][0]) & (e2.group == c)].n.iloc[0]) for c, *_ in cols]
    ex = BASE / "results" / "large" / "examples"

    fig = plt.figure(figsize=(COL_W, 2.9))
    left, right, top, bot = 0.27, 0.995, 0.97, 0.23
    thumb_h = 0.20
    grid_top = top - thumb_h - 0.008 if with_thumbnails else top
    W = (right - left) / len(cols)
    # thumbnails (original above the row-label column)
    def thumb(x0, name, label):
        a = fig.add_axes([x0 + W * 0.06, top - thumb_h, W * 0.88, thumb_h])
        a.imshow(Image.open(ex / f"ref_nouns_5__{name}.png").convert("RGB"))
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_color(GRID); s.set_linewidth(0.5)
        if label:
            a.set_title(label, fontsize=8, color=INK2, pad=1.5)
    if with_thumbnails:
        thumb(left - W * 1.02, "original", "original")
        for j, (_, _, name) in enumerate(cols):
            thumb(left + j * W, name, "")
    ax = fig.add_axes([left, bot, right - left, grid_top - bot])
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}".lstrip("0") if v < 1 else "1", ha="center", va="center",
                    fontsize=9, color="white" if v > 0.55 else INK)
    ax.set_xticks(range(len(cols)))
    labels = ["Comp.", "Adaptive", "img2img\n0.3", "img2img\n0.5",
              "img2img\n0.7", "IP2P"]
    ax.set_xticklabels([f"{lab}\n(n={n})" for lab, n in zip(labels, N)],
                       fontsize=7.5, linespacing=1.1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(["M7", "CLIP", "SSCD", "Cascade\n(CLIP)",
                        "Cascade\n(SSCD)"], fontsize=8)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True); ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(2.5, color=INK2, lw=0.6)  # single models | cascades
    save(fig, "fig_hard_attacks")


if __name__ == "__main__":
    style()
    fig_scale()
    fig_hard()
