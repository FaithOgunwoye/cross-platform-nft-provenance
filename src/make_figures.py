"""Publication-quality versions of paper Figs. 2-4 (camera-ready request:
"regenerated with improved readability and publication-quality formatting").

Same numbers as the submitted figures - read unchanged from the committed
results CSVs - only the presentation changes:
  * sized for one IEEE column (3.5 in), serif 8 pt text to match the body font
  * vector PDF (embedded TrueType fonts) + 600 dpi PNG
  * colour-blind-safe palette (validated categorical slots; single-hue
    sequential heatmap instead of red-yellow-green), plus shape/label
    redundancy so the figures survive grayscale printing
  * no in-figure titles (the LaTeX caption carries them)

Also the camera-ready E1/E2 figures (from results/large/).
Output: results/figures/*.{pdf,png}.  paper/ is not touched.
Run:  python src/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

BASE = Path(__file__).parent.parent
RES = BASE / "results"
OUT = RES / "figures"

COL_W = 3.5  # IEEE single column, inches
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
# categorical slots 1-4 (validated: adjacent + first-three all-pairs, light)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
# sequential blue ramp 100 -> 700
SEQ = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
       "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

SHORT = {"M1 Raw SHA-256": "M1 SHA-256", "M2 Canonical Hash": "M2 Canonical",
         "M3 Color Quant.": "M3 Colour-quant.", "M4 pHash": "M4 pHash",
         "M5 dHash": "M5 dHash", "M6 Edge + pHash": "M6 Edge+pHash",
         "M7 Transform-aware pHash": "M7 TA-pHash", "M8 SSIM": "M8 SSIM",
         "M9 ORB": "M9 ORB", "M10 CLIP": "M10 CLIP", "M11 SSCD": "M11 SSCD"}


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Nimbus Roman No9 L",
                       "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)
    print(f"  -> {OUT / name}.pdf/.png")


def fig2_prf1():
    """Horizontal grouped bars: model names read horizontally, no rotated labels."""
    df = pd.read_csv(RES / "metrics.csv")
    models = [SHORT.get(m, m) for m in df["Model"]][::-1]
    y = np.arange(len(models))
    h = 0.26
    fig, ax = plt.subplots(figsize=(COL_W, 3.3))
    for k, (col, c, lab) in enumerate([("Prec.", C1, "Precision"), ("Rec.", C2, "Recall"),
                                       ("F1", C3, "F1")]):
        v = df[col].values[::-1]
        ax.barh(y + (1 - k) * h, v, height=h * 0.86, color=c, label=lab, linewidth=0)
    ax.set_yticks(y, models)
    ax.set_xlim(0, 1.0)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Score")
    ax.set_ylim(-0.6, len(models) - 0.4)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.42, 1.0),
              handlelength=1.2, columnspacing=1.2, borderaxespad=0.2)
    save(fig, "fig2_model_prf1")


def fig3_heatmap():
    """Sequential single-hue heatmap; rows sorted easy -> hard (mean over models)."""
    piv = pd.read_csv(RES / "per_distortion.csv").set_index("distortion")
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    fig = plt.figure(figsize=(COL_W, 3.75))
    ax = fig.add_axes([0.255, 0.14, 0.64, 0.79])
    im = ax.imshow(piv.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    # 1 px surface gap between cells
    ax.set_xticks(np.arange(-0.5, piv.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, piv.shape[0]), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    # Model IDs refer to Table I, avoiding shrunken diagonal model names.
    ax.set_xticks(range(piv.shape[1]), [c.split()[0] for c in piv.columns],
                  rotation=90, fontsize=8)
    ax.set_yticks(range(piv.shape[0]), [d.replace("_", " ") for d in piv.index],
                  fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            txt = "1" if v == 1 else ("0" if v == 0 else f"{v:.2f}".lstrip("0"))
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.1,
                    color="white" if v >= 0.5 else INK)
    cb = fig.colorbar(im, cax=fig.add_axes([0.925, 0.14, 0.025, 0.79]),
                     ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=2)
    save(fig, "fig3_distortion_heatmap")


def fig4_scalability():
    df = pd.read_csv(RES / "scalability.csv")
    spec = {"CLIP exact (Flat)": (C1, "o", "-", "CLIP exact"),
            "CLIP ANN (HNSW)": (C3, "s", "--", "CLIP HNSW"),
            "pHash exact (popcount)": (C2, "^", "-", "pHash exact"),
            "pHash ANN (MIH)": (C4, "D", "--", "pHash MIH")}
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for name, g in df.groupby("index", sort=False):
        c, m, ls, lab = spec[name]
        ax.plot(g["n"], g["mean_ms"], ls, marker=m, ms=4, lw=1.4, color=c, label=lab,
                markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Index size (entries)")
    ax.set_ylabel("Mean query latency (ms)")
    ax.axhline(1.0, color=INK2, lw=0.6, ls=":")
    ax.text(df["n"].min() * 0.9, 1.12, "1 ms", fontsize=6.5, color=INK2, va="bottom")
    ax.grid(True, which="major", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(df["n"].min() * 0.75, df["n"].max() * 1.35)
    ax.legend(frameon=False, loc="upper left", handlelength=2.2)
    save(fig, "fig4_scalability")


def fig_e1_fpr_scaling():
    """Camera-ready E1: false-positive rate vs registry size with real distractors
    (a-priori thresholds, all queries), Wilson 95% bands."""
    e = pd.read_csv(RES / "large" / "e1_scaling.csv")
    e = e[(e.setting == "apriori") & (e.split == "all")]
    spec = [("M4 pHash", C1, "o", "-", "M4 / M7 pHash"),  # M7's FPR equals M4's at every size
            ("M10 CLIP", C2, "s", "-", "M10 CLIP"),
            ("M11 SSCD", C3, "^", "-", "M11 SSCD"),
            ("Cascade T3=CLIP", C4, "D", "--", "Cascade (T3 = CLIP)")]
    m7 = e[e.system == "M7 Transf. pHash"].set_index("index_size")["FPR"]
    m4 = e[e.system == "M4 pHash"].set_index("index_size")["FPR"]
    assert np.allclose(m4.values, m7.values), "M4/M7 FPR differ - relabel the figure"
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    for sys_, c, mk, ls, lab in spec:
        g = e[e.system == sys_].sort_values("index_size")
        ax.fill_between(g.index_size, g.FPR_lo, g.FPR_hi, color=c, alpha=0.15, lw=0)
        ax.plot(g.index_size, g.FPR, ls, marker=mk, ms=4, lw=1.4, color=c, label=lab,
                markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xlabel("Registry size (100 references + real NFT distractors)")
    ax.set_ylabel("False-positive rate")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, e.index_size.max() * 1.03)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.6), handlelength=2.0)
    save(fig, "fig_e1_fpr_scaling")


def fig_e2_hard():
    """Camera-ready E2: detection rate on hard attacks (a-priori thresholds, full index)."""
    h = pd.read_csv(RES / "large" / "e2_hard.csv")
    h = h[h.setting == "apriori"]
    systems = ["M4 pHash", "M7 Transf. pHash", "M10 CLIP", "M11 SSCD",
               "Cascade T3=CLIP", "Cascade T3=SSCD"]
    labels = ["M4 pHash", "M7 TA-pHash", "M10 CLIP", "M11 SSCD", "Casc. T3=CLIP", "Casc. T3=SSCD"]
    groups = ["compound", "adaptive", "img2img_0.3", "img2img_0.5", "img2img_0.7", "instruct_pix2pix"]
    glab = ["Compound", "Adaptive\n(M7-evading)", "img2img\ns = 0.3", "img2img\ns = 0.5",
            "img2img\ns = 0.7", "Instruct-\nPix2Pix"]
    piv = h.pivot_table(index="system", columns="group", values="rate").loc[systems, groups]
    n = h.pivot_table(index="system", columns="group", values="n").loc[systems[0], groups]
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    im = ax.imshow(piv.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(-0.5, piv.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, piv.shape[0]), minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(len(groups)), [f"{g}\n(n={int(k)})" for g, k in zip(glab, n.values)], fontsize=6.3)
    ax.set_yticks(range(len(systems)), labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.3,
                    color="white" if v >= 0.5 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, ticks=[0, 0.5, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.set_label("Correct-source detection rate", fontsize=7)
    save(fig, "fig_e2_hard_attacks")


def main():
    style()
    fig2_prf1()
    fig3_heatmap()
    fig4_scalability()
    if (RES / "large" / "e1_scaling.csv").exists():
        fig_e1_fpr_scaling()
        fig_e2_hard()


if __name__ == "__main__":
    main()
