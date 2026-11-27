"""Generate the introduction figure: two visually identical NFT images whose
SHA-256 digests differ completely because a single pixel was modified.

Uses a D1 reference and its one-pixel D2 candidate (BAYC #10), verifies that
exactly one pixel differs, and renders them side by side with their digests.

Output: paper/Image/hash_demo.png
Run:    python src/make_hash_demo.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).parent.parent
ORIG = BASE / "data" / "registry" / "ref_bored_ape_yacht_club_10.png"
MOD = BASE / "data" / "candidates" / "ref_bored_ape_yacht_club_10__one_pixel.png"
OUT = BASE / "paper" / "Image" / "hash_demo.png"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h_orig = hashlib.sha256(ORIG.read_bytes()).hexdigest()
    h_mod = hashlib.sha256(MOD.read_bytes()).hexdigest()
    a = np.array(Image.open(ORIG).convert("RGB"))
    b = np.array(Image.open(MOD).convert("RGB"))
    n_diff = int((a != b).any(axis=2).sum())
    assert n_diff == 1, f"expected exactly one differing pixel, got {n_diff}"

    def wrap(h):
        return h[:16] + "\n..." + h[-16:]

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.4))
    panels = [(axes[0], a, "(a) original media", h_orig),
              (axes[1], b, "(b) one pixel modified", h_mod)]
    for ax, img, title, h in panels:
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=8, pad=4)
        ax.text(0.5, -0.06, "SHA-256:", transform=ax.transAxes, ha="center",
                va="top", fontsize=8, color="#0b0b0b")
        ax.text(0.5, -0.13, wrap(h), transform=ax.transAxes, ha="center",
                va="top", fontsize=8, family="monospace", color="#0b0b0b")
    plt.subplots_adjust(wspace=0.06, top=0.92, bottom=0.22, left=0.01, right=0.99)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600)
    print(f"written {OUT}  (pixels differing: {n_diff})")


if __name__ == "__main__":
    main()
