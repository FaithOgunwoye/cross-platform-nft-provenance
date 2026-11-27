"""Scalability experiment: retrieval latency and accuracy at registry scale.

The main evaluation indexes 100 references, small enough for exhaustive
comparison.  Production screening must search marketplace-scale registries, so
this experiment measures how query latency and top-1 retrieval accuracy behave
when the index is grown with distractors up to 2x10^5 entries, comparing
exact (brute-force) search against sublinear ANN indexes:

  CLIP  512-d unit embeddings : FAISS IndexFlatIP (exact)  vs  HNSW (M=32)
  pHash 256-bit hashes        : NumPy popcount / FAISS IndexBinaryFlat (exact)
                                vs  multi-index hashing (IndexBinaryMultiHash)

Distractors are synthetic but distribution-matched to the real registry:
CLIP distractors are drawn from a per-dimension Gaussian fitted to the real
reference embeddings and re-normalised to the unit sphere; pHash distractors
are Bernoulli bit vectors matched to the empirical per-bit frequencies of the
real reference hashes.  Queries are always the REAL 840 benchmark candidates;
only the index population is varied, so the experiment is a controlled
ablation over (index type x corpus size) with everything else held fixed.

Outputs: results/scalability.csv, paper/scalability.png
Run:     python src/scalability.py
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from models import M4PHash, M10CLIP, HASH_SIZE, NBITS

BASE = Path(__file__).parent.parent
DATA, RESULTS, PAPER = BASE / "data", BASE / "results", BASE / "paper"
SCALES = [100, 1_000, 10_000, 100_000, 200_000]
SEED = 42
HNSW_M, HNSW_EFC, HNSW_EFS = 32, 200, 64
MIH_NHASH, MIH_B, MIH_NFLIP = 4, 16, 2

POP = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)


def load_benchmark():
    rows = list(csv.DictReader(open(DATA / "ground_truth.csv")))
    for r in rows:
        d = DATA / ("candidates" if r["label"] == "duplicate" else "nonduplicates")
        r["path"] = d / r["candidate"]
    refs = {p.name: p for p in sorted((DATA / "registry").glob("*.png"))}
    return refs, rows


def real_fingerprints(refs, cands):
    """CLIP embeddings + packed pHash bits for all real images (cached)."""
    cache = RESULTS / "scalability_fps.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return (z["ref_emb"], z["cand_emb"], z["ref_bits"], z["cand_bits"],
                list(z["ref_names"]))
    clip, ph = M10CLIP(), M4PHash()

    def phash_bits(path):
        h = ph.fingerprint(path).hash.flatten()
        return np.packbits(h.astype(np.uint8))

    ref_names = list(refs)
    print(f"fingerprinting {len(refs)} references ...")
    ref_emb = np.stack([clip.fingerprint(refs[n]) for n in ref_names])
    ref_bits = np.stack([phash_bits(refs[n]) for n in ref_names])
    print(f"fingerprinting {len(cands)} candidates ...")
    cand_emb = np.stack([clip.fingerprint(c["path"]) for c in cands])
    cand_bits = np.stack([phash_bits(c["path"]) for c in cands])
    np.savez_compressed(cache, ref_emb=ref_emb, cand_emb=cand_emb,
                        ref_bits=ref_bits, cand_bits=cand_bits,
                        ref_names=np.array(ref_names))
    return ref_emb, cand_emb, ref_bits, cand_bits, ref_names


def make_distractors(ref_emb, ref_bits, n, rng):
    """Distribution-matched synthetic index entries (see module docstring)."""
    mu, sd = ref_emb.mean(0), ref_emb.std(0) + 1e-8
    emb = rng.normal(mu, sd, size=(n, ref_emb.shape[1])).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    p = np.unpackbits(ref_bits, axis=1).mean(0)
    bits = np.packbits(rng.random((n, NBITS)) < p, axis=1)
    return emb, bits


def timed_queries(search_one, queries):
    """Per-query latency (single-query regime, as in interactive screening)."""
    tops, times = [], []
    for q in queries:
        t0 = time.perf_counter()
        tops.append(search_one(q))
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    return np.array(tops), times.mean() * 1e3, np.percentile(times, 95) * 1e3


def recall_at_1(tops, cands, ref_names, n_refs):
    dup = [i for i, c in enumerate(cands) if c["label"] == "duplicate"]
    hit = sum(1 for i in dup
              if tops[i] < n_refs and ref_names[tops[i]] == cands[i]["source"])
    return hit / len(dup)


def run():
    import faiss
    rng = np.random.default_rng(SEED)
    refs, cands = load_benchmark()
    ref_emb, cand_emb, ref_bits, cand_bits, ref_names = real_fingerprints(refs, cands)
    n_refs, dim = ref_emb.shape
    max_extra = max(SCALES) - n_refs
    x_emb, x_bits = make_distractors(ref_emb, ref_bits, max_extra, rng)

    rows = []
    for scale in SCALES:
        extra = scale - n_refs
        emb_db = np.vstack([ref_emb, x_emb[:extra]])
        bits_db = np.vstack([ref_bits, x_bits[:extra]])

        # --- CLIP exact: FAISS flat inner-product ------------------------
        t0 = time.time()
        flat = faiss.IndexFlatIP(dim)
        flat.add(emb_db)
        build = time.time() - t0
        tops, ms, p95 = timed_queries(
            lambda q: flat.search(q[None, :], 1)[1][0, 0], cand_emb)
        rows.append(dict(index="CLIP exact (Flat)", n=scale, build_s=build,
                         mean_ms=ms, p95_ms=p95,
                         recall1=recall_at_1(tops, cands, ref_names, n_refs),
                         mem_mb=emb_db.nbytes / 2**20))
        exact_tops = tops

        # --- CLIP ANN: HNSW ----------------------------------------------
        t0 = time.time()
        hnsw = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = HNSW_EFC
        hnsw.hnsw.efSearch = HNSW_EFS
        hnsw.add(emb_db)
        build = time.time() - t0
        tops, ms, p95 = timed_queries(
            lambda q: hnsw.search(q[None, :], 1)[1][0, 0], cand_emb)
        rows.append(dict(index="CLIP ANN (HNSW)", n=scale, build_s=build,
                         mean_ms=ms, p95_ms=p95,
                         recall1=recall_at_1(tops, cands, ref_names, n_refs),
                         mem_mb=emb_db.nbytes / 2**20,
                         agree_exact=float((tops == exact_tops).mean())))

        # --- pHash exact: vectorised popcount ----------------------------
        tops, ms, p95 = timed_queries(
            lambda q: int(POP[np.bitwise_xor(bits_db, q)].sum(1).argmin()),
            cand_bits)
        rows.append(dict(index="pHash exact (popcount)", n=scale, build_s=0.0,
                         mean_ms=ms, p95_ms=p95,
                         recall1=recall_at_1(tops, cands, ref_names, n_refs),
                         mem_mb=bits_db.nbytes / 2**20))

        # --- pHash ANN: multi-index hashing -------------------------------
        t0 = time.time()
        mih = faiss.IndexBinaryMultiHash(NBITS, MIH_NHASH, MIH_B)
        mih.nflip = MIH_NFLIP
        mih.add(bits_db)
        build = time.time() - t0
        tops, ms, p95 = timed_queries(
            lambda q: int(mih.search(q[None, :], 1)[1][0, 0]), cand_bits)
        rows.append(dict(index="pHash ANN (MIH)", n=scale, build_s=build,
                         mean_ms=ms, p95_ms=p95,
                         recall1=recall_at_1(tops, cands, ref_names, n_refs),
                         mem_mb=bits_db.nbytes / 2**20))

        for r in rows[-4:]:
            print(f"N={scale:>7,}  {r['index']:<24} "
                  f"{r['mean_ms']:8.3f} ms  R@1={r['recall1']:.3f}")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "scalability.csv", index=False)
    plot(df)
    print(f"\nwritten: {RESULTS/'scalability.csv'}, {PAPER/'scalability.png'}")


def plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # categorical slots 1-4 (validated); yellow/aqua relieved by direct labels
    colors = {"CLIP exact (Flat)": "#2a78d6", "CLIP ANN (HNSW)": "#1baf7a",
              "pHash exact (popcount)": "#eda100", "pHash ANN (MIH)": "#008300"}
    markers = {"CLIP exact (Flat)": "o", "CLIP ANN (HNSW)": "s",
               "pHash exact (popcount)": "^", "pHash ANN (MIH)": "D"}
    styles = {"CLIP exact (Flat)": "-", "CLIP ANN (HNSW)": "--",
              "pHash exact (popcount)": "-", "pHash ANN (MIH)": "--"}

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for name, g in df.groupby("index", sort=False):
        ax.plot(g["n"], g["mean_ms"], styles[name], marker=markers[name],
                ms=5, lw=1.8, color=colors[name], label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("index size (entries)")
    ax.set_ylabel("mean query latency (ms)")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(df["n"].min() * 0.8, df["n"].max() * 1.6)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    plt.tight_layout()
    fig.savefig(PAPER / "scalability.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run()
