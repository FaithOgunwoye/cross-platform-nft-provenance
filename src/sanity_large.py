"""Sanity checks for the camera-ready run (numbers quoted in results/large/RUN_NOTES.md).

Re-uses analyze_large.py's sets and search on the cached fingerprints:
  1. false positives on negatives at full index, a-priori thresholds: per query
     collection, same- vs cross-collection, and a direct recount of the
     collection labels (guards against a broken sibling split)
  2. E3b: Phunks original (2017) vs copy (2021) mint order; for pairs whose
     original is not in the match set, what the firing tier matched instead
Run:  python src/sanity_large.py
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

import analyze_large as A


def main():
    fp, gid = A.load()
    S = A.build_sets(fp, gid)
    col = S["col"]
    path = fp["path"]
    full_idx = np.concatenate([S["d1"], S["idx_l"]])
    q = np.concatenate([S["d2_dup"], S["d2_neg"], S["neg_l"]])
    is_dup = np.isin(q, S["d2_dup"])
    thr = {m: t for m, t in A.APRIORI.items() if t is not None}
    res = A.run_models(fp, q, full_idx, [m for m in A.MODELS if m in thr])

    print("== 1. negative-query false positives at full index (a-priori thresholds)")
    for m, r in res.items():
        pred = A.decide(r, m, thr[m]) & ~is_dup
        qc, mc = col[q[pred]], col[r["match"][pred]]
        same = int((qc == mc).sum())
        print(f"{m:18s} FP={pred.sum():4d} same-collection={same:4d} cross={pred.sum() - same:3d} "
              f"| unknown labels q={int((qc == 'unknown').sum())} m={int((mc == 'unknown').sum())}")
        top = Counter(qc).most_common(6)
        print("   by collection:", ", ".join(f"{c} {n}" for c, n in top))
        # nearest non-sibling distance for the FP queries, to show cross-collection matches are far
        if pred.any() and m in A.EMB:
            e = fp[A.EMB[m]]
            other = [1 - (e[full_idx[col[full_idx] != c]] @ e[g]).max() for g, c in zip(q[pred][:200], qc[:200])]
            print(f"   nearest cross-collection distance for FP queries: min {np.min(other):.3f} "
                  f"median {np.median(other):.3f} (threshold {thr[m]})")

    print("\n== 2. E3b (earliest observable matching mint)")
    pairs = json.loads((A.DATA / "copymints" / "pairs.json").read_text())
    ph = [p for p in pairs if p["case"] == "phunks"]
    ok = sum(1 for p in ph if p["orig_mint"]["timestamp"] < p["copy_mint"]["timestamp"])
    oy = Counter(__import__("time").gmtime(p["orig_mint"]["timestamp"]).tm_year for p in ph)
    cy = Counter(__import__("time").gmtime(p["copy_mint"]["timestamp"]).tm_year for p in ph)
    print(f"phunks: original minted before copy in {ok}/{len(ph)} pairs; orig years {dict(oy)}, copy years {dict(cy)}")
    rr = [p for p in pairs if p["case"] == "rrbayc"]
    ok = sum(1 for p in rr if p["orig_mint"]["timestamp"] < p["copy_mint"]["timestamp"])
    print(f"rrbayc: original minted before copy in {ok}/{len(rr)} pairs")

    # re-derive the match set of the firing tier for pairs where the original is missing
    import pandas as pd
    e3 = pd.read_csv(A.OUT / "e3_earliest_mint.csv")
    miss = e3[~e3["orig_in_matches"].astype(bool)]
    print(f"pairs with original NOT in match set: {len(miss)} "
          f"({miss.groupby(['case', 'fired_model']).size().to_dict()})")
    kinds = Counter()
    for _, row in miss.iterrows():
        p = next(x for x in pairs if x["case"] == row["case"] and x["copy_token"] == row["copy_token"])
        g = gid["data/copymints/copy/" + p["copy_file"]]
        cands = np.array([gid[k] for k in gid if k.startswith("data/copymints/") and gid[k] != g] +
                         list(full_idx), int)
        ref = fp["ph"][cands, 0]
        D = A.POP[np.bitwise_xor(ref, fp["ph"][g, 0])].sum(-1) / 256.0  # M4 = variant 0 only
        for h in cands[D <= thr["M4 pHash"]]:
            ps = str(path[h])
            kinds["other Punk original" if "/original/phunks_" in ps else
                  "other Phunk copy" if "/copy/phunks_" in ps else ps.split("/")[1]] += 1
    print("what the M4 tier matched instead (entries):", dict(kinds))
    print("earliest_is_original:", e3.groupby("case")["earliest_is_original"]
          .apply(lambda s: f"{int(s.fillna(False).astype(bool).sum())}/{len(s)}").to_dict())


if __name__ == "__main__":
    main()
