"""Collection-aware screening (post-GPU analysis on cached fingerprints).

Finding from E1: at registry scale every false positive on a genuine NFT is a
*same-collection sibling* (0 cross-collection FPs for every model).  A buyer
always knows the candidate's own contract, and a copymint of collection X is,
by definition, minted outside X.  Collection-aware screening therefore
searches each candidate only against entries from OTHER collections.

Assumptions (stated in the paper):
  - collection = dedicated collection contract (Art Blocks: contract+project);
    shared minting contracts (marketplace storefronts, Manifold/Zora shared
    contracts) must NOT be treated as a collection, since copymints are often
    minted there; our registry contains none.
  - D2/D3 duplicates are synthetic copies with no contract; they are treated
    as foreign to every indexed collection (no exclusion), i.e. copymints.
  - Genuine negatives exclude their own collection.

Everything else (thresholds a-priori, cascade tiers, scoring) is identical to
analyze_large.py, so rows are directly comparable with e1_scaling.csv /
e3_earliest_mint.csv.  No parameter is tuned.

Outputs: results/large/sib_e1.csv, sib_e2.csv, sib_e3b.csv, sib_summary.md
Run:     python src/analyze_sibling.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
import analyze_large as A

OUT = A.OUT
T3S = ("M10 CLIP", "M11 SSCD", "M12 DINOv2")


def dist_matrix(fp, model, q, idx):
    if A.is_exact(model):
        key = fp["m1" if model.split()[0] == "M1" else "m2"]
        return (key[q][:, None] != key[idx][None]).astype(np.float32)
    if model == "M4 pHash":
        return A.POP[np.bitwise_xor(fp["ph"][q, 0][:, None, :], fp["ph"][idx, 0][None])].sum(-1) / 256.0
    if model == "M7 Transf. pHash":
        ref = fp["ph"][idx, 0][None]
        D = np.ones((len(q), len(idx)))
        for v in range(16):
            D = np.minimum(D, A.POP[np.bitwise_xor(fp["ph"][q, v][:, None, :], ref)].sum(-1) / 256.0)
        return D
    e = fp[A.EMB[model]]
    return 1.0 - e[q] @ e[idx].T


def run(fp, q, idx, qown, icol, exclude, models=A.MODELS, chunk=256):
    """top-1/top-2 per model; if exclude, entries of the query's own
    collection are removed from its search (qown == '' -> nothing removed)."""
    res = {}
    for m in models:
        if m in A.EMB and A.EMB[m] not in fp:
            continue
        t1 = np.empty(len(q), int); d1 = np.empty(len(q)); d2 = np.empty(len(q))
        for s in range(0, len(q), chunk):
            D = dist_matrix(fp, m, q[s:s + chunk], idx).astype(np.float64)
            if exclude:
                D[qown[s:s + chunk][:, None] == icol[None]] = np.inf
            o = np.argsort(D, 1)[:, :2]
            t1[s:s + chunk] = o[:, 0]
            d1[s:s + chunk] = np.take_along_axis(D, o[:, :1], 1)[:, 0]
            d2[s:s + chunk] = np.take_along_axis(D, o[:, 1:2], 1)[:, 0]
        d1 = np.minimum(d1, 1.0); d2 = np.minimum(d2, 1.0)
        res[m] = dict(match=idx[t1], d1=d1, d2=d2, ms=np.nan)
    return res


def systems(res, thr):
    out = {m: (A.decide(r, m, thr.get(m)), r["match"]) for m, r in res.items()
           if thr.get(m) is not None or A.is_exact(m)}
    for t3 in T3S:
        if t3 in res and thr.get(t3) is not None:
            f, mt, _ = A.cascade(res, thr, (t3,))
            out[f"Cascade T3={t3.split()[1]}"] = (f, mt)
    return out


def main():
    fp, gid = A.load()
    S = A.build_sets(fp, gid)
    col = S["col"]
    thr = {m: t for m, t in A.APRIORI.items() if t is not None}

    q = np.concatenate([S["d2_dup"], S["d2_neg"], S["neg_l"]])
    is_dup = np.isin(q, S["d2_dup"])
    src = np.array([S["src_of"].get(g, -2) for g in q])
    qcol = col[q]
    qown = np.where(is_dup, "", qcol)          # copies are foreign to every collection

    # ---------------- E1 with / without collection-aware exclusion ----------
    n_l = len(S["idx_l"])
    sizes = sorted({0, 500, 1000, 2000, n_l})
    rows = []
    for k in sizes:
        idx = np.concatenate([S["d1"], S["idx_l"][:k]])
        for excl in (False, True):
            res = run(fp, q, idx, qown, col[idx], excl)
            for name, (pred, match) in systems(res, thr).items():
                s = A.score(pred, match, is_dup, src, qcol, col[np.clip(match, 0, None)])
                rows.append(dict(index_size=len(idx), collection_aware=excl, system=name, **s))
        print(f"  index {len(idx)} done")
    e1 = pd.DataFrame(rows)
    e1.to_csv(OUT / "sib_e1.csv", index=False)

    # ---------------- E2: robustness is unchanged (copies are foreign) ------
    # Recomputed anyway as a check: with no exclusion for copies, detection
    # must equal e2_hard.csv exactly.
    full_idx = np.concatenate([S["d1"], S["idx_l"]])
    hq = np.array([h[0] for h in S["hard"]], int)
    hsrc = np.array([S["src_of"][g] for g in hq])
    hgrp = np.array([h[2] if h[1] == "generative" else h[1] for h in S["hard"]])
    hres = run(fp, hq, full_idx, np.full(len(hq), ""), col[full_idx], True)
    rows = []
    for name, (pred, match) in systems(hres, thr).items():
        for g in sorted(set(hgrp)):
            m = hgrp == g
            k = int((pred & (match == hsrc))[m].sum()); n = int(m.sum())
            p, lo, hi = A.wilson(k, n)
            rows.append(dict(system=name, group=g, n=n, detected=k, rate=p, lo=lo, hi=hi))
    pd.DataFrame(rows).to_csv(OUT / "sib_e2.csv", index=False)

    # ---------------- E3b: earliest observable matching mint ---------------
    pairs = json.loads((A.DATA / "copymints" / "pairs.json").read_text())
    orig = {p["orig_file"]: gid["data/copymints/original/" + p["orig_file"]] for p in pairs}
    copy = {p["copy_file"]: gid["data/copymints/copy/" + p["copy_file"]] for p in pairs}
    # same token-overlap exclusion as analyze_large (E3 fix 6)
    cm_tok = {(p["orig_contract"].lower(), int(p["orig_token"])) for p in pairs} | \
             {(p["copy_contract"].lower(), int(p["copy_token"])) for p in pairs}
    tok_of = {}
    for f, sub in (("registry.json", "data/registry/"), ("nonduplicates.json", "data/nonduplicates/"),
                   ("large/registry_large.json", "data/large/images/")):
        for e in json.loads((A.DATA / f).read_text()):
            if sub + e["local_file"] in gid:
                tok_of[gid[sub + e["local_file"]]] = (e["contract"].lower(), int(e["token_id"]))
    idx_e3 = np.array([g for g in full_idx if tok_of.get(g) not in cm_tok], int)
    ts, colx = {}, col.copy()
    for p in pairs:
        ts[orig[p["orig_file"]]] = (p.get("orig_mint") or {}).get("timestamp")
        ts[copy[p["copy_file"]]] = (p.get("copy_mint") or {}).get("timestamp")
        colx[copy[p["copy_file"]]] = "RR/BAYC" if p["case"] == "rrbayc" else "CryptoPhunks"
        colx[orig[p["orig_file"]]] = "Bored Ape Yacht Club" if p["case"] == "rrbayc" else "CryptoPunks"
    idx_b = np.concatenate([idx_e3, np.array(sorted(set(orig.values()))), np.array(sorted(set(copy.values())))])
    rows = []
    for excl in (False, True):
        for p in pairs:
            g, o = copy[p["copy_file"]], orig[p["orig_file"]]
            cand = idx_b[idx_b != g]
            if excl:
                cand = cand[colx[cand] != colx[g]]  # a copy's own contract is not evidence
            matched, tier = set(), ""
            for m in ("M1 Raw SHA-256", "M2 Canonical Hash", "M4 pHash", "M7 Transf. pHash", "M10 CLIP"):
                D = dist_matrix(fp, m, np.array([g]), cand)[0]
                hits = cand[D <= (0.0 if A.is_exact(m) else thr[m])]
                if len(hits):
                    matched, tier = set(hits.tolist()), m
                    break
            with_ts = [(ts.get(h), h) for h in matched | {g} if ts.get(h)]
            earliest = min(with_ts)[1] if with_ts else None
            rows.append(dict(collection_aware=excl, case=p["case"], copy_token=p["copy_token"],
                             n_matches=len(matched), orig_in_matches=o in matched,
                             earliest_is_original=earliest == o,
                             earliest_in_orig_collection=(earliest is not None and colx[earliest] == colx[o]), fired_model=tier))
    e3 = pd.DataFrame(rows)
    e3.to_csv(OUT / "sib_e3b.csv", index=False)

    # ---------------- summary ----------------------------------------------
    L = ["# Collection-aware screening (auto-generated by analyze_sibling.py)", ""]
    full = e1.index_size.max()
    for excl in (False, True):
        t = e1[(e1.index_size == full) & (e1.collection_aware == excl)]
        L += [f"## E1 full index ({full}), a-priori thresholds, collection_aware={excl}", "",
              t[["system", "prec", "prec_lo", "rec", "F1", "FPR", "FPR_lo", "FPR_hi", "FP_neg_sibling",
                 "FP_neg_cross", "FP_wrong_src"]].round(3).to_markdown(index=False), ""]
    L += ["## FPR vs index size (a-priori)", "",
          e1.pivot_table(index=["system", "collection_aware"], columns="index_size", values="FPR")
          .round(3).to_markdown(), ""]
    L += ["## E2 detection (unchanged by construction; check vs e2_hard.csv)", "",
          pd.read_csv(OUT / "sib_e2.csv").pivot_table(index="system", columns="group", values="rate")
          .round(2).to_markdown(), ""]
    L += ["## E3b earliest observable matching mint", "",
          e3.groupby(["collection_aware", "case"])[["orig_in_matches", "earliest_is_original", "earliest_in_orig_collection"]]
          .agg(lambda s: f"{int(s.sum())}/{len(s)}").to_markdown(), ""]
    (OUT / "sib_summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
