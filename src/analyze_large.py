"""Camera-ready analysis on cached fingerprints (CPU / NumPy only).

Reads results/large/fingerprints.npz (from fingerprint_all.py) and runs:

  E1  registry scaling with REAL distractors (R1/R4): index = 100 D1 refs +
      k real NFTs (k = 0 .. all); queries = 800 D2 duplicates + all negatives
      (40 original + held-out real NFTs).  Per model and for the cascade:
      P/R/F1/FPR with Wilson 95% CIs, and false positives split into
      same-collection ("stylistic sibling") vs cross-collection.
  E2  hard attacks (R2/R3): compound edits, adaptive M7-evasion, generative
      (img2img by strength, InstructPix2Pix) at full index size.
  E3  real copymints (R2/R5): RR/BAYC -> BAYC and CryptoPhunks -> CryptoPunks.
      (a) retrieval of the true original; (b) earliest-observable-matching-
      mint resolution when both original and copy are indexed.
  CAL thresholds calibrated on a disjoint split (half of the D1 sources and
      half of the negatives) and reported on the other half, alongside the
      paper's a-priori thresholds; plus a top-1/top-2 margin rule for the
      learned tier.

Outputs: results/large/*.csv, results/large/summary.md, results/large/plots/
Run:     python src/analyze_large.py
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent
DATA = BASE / "data"
OUT = BASE / "results" / "large"
SEED = 42

APRIORI = {"M4 pHash": 0.15, "M7 Transf. pHash": 0.15, "M10 CLIP": 0.10,
           "M11 SSCD": 0.50, "M12 DINOv2": None}
EMB = {"M10 CLIP": "clip", "M11 SSCD": "sscd", "M12 DINOv2": "dino"}
POP = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint16)


def is_exact(model: str) -> bool:
    """M1/M2 exact-hash models.  (Not startswith("M1"): that also matches M10/M11/M12.)"""
    return model.split()[0] in ("M1", "M2")


# ------------------------------------------------------------------ utils
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load():
    z = np.load(OUT / "fingerprints.npz", allow_pickle=False)
    fp = {k: z[k] for k in z.files}
    for k in ("clip", "sscd", "dino"):
        if k in fp:
            fp[k] = fp[k].astype(np.float32)
    path = list(fp["path"])
    gid = {p: i for i, p in enumerate(path)}
    return fp, gid


def collections(fp):
    """collection name per global id (for sibling analysis)."""
    col = {}
    reg = {e["local_file"]: e["collection"] for e in json.loads((DATA / "registry.json").read_text())}
    neg = {e["local_file"]: e["collection"] for e in json.loads((DATA / "nonduplicates.json").read_text())}
    lp = DATA / "large" / "registry_large.json"
    lar = {e["local_file"]: e["collection"] for e in json.loads(lp.read_text())} if lp.exists() else {}
    src = {r["candidate"]: r["source"] for r in csv.DictReader(open(DATA / "ground_truth.csv"))}
    out = []
    for p in fp["path"]:
        name = p.split("/")[-1]
        if p.startswith("data/registry/"):
            c = reg.get(name)
        elif p.startswith("data/nonduplicates/"):
            c = neg.get(name)
        elif p.startswith("data/candidates/"):
            c = reg.get(src.get(name, ""))
        elif p.startswith("data/large/"):
            c = lar.get(name)
        elif p.startswith("data/hard/"):
            c = reg.get(name.split("__")[0] + ".png")
        elif p.startswith("data/copymints/"):
            c = ("BAYC" if "rrbayc" in name else "CryptoPunks") + ("" if "/original/" in p else " (copy)")
        else:
            c = None
        out.append(c or "unknown")
    return np.array(out)


# ------------------------------------------------------------------ search
def topk(fp, model, q, idx, chunk=512):
    """top-2 (index position, distance) of queries q against index ids idx."""
    n = len(idx)
    t1 = np.zeros(len(q), int); d1 = np.ones(len(q)); t2 = np.zeros(len(q), int); d2 = np.ones(len(q))
    if n == 0:
        return t1 - 1, d1, t2 - 1, d2, 0.0
    t0 = time.perf_counter()
    if model in ("M1 Raw SHA-256", "M2 Canonical Hash"):
        key = fp["m1" if model.split()[0] == "M1" else "m2"]
        first = {}
        for pos, g in enumerate(idx):
            first.setdefault(key[g], pos)
        for i, g in enumerate(q):
            pos = first.get(key[g])
            t1[i], d1[i] = (pos, 0.0) if pos is not None else (0, 1.0)
            t2[i] = -1
    else:
        for s in range(0, len(q), chunk):
            qs = q[s:s + chunk]
            if model == "M4 pHash":
                D = POP[np.bitwise_xor(fp["ph"][qs, 0][:, None, :], fp["ph"][idx, 0][None])].sum(-1) / 256.0
            elif model == "M7 Transf. pHash":
                ref = fp["ph"][idx, 0][None]
                D = np.full((len(qs), n), 1.0)
                for v in range(16):
                    D = np.minimum(D, POP[np.bitwise_xor(fp["ph"][qs, v][:, None, :], ref)].sum(-1) / 256.0)
            else:
                e = fp[EMB[model]]
                D = 1.0 - e[qs] @ e[idx].T
            if n >= 2:
                part = np.argpartition(D, 1, axis=1)[:, :2]
                pd_ = np.take_along_axis(D, part, 1)
                o = np.argsort(pd_, 1)
                part = np.take_along_axis(part, o, 1); pd_ = np.take_along_axis(pd_, o, 1)
                t1[s:s + chunk], d1[s:s + chunk] = part[:, 0], pd_[:, 0]
                t2[s:s + chunk], d2[s:s + chunk] = part[:, 1], pd_[:, 1]
            else:
                t1[s:s + chunk], d1[s:s + chunk] = 0, D[:, 0]; t2[s:s + chunk] = -1
    ms = (time.perf_counter() - t0) / max(1, len(q)) * 1000
    return t1, d1, t2, d2, ms


MODELS = ["M1 Raw SHA-256", "M2 Canonical Hash", "M4 pHash", "M7 Transf. pHash",
          "M10 CLIP", "M11 SSCD", "M12 DINOv2"]


def run_models(fp, q, idx, models):
    res = {}
    for m in models:
        if m in EMB and EMB[m] not in fp:
            continue
        t1, d1, t2, d2, ms = topk(fp, m, q, idx)
        res[m] = dict(match=np.where(t1 >= 0, idx[np.clip(t1, 0, None)], -1), d1=d1,
                      d2=d2, ms=ms)
    return res


def decide(r, model, thr, margin=0.0):
    if is_exact(model):
        return r["d1"] <= 0.0
    ok = r["d1"] <= thr
    if margin > 0:
        ok &= (r["d2"] - r["d1"]) >= margin
    return ok


def cascade(res, thr, tier3=("M10 CLIP",), margin=0.0):
    """Paper cascade: T1 M1|M2 -> T2 M4|M7 -> T3 (learned).  Returns
    (fired bool, matched id, tier) per query."""
    nq = len(next(iter(res.values()))["d1"])
    fired = np.zeros(nq, bool); match = np.full(nq, -1); tier = np.zeros(nq, int)
    tiers = [("M1 Raw SHA-256", "M2 Canonical Hash"), ("M4 pHash", "M7 Transf. pHash"), tuple(tier3)]
    for ti, ms in enumerate(tiers, 1):
        best_d = np.full(nq, np.inf); best_m = np.full(nq, -1)
        for m in ms:
            if m not in res:
                continue
            ok = decide(res[m], m, thr.get(m), margin if ti == 3 else 0.0) & ~fired
            better = ok & (res[m]["d1"] < best_d)
            best_d[better] = res[m]["d1"][better]; best_m[better] = res[m]["match"][better]
        new = best_m >= 0
        fired |= new; match[new] = best_m[new]; tier[new] = ti
    return fired, match, tier


def score(pred, match, is_dup, src, qcol, mcol):
    tp = int((pred & is_dup & (match == src)).sum())
    fp_wrong = int((pred & is_dup & (match != src)).sum())
    fp_neg = pred & ~is_dup
    fp_sib = int((fp_neg & (qcol == mcol)).sum()); fp_x = int(fp_neg.sum()) - fp_sib
    fp = fp_wrong + int(fp_neg.sum())
    fn = int((~pred & is_dup).sum())
    tn = int((~pred & ~is_dup).sum())
    P = wilson(tp, tp + fp); R = wilson(tp, tp + fn); F = wilson(int(fp_neg.sum()), int((~is_dup).sum()))
    f1 = 2 * P[0] * R[0] / (P[0] + R[0]) if tp else 0.0
    return dict(TP=tp, FP=fp, FP_wrong_src=fp_wrong, FP_neg_sibling=fp_sib, FP_neg_cross=fp_x,
                FN=fn, TN=tn, prec=P[0], prec_lo=P[1], prec_hi=P[2], rec=R[0], rec_lo=R[1],
                rec_hi=R[2], F1=f1, FPR=F[0], FPR_lo=F[1], FPR_hi=F[2])


# ------------------------------------------------------------------ setup
def build_sets(fp, gid):
    sets = fp["set"]
    rng = np.random.default_rng(SEED)
    d1 = np.where(sets == "d1_ref")[0]
    # large: drop exact canonical copies of any D1/D2 image, then split
    # 80% index distractors / 20% held-out negatives (per collection)
    bench_m2 = set(fp["m2"][np.isin(sets, ["d1_ref", "d2_neg"])])
    large = np.array([i for i in np.where(sets == "large")[0] if fp["m2"][i] not in bench_m2], int)
    col = collections(fp)
    idx_l, neg_l = [], []
    for c in sorted(set(col[large])):
        ids = large[col[large] == c]
        ids = ids[rng.permutation(len(ids))]
        k = int(round(len(ids) * 0.2))
        neg_l += list(ids[:k]); idx_l += list(ids[k:])
    idx_l = np.array(idx_l, int)[rng.permutation(len(idx_l))]  # nested scaling order
    neg_l = np.array(sorted(neg_l), int)

    gt = list(csv.DictReader(open(DATA / "ground_truth.csv")))
    src_of = {}
    for r in gt:
        if r["label"] == "duplicate":
            src_of[gid["data/candidates/" + r["candidate"]]] = gid["data/registry/" + r["source"]]
    d2_dup = np.array(sorted(src_of), int)
    d2_neg = np.where(sets == "d2_neg")[0]

    hard = []
    hp = DATA / "hard" / "ground_truth_hard.csv"
    if hp.exists():
        for r in csv.DictReader(open(hp)):
            if r["candidate"]:
                g = gid["data/hard/images/" + r["candidate"]]
                src_of[g] = gid["data/registry/" + r["source"]]
                hard.append((g, r["distortion_class"], r["distortion"], r["param"]))
    return dict(d1=d1, idx_l=idx_l, neg_l=neg_l, d2_dup=d2_dup, d2_neg=d2_neg,
                hard=hard, src_of=src_of, col=col)


def calib_split(S):
    """Half of the D1 sources (and their D2 copies) + half the negatives."""
    rng = np.random.default_rng(SEED + 1)
    d1 = S["d1"][rng.permutation(len(S["d1"]))]
    cal_src = set(d1[: len(d1) // 2])
    negs = np.concatenate([S["d2_neg"], S["neg_l"]])
    negs = negs[rng.permutation(len(negs))]
    cal_neg = set(negs[: len(negs) // 2])
    return cal_src, cal_neg


def calibrate(res, q, is_dup, src, qcol, mcol, cal_mask, min_prec=0.99):
    """Per model: threshold maximising F1 subject to precision >= min_prec on
    the calibration half.  Also the best top1/top2 margin for learned models."""
    thr, marg, rows = {}, {}, []
    for m, r in res.items():
        if is_exact(m):
            continue
        best = (-1, None)
        for t in np.round(np.arange(0.0, 0.81, 0.005), 3):
            pred = (r["d1"] <= t) & cal_mask
            s = score(pred[cal_mask], r["match"][cal_mask], is_dup[cal_mask], src[cal_mask],
                      qcol[cal_mask], mcol[m][cal_mask])
            if s["prec"] >= min_prec and s["F1"] > best[0]:
                best = (s["F1"], t)
        thr[m] = best[1] if best[1] is not None else 0.0
        bm = (-1, 0.0)
        if m in EMB:
            for mg in np.round(np.arange(0.0, 0.2, 0.005), 3):
                pred = decide(r, m, thr[m] + 0.2, mg) & cal_mask  # loosened thr + margin
                s = score(pred[cal_mask], r["match"][cal_mask], is_dup[cal_mask], src[cal_mask],
                          qcol[cal_mask], mcol[m][cal_mask])
                if s["prec"] >= min_prec and s["F1"] > bm[0]:
                    bm = (s["F1"], mg)
        marg[m] = bm[1]
        rows.append(dict(model=m, apriori_thr=APRIORI.get(m), cal_thr=thr[m], cal_margin=marg[m]))
    return thr, marg, pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    fp, gid = load()
    S = build_sets(fp, gid)
    col = S["col"]
    print(f"D1 refs {len(S['d1'])}, real distractors {len(S['idx_l'])}, "
          f"held-out real negatives {len(S['neg_l'])}, D2 dups {len(S['d2_dup'])}, "
          f"D2 negs {len(S['d2_neg'])}, D3 hard {len(S['hard'])}")

    q = np.concatenate([S["d2_dup"], S["d2_neg"], S["neg_l"]])
    is_dup = np.isin(q, S["d2_dup"])
    src = np.array([S["src_of"].get(g, -2) for g in q])
    qcol = col[q]
    cal_src, cal_neg = calib_split(S)
    cal_mask = np.array([(src[i] in cal_src) if is_dup[i] else (q[i] in cal_neg) for i in range(len(q))])
    test_mask = ~cal_mask

    full_idx = np.concatenate([S["d1"], S["idx_l"]])
    res_full = run_models(fp, q, full_idx, MODELS)
    mcol_full = {m: col[np.clip(r["match"], 0, None)] for m, r in res_full.items()}
    cal_thr, cal_marg, cal_df = calibrate(res_full, q, is_dup, src, qcol, mcol_full, cal_mask)
    cal_df.to_csv(OUT / "calibration.csv", index=False)
    print("\n=== calibrated thresholds (precision>=0.99 on calibration half, full index) ===")
    print(cal_df.to_string(index=False))

    settings = {"apriori": ({m: t for m, t in APRIORI.items() if t is not None}, {}),
                "calibrated": (cal_thr, cal_marg)}

    # ---------------- E1: scaling with real distractors -----------------
    n_l = len(S["idx_l"])
    sizes = sorted(set([0, 500, 1000, 2000, 4000, n_l]) & set(range(0, n_l + 1)) | {n_l})
    rows = []
    for k in sizes:
        idx = np.concatenate([S["d1"], S["idx_l"][:k]])
        res = res_full if k == n_l else run_models(fp, q, idx, MODELS)
        mcol = {m: col[np.clip(r["match"], 0, None)] for m, r in res.items()}
        for sname, (thr, marg) in settings.items():
            ev = [("all", np.ones(len(q), bool))] + ([("test", test_mask)] if sname == "calibrated" else [])
            if sname == "apriori":  # paper-comparable query set: 800 D2 dups + 40 D2 negatives
                ev.append(("d2only", np.isin(q, np.concatenate([S["d2_dup"], S["d2_neg"]]))))
            for split, mask in ev:
                for m, r in res.items():
                    if thr.get(m) is None and not is_exact(m):
                        continue
                    pred = decide(r, m, thr.get(m))
                    s = score(pred[mask], r["match"][mask], is_dup[mask], src[mask], qcol[mask], mcol[m][mask])
                    rows.append(dict(index_size=len(idx), setting=sname, split=split, system=m,
                                     search_ms=r["ms"], **s))
                for t3 in (("M10 CLIP",), ("M11 SSCD",), ("M12 DINOv2",)):
                    if t3[0] not in res or thr.get(t3[0]) is None:
                        continue
                    for mg_name, mg in (("", 0.0), ("+margin", marg.get(t3[0], 0.0))):
                        if mg_name and not mg:
                            continue
                        fired, match, _ = cascade(res, thr, t3, mg)
                        mc = col[np.clip(match, 0, None)]
                        s = score(fired[mask], match[mask], is_dup[mask], src[mask], qcol[mask], mc[mask])
                        rows.append(dict(index_size=len(idx), setting=sname, split=split,
                                         system=f"Cascade T3={t3[0].split()[1]}{mg_name}", search_ms=np.nan, **s))
        print(f"  E1 index size {len(idx)} done")
    e1 = pd.DataFrame(rows)
    e1.to_csv(OUT / "e1_scaling.csv", index=False)

    # ---------------- E2: hard attacks at full index ---------------------
    if S["hard"]:
        hq = np.array([h[0] for h in S["hard"]], int)
        hmeta = pd.DataFrame(S["hard"], columns=["gid", "cls", "distortion", "param"])
        hsrc = np.array([S["src_of"][g] for g in hq])
        hres = run_models(fp, hq, full_idx, MODELS)
        rows = []
        for sname, (thr, marg) in settings.items():
            systems = {m: (decide(r, m, thr.get(m)), r["match"]) for m, r in hres.items()
                       if thr.get(m) is not None or is_exact(m)}
            for t3 in ("M10 CLIP", "M11 SSCD", "M12 DINOv2"):
                if t3 in hres and thr.get(t3) is not None:
                    f, mt, _ = cascade(hres, thr, (t3,))
                    systems[f"Cascade T3={t3.split()[1]}"] = (f, mt)
            for name, (pred, match) in systems.items():
                hit = pred & (match == hsrc)
                g = hmeta.assign(hit=hit)
                g["group"] = np.where(g["cls"] == "generative", g["distortion"], g["cls"])
                for grp, gg in g.groupby("group"):
                    k, n = int(gg["hit"].sum()), len(gg)
                    p, lo, hi = wilson(k, n)
                    rows.append(dict(setting=sname, system=name, group=grp, n=n, detected=k,
                                     rate=p, lo=lo, hi=hi))
        e2 = pd.DataFrame(rows)
        e2.to_csv(OUT / "e2_hard.csv", index=False)
        # adaptive evasion budget distribution
        ad = hmeta[hmeta["cls"] == "adaptive"]["param"].str.extract(r"mag=([\d.]+)")[0].astype(float)
        pd.DataFrame({"mag": ad}).describe().to_csv(OUT / "e2_adaptive_budget.csv")

    # ---------------- E3: real copymints ---------------------------------
    pj = DATA / "copymints" / "pairs.json"
    if pj.exists():
        pairs = json.loads(pj.read_text())
        orig = {p["orig_file"]: gid["data/copymints/original/" + p["orig_file"]] for p in pairs}
        copy = {p["copy_file"]: gid["data/copymints/copy/" + p["copy_file"]] for p in pairs}
        cq = np.array([copy[p["copy_file"]] for p in pairs])
        csrc = np.array([orig[p["orig_file"]] for p in pairs])
        case = np.array([p["case"] for p in pairs])
        o_ids = np.array(sorted(set(orig.values())))
        # The large registry / D1 may hold the very same token as a copymint
        # original (random BAYC ids); keep only the copymints/original entry so
        # retrieving the identical image is not scored as a wrong match.
        cm_tok = {(p["orig_contract"].lower(), int(p["orig_token"])) for p in pairs} | \
                 {(p["copy_contract"].lower(), int(p["copy_token"])) for p in pairs}
        tok_of = {}
        for f in ("registry.json", "nonduplicates.json", "large/registry_large.json"):
            fpth = DATA / f
            if fpth.exists():
                sub = "data/" + ("large/images/" if f.startswith("large") else
                                 "registry/" if f == "registry.json" else "nonduplicates/")
                for e in json.loads(fpth.read_text()):
                    if sub + e["local_file"] in gid:
                        tok_of[gid[sub + e["local_file"]]] = (e["contract"].lower(), int(e["token_id"]))
        overlap = [g for g in full_idx if tok_of.get(g) in cm_tok]
        print(f"E3: {len(overlap)} index entries are the same token as a copymint pair -> excluded from E3 index")
        full_idx_e3 = np.array([g for g in full_idx if tok_of.get(g) not in cm_tok], int)
        idx_a = np.concatenate([full_idx_e3, o_ids])
        # (a) retrieval of the original with the copy as query
        ra = run_models(fp, cq, idx_a, MODELS)
        rows = []
        for sname, (thr, marg) in settings.items():
            systems = {m: (decide(r, m, thr.get(m)), r["match"]) for m, r in ra.items()
                       if thr.get(m) is not None or is_exact(m)}
            for t3 in ("M10 CLIP", "M11 SSCD", "M12 DINOv2"):
                if t3 in ra and thr.get(t3) is not None:
                    f, mt, _ = cascade(ra, thr, (t3,))
                    systems[f"Cascade T3={t3.split()[1]}"] = (f, mt)
            for name, (pred, match) in systems.items():
                for cs in sorted(set(case)):
                    msk = case == cs
                    k = int((pred & (match == csrc))[msk].sum()); n = int(msk.sum())
                    wrong = int((pred & (match != csrc))[msk].sum())
                    p, lo, hi = wilson(k, n)
                    rows.append(dict(setting=sname, system=name, case=cs, n=n, detected=k,
                                     wrong_match=wrong, rate=p, lo=lo, hi=hi))
        pd.DataFrame(rows).to_csv(OUT / "e3_copymint_retrieval.csv", index=False)

        # (b) earliest observable matching mint: index holds originals AND
        # copies; every index entry within threshold of the cascade's firing
        # model counts as a match; resolver = argmin mint timestamp.
        ts = {}
        for p in pairs:
            om, cm = p.get("orig_mint") or {}, p.get("copy_mint") or {}
            ts[orig[p["orig_file"]]] = om.get("timestamp")
            ts[copy[p["copy_file"]]] = cm.get("timestamp")
        c_ids = np.array(sorted(set(copy.values())))
        idx_b = np.concatenate([full_idx_e3, o_ids, c_ids])
        thr = settings["apriori"][0]
        rows = []
        for i, p in enumerate(pairs):
            g = cq[i]
            cand = idx_b[idx_b != g]  # the query copy must not match itself
            matched, tier_model = set(), ""
            for m in ("M1 Raw SHA-256", "M2 Canonical Hash", "M4 pHash", "M7 Transf. pHash", "M10 CLIP"):
                # all entries within threshold, not only top-1
                if is_exact(m):
                    key = fp["m1" if m.split()[0] == "M1" else "m2"]
                    hits = cand[key[cand] == key[g]]
                elif m in EMB:
                    e = fp[EMB[m]]; D = 1 - e[cand] @ e[g]
                    hits = cand[D <= thr[m]]
                else:
                    ref = fp["ph"][cand, 0]
                    vs = range(16) if m.startswith("M7") else [0]
                    D = np.min([POP[np.bitwise_xor(ref, fp["ph"][g, v])].sum(-1) / 256.0 for v in vs], 0)
                    hits = cand[D <= thr[m]]
                if len(hits):
                    matched = set(hits.tolist()); tier_model = m
                    break
            matched.discard(-1)
            # resolver compares the query's own mint with all matched entries
            with_ts = [(ts.get(h), h) for h in matched | {g} if ts.get(h)]
            earliest = min(with_ts)[1] if with_ts else None
            rows.append(dict(case=p["case"], copy_token=p["copy_token"], orig_token=p["orig_token"],
                             n_matches=len(matched), orig_in_matches=csrc[i] in matched,
                             orig_ts=ts.get(csrc[i]), copy_ts=ts.get(g),
                             earliest_is_original=(earliest == csrc[i]) if earliest is not None else None,
                             fired_model=tier_model if matched else ""))
        e3b = pd.DataFrame(rows)
        e3b.to_csv(OUT / "e3_earliest_mint.csv", index=False)

    write_summary(e1, OUT)
    print(f"\nall outputs in {OUT}")


def write_summary(e1, out):
    lines = ["# Camera-ready results summary (auto-generated by analyze_large.py)", ""]
    full = e1["index_size"].max()
    for sname in ("apriori", "calibrated"):
        split = "all" if sname == "apriori" else "test"
        t = e1[(e1.index_size == full) & (e1.setting == sname) & (e1.split == split)]
        lines += [f"## E1 at full index ({full} entries), {sname} thresholds, split={split}", "",
                  t[["system", "prec", "prec_lo", "rec", "F1", "FPR", "FPR_lo", "FPR_hi", "FP_neg_sibling",
                     "FP_neg_cross", "FP_wrong_src"]].round(3).to_markdown(index=False), ""]
    lines += ["## E1 FPR vs index size (a-priori thresholds)", "",
              e1[(e1.setting == "apriori") & (e1.split == "all")].pivot_table(index="system", columns="index_size", values="FPR")
              .round(3).to_markdown(), ""]
    for f in ("e2_hard.csv", "e3_copymint_retrieval.csv", "e3_earliest_mint.csv"):
        p = out / f
        if p.exists():
            d = pd.read_csv(p)
            if f == "e2_hard.csv":
                d = d[d.setting == "apriori"].pivot_table(index="system", columns="group", values="rate").round(2)
                lines += ["## E2 detection rate on hard attacks (a-priori)", "", d.to_markdown(), ""]
            elif f == "e3_copymint_retrieval.csv":
                d = d[d.setting == "apriori"].pivot_table(index="system", columns="case", values="rate").round(3)
                lines += ["## E3a copymint retrieval of the true original (a-priori)", "", d.to_markdown(), ""]
            else:
                lines += ["## E3b earliest observable matching mint", "",
                          d.groupby("case")[["orig_in_matches", "earliest_is_original"]]
                          .agg(lambda s: f"{s.dropna().astype(bool).sum()}/{s.notna().sum()}").to_markdown(), ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
