"""Secondary analyses over the recorded evaluation distances.

Reads results/distance_matrix.csv (one best-match record per model x candidate)
and produces, without re-running any model:

  1. per-category detection rates      -> results/per_category.csv
  2. threshold sensitivity (F1 vs t)   -> results/threshold_sweep.csv
  3. layered-pipeline simulation       -> results/layered_pipeline.csv

Run:  python src/analyze_extra.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent
RESULTS = BASE / "results"

# distance <= threshold => predicted duplicate (must mirror models.py)
THRESHOLDS = {
    "M1 Raw SHA-256": 0.0, "M2 Canonical Hash": 0.0, "M3 Color Quant.": 0.0,
    "M4 pHash": 0.15, "M5 dHash": 0.15, "M6 Edge + pHash": 0.22,
    "M7 Transform-aware pHash": 0.15, "M8 SSIM": 0.25, "M9 ORB": 0.85,
    "M10 CLIP": 0.10, "M11 SSCD": 0.50,
}


def category_of(source: str, registry: dict) -> str:
    return registry.get(source, {}).get("category", "unknown")


def per_category(df: pd.DataFrame, registry: dict) -> pd.DataFrame:
    dup = df[df["label"] == "duplicate"].copy()
    dup["category"] = dup["true_source"].map(lambda s: category_of(s, registry))
    dup["hit"] = (dup["predicted"] == "duplicate") & (dup["best_match"] == dup["true_source"])
    piv = dup.pivot_table(index="category", columns="model", values="hit", aggfunc="mean")
    order = [m for m in THRESHOLDS if m in piv.columns]
    return piv[order]


def f1_at(g: pd.DataFrame, t: float) -> float:
    dup = g["label"] == "duplicate"
    pred = g["best_distance"] <= t
    tp = int((dup & pred & (g["best_match"] == g["true_source"])).sum())
    fp = int((~dup & pred).sum() + (dup & pred & (g["best_match"] != g["true_source"])).sum())
    fn = int((dup & ~pred).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def threshold_sweep(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grid = np.round(np.arange(0.0, 1.01, 0.01), 2)
    for model, g in df.groupby("model", sort=False):
        if THRESHOLDS.get(model) == 0.0:
            continue  # exact hashes have no meaningful threshold
        f1s = [f1_at(g, t) for t in grid]
        best_i = int(np.argmax(f1s))
        used_t = THRESHOLDS[model]
        used_f1 = f1_at(g, used_t)
        # width of the threshold band keeping F1 within 95% of its optimum
        near = [t for t, f in zip(grid, f1s) if f >= 0.95 * f1s[best_i]]
        rows.append({"model": model, "used_t": used_t, "used_F1": round(used_f1, 3),
                     "opt_t": grid[best_i], "opt_F1": round(f1s[best_i], 3),
                     "stable_band": f"[{min(near):.2f}, {max(near):.2f}]"})
    return pd.DataFrame(rows)


def layered(df: pd.DataFrame, times_ms: dict,
            layer3=("M10 CLIP",)) -> pd.DataFrame:
    """Simulate the layered pipeline from recorded per-model results."""
    wide_d = df.pivot(index="candidate", columns="model", values="best_distance")
    wide_m = df.pivot(index="candidate", columns="model", values="best_match")
    meta = df[df["model"] == "M1 Raw SHA-256"].set_index("candidate")[
        ["label", "true_source", "distortion_class"]]

    # default layer 3 = CLIP only: adding M8 SSIM was measured to add false
    # positives (9 vs 0) while raising recall by only 0.004
    LAYERS = [
        ["M1 Raw SHA-256", "M2 Canonical Hash"],
        ["M4 pHash", "M7 Transform-aware pHash"],
        list(layer3),
    ]
    recs = []
    for cand in wide_d.index:
        cost, verdict, match, stopped = 0.0, "non_duplicate", None, 3
        for li, layer in enumerate(LAYERS):
            fired = []
            for m in layer:
                cost += times_ms[m]
                if wide_d.loc[cand, m] <= THRESHOLDS[m]:
                    fired.append(m)
            if fired:
                # take the match of the strongest (lowest-distance) firing model
                best = min(fired, key=lambda m: wide_d.loc[cand, m])
                verdict, match, stopped = "duplicate", wide_m.loc[cand, best], li
                break
        row = meta.loc[cand]
        recs.append({"candidate": cand, "label": row["label"],
                     "distortion_class": row["distortion_class"],
                     "verdict": verdict, "match": match,
                     "correct": int((verdict == "duplicate") == (row["label"] == "duplicate")
                                    and (verdict != "duplicate" or match == row["true_source"])),
                     "stopped_at_layer": stopped + 1 if verdict == "duplicate" else 3,
                     "cost_ms": round(cost, 1)})
    return pd.DataFrame(recs)


def main():
    df = pd.read_csv(RESULTS / "distance_matrix.csv")
    registry = {e["local_file"]: e for e in
                json.loads((BASE / "data" / "registry.json").read_text())}
    times_ms = df.groupby("model")["query_time_s"].mean().mul(1000).to_dict()

    cat = per_category(df, registry)
    cat.to_csv(RESULTS / "per_category.csv")
    print("=== detection rate by visual category ===")
    print(cat.round(2).to_string())

    sweep = threshold_sweep(df)
    sweep.to_csv(RESULTS / "threshold_sweep.csv", index=False)
    print("\n=== threshold sensitivity ===")
    print(sweep.to_string(index=False))

    compositions = {
        "CLIP": ("M10 CLIP",),
        "SSIM": ("M8 SSIM",),
        "SSIM + CLIP": ("M8 SSIM", "M10 CLIP"),
        "SSCD": ("M11 SSCD",),
        "SSCD + CLIP": ("M11 SSCD", "M10 CLIP"),
    }
    comp_rows = []
    for name, l3 in compositions.items():
        if any(m not in df["model"].unique() for m in l3):
            continue
        lay = layered(df, times_ms, layer3=l3)
        if name == "CLIP":  # canonical pipeline: keep the per-candidate dump
            lay.to_csv(RESULTS / "layered_pipeline.csv", index=False)
            canonical = lay
        dup = lay["label"] == "duplicate"
        tp = int((dup & (lay["verdict"] == "duplicate") & lay["correct"].astype(bool)).sum())
        fp = int((~dup & (lay["verdict"] == "duplicate")).sum()
                 + (dup & (lay["verdict"] == "duplicate") & ~lay["correct"].astype(bool)).sum())
        fn = int((dup & (lay["verdict"] != "duplicate")).sum())
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        comp_rows.append({"layer3": name, "prec": round(prec, 3),
                          "rec": round(rec, 3),
                          "F1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0,
                          "FP": fp, "mean_ms": round(lay["cost_ms"].mean(), 0)})
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(RESULTS / "layered_compositions.csv", index=False)
    print("\n=== layered pipeline (M1/M2 -> M4/M7 -> layer 3) ===")
    print(comp.to_string(index=False))
    print(f"(vs {sum(times_ms[m] for m in THRESHOLDS if m in times_ms):.0f} ms running everything)")
    lay, dup = canonical, canonical["label"] == "duplicate"
    print(lay.groupby("stopped_at_layer")["candidate"].count().rename("candidates").to_string())
    print("\ncanonical (CLIP) recall by distortion class:")
    print(lay[dup].groupby("distortion_class")["correct"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
