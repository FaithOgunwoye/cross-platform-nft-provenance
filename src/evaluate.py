"""Evaluate all matching models on the D1/D2 benchmark.

For each model: index every reference image, screen every candidate
(distorted duplicates + genuine non-duplicates), and score the paper's
metrics — precision, recall, F1, FPR, FNR, Top-1 accuracy, mean query time.

Outputs (in results/):
  metrics.csv / table2.tex        overall model comparison   (paper Table II)
  per_distortion.csv / table3.tex recall by distortion class (paper Table III)
  distance_matrix.csv             raw best-match records for inspection
  plots/*.png                     summary figures

Run:  python src/evaluate.py [--data data] [--results results] [--no-clip]
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from models import build_models


def load_ground_truth(data: Path):
    rows = list(csv.DictReader(open(data / "ground_truth.csv")))
    for r in rows:
        d = data / ("candidates" if r["label"] == "duplicate" else "nonduplicates")
        r["path"] = d / r["candidate"]
    return rows


def evaluate_model(model, refs: dict, candidates: list):
    """Index refs, screen candidates; return per-candidate records."""
    index = {name: model.fingerprint(p) for name, p in refs.items()}
    records = []
    for cand in candidates:
        t0 = time.perf_counter()
        fp = model.fingerprint(cand["path"])
        best_name, best_dist = None, float("inf")
        for name, ref_fp in index.items():
            d = model.distance(ref_fp, fp)
            if d < best_dist:
                best_name, best_dist = name, d
        elapsed = time.perf_counter() - t0
        predicted_dup = best_dist <= model.threshold
        records.append({
            "model": model.name, "candidate": cand["candidate"],
            "true_source": cand["source"], "label": cand["label"],
            "distortion": cand["distortion"], "distortion_class": cand["distortion_class"],
            "best_match": best_name, "best_distance": round(best_dist, 5),
            "predicted": "duplicate" if predicted_dup else "non_duplicate",
            "top1_correct": int(predicted_dup and best_name == cand["source"]),
            "query_time_s": elapsed,
        })
    return records


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for model, g in df.groupby("model", sort=False):
        dup = g["label"] == "duplicate"
        pred = g["predicted"] == "duplicate"
        tp = int((dup & pred & (g["best_match"] == g["true_source"])).sum())
        fp = int((~dup & pred).sum() + (dup & pred & (g["best_match"] != g["true_source"])).sum())
        fn = int((dup & ~pred).sum())
        tn = int((~dup & ~pred).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        fnr = fn / (fn + tp) if fn + tp else 0.0
        top1 = g.loc[dup, "top1_correct"].mean() if dup.any() else 0.0
        out.append({"Model": model, "Prec.": prec, "Rec.": rec, "F1": f1,
                    "FPR": fpr, "FNR": fnr, "Top-1": top1,
                    "Time (ms)": g["query_time_s"].mean() * 1000})
    return pd.DataFrame(out)


def per_distortion(df: pd.DataFrame) -> pd.DataFrame:
    dup = df[df["label"] == "duplicate"]
    tbl = dup.assign(hit=(dup["predicted"] == "duplicate")
                     & (dup["best_match"] == dup["true_source"]))
    piv = tbl.pivot_table(index="distortion", columns="model", values="hit",
                          aggfunc="mean", sort=False)
    return piv


def to_latex(df: pd.DataFrame, path: Path, caption: str, label: str, floatfmt="%.2f"):
    body = df.to_latex(index=False, float_format=lambda x: floatfmt % x,
                       escape=True, column_format="l" + "r" * (len(df.columns) - 1))
    path.write_text(
        "\\begin{table}[htbp]\n\\caption{%s}\n\\label{%s}\n\\centering\n%s\\end{table}\n"
        % (caption, label, body))


def make_plots(summary, piv, plots_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5))
    s = summary.set_index("Model")[["Prec.", "Rec.", "F1"]]
    s.plot(kind="bar", ax=ax, width=0.75)
    ax.set_ylabel("score"); ax.set_ylim(0, 1.05)
    ax.set_title("Duplicate screening: precision / recall / F1 per model")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    fig.savefig(plots_dir / "model_prf1.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(piv.index)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="black")
    ax.set_title("Detection rate (correct-source recall) by distortion type")
    fig.colorbar(im, shrink=0.7)
    plt.tight_layout()
    fig.savefig(plots_dir / "distortion_heatmap.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    base = Path(__file__).parent.parent
    ap.add_argument("--data", default=str(base / "data"))
    ap.add_argument("--results", default=str(base / "results"))
    ap.add_argument("--no-clip", action="store_true", help="skip M10 (CLIP)")
    ap.add_argument("--only", default=None,
                    help="comma-separated model prefixes (e.g. 'M11'); other "
                         "models' records are reused from distance_matrix.csv")
    args = ap.parse_args()

    data, results = Path(args.data), Path(args.results)
    plots = results / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    refs = {p.name: p for p in sorted((data / "registry").glob("*.png"))}
    candidates = load_ground_truth(data)
    print(f"registry: {len(refs)} references; candidates: {len(candidates)}")

    only = [s.strip() for s in args.only.split(",")] if args.only else None
    models = build_models(include_clip=not args.no_clip)
    if only:
        models = [m for m in models if any(m.name.startswith(p) for p in only)]

    all_records = []
    for model in models:
        t0 = time.time()
        recs = evaluate_model(model, refs, candidates)
        all_records.extend(recs)
        hits = sum(r["top1_correct"] for r in recs if r["label"] == "duplicate")
        total = sum(1 for r in recs if r["label"] == "duplicate")
        print(f"{model.name:<28} top-1 {hits}/{total}   ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(all_records)
    if only:
        prev_path = results / "distance_matrix.csv"
        if prev_path.exists():
            prev = pd.read_csv(prev_path)
            prev = prev[~prev["model"].isin(df["model"].unique())]
            # keep canonical M1..M11 ordering after the merge
            df = pd.concat([prev, df], ignore_index=True)
            order = {m.name: i for i, m in enumerate(build_models())}
            df = df.sort_values("model", kind="stable",
                                key=lambda s: s.map(lambda n: order.get(n, 99)))
    df.to_csv(results / "distance_matrix.csv", index=False)

    summary = summarize(df)
    summary.to_csv(results / "metrics.csv", index=False)
    to_latex(summary, results / "table2.tex",
             "Evaluation metrics for duplicate and near-duplicate screening models.",
             "tab:metrics")

    piv = per_distortion(df)
    piv.to_csv(results / "per_distortion.csv")
    to_latex(piv.reset_index(), results / "table3.tex",
             "Correct-source detection rate by distortion type.", "tab:distortion")

    make_plots(summary, piv, plots)
    print("\n=== Table II (metrics) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nresults written to {results}/")


if __name__ == "__main__":
    main()
