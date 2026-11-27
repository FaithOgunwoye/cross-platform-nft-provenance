"""Buyer-side NFT duplicate-risk screening with who-minted-first resolution.

Given a candidate image (the media of an NFT someone is offering you), this
tool runs the layered pipeline recommended in the paper:

  Layer 1  M1 raw SHA-256 + M2 canonical hash    -> exact / canonical duplicate
  Layer 2  M4 pHash + M7 transform-aware pHash   -> near-duplicate & evasion
  Layer 3  M8 SSIM (+ M10 CLIP if available)     -> uncertain cases

and prints an explainable duplicate-risk report.  If one or more registry
entries match, their on-chain provenance (chain, contract, token id, mint
block and timestamp) is compared and the EARLIEST mint is reported as the
presumed original — "who minted first".  Matching entries minted later are
flagged as probable remints.

Run:  python src/screen.py path/to/candidate.png [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models import (M1RawSHA256, M2CanonicalHash, M4PHash,
                    M7TransformAwarePHash, M8SSIM, M10CLIP)

RISK_ORDER = {"EXACT_DUPLICATE": 3, "NEAR_DUPLICATE": 2, "POSSIBLE_MATCH": 1, "NO_MATCH": 0}


def load_registry(data: Path):
    entries = json.loads((data / "registry.json").read_text())
    return {e["local_file"]: e for e in entries}


def screen(candidate: Path, data: Path, use_clip: bool = False) -> dict:
    registry = load_registry(data)
    ref_paths = {name: data / "registry" / name for name in registry}

    # layer 3 prefers CLIP: benchmarked at zero false positives, whereas SSIM
    # (the fallback when CLIP deps are absent) adds FPs on collection-mates
    layers = [
        ("exact",   [M1RawSHA256(), M2CanonicalHash()],       "EXACT_DUPLICATE"),
        ("percept", [M4PHash(), M7TransformAwarePHash()],     "NEAR_DUPLICATE"),
        ("deep",    [M10CLIP()] if use_clip else [M8SSIM()],  "POSSIBLE_MATCH"),
    ]

    matches = {}   # ref name -> evidence list
    verdict = "NO_MATCH"
    for layer_name, models, layer_verdict in layers:
        for model in models:
            cand_fp = model.fingerprint(candidate)
            for name, rp in ref_paths.items():
                d = model.distance(model.fingerprint(rp), cand_fp)
                if d <= model.threshold:
                    matches.setdefault(name, []).append(
                        {"model": model.name, "distance": round(d, 5), "layer": layer_name})
        if matches:
            verdict = layer_verdict
            break  # layered short-circuit: stop at the cheapest conclusive layer

    matched = []
    for name, evidence in matches.items():
        e = dict(registry[name])
        e["evidence"] = evidence
        matched.append(e)

    # who-minted-first: order matches by on-chain mint time
    presumed_original, remints = None, []
    dated = [m for m in matched if m.get("mint_timestamp")]
    if dated:
        dated.sort(key=lambda m: m["mint_timestamp"])
        presumed_original = dated[0]
        remints = dated[1:] + [m for m in matched if not m.get("mint_timestamp")]
    elif matched:
        presumed_original = matched[0]

    return {
        "candidate": str(candidate),
        "verdict": verdict,
        "risk": {"EXACT_DUPLICATE": "HIGH", "NEAR_DUPLICATE": "HIGH",
                 "POSSIBLE_MATCH": "MEDIUM", "NO_MATCH": "LOW"}[verdict],
        "matches": matched,
        "presumed_original": presumed_original,
        "probable_remints": remints,
    }


def fmt_ts(ts):
    if not ts:
        return "unknown"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def print_report(report: dict):
    print("=" * 66)
    print("NFT DUPLICATE-RISK REPORT")
    print("=" * 66)
    print(f"candidate : {report['candidate']}")
    print(f"verdict   : {report['verdict']}   (risk: {report['risk']})")
    if not report["matches"]:
        print("\nNo indexed NFT media matches this image. The candidate is not a")
        print("known duplicate of anything in the reference registry.")
        return
    orig = report["presumed_original"]
    print(f"\nmatched registry entries: {len(report['matches'])}")
    print("\n--- presumed ORIGINAL (earliest mint) ---")
    print(f"  collection : {orig['collection']}  (token #{orig['token_id']})")
    print(f"  chain      : {orig['chain']}")
    print(f"  contract   : {orig['contract']}")
    print(f"  minted     : {fmt_ts(orig.get('mint_timestamp'))}  block {orig.get('mint_block')}")
    print(f"  mint tx    : {orig.get('mint_tx', '')}")
    print(f"  evidence   : " + "; ".join(
        f"{e['model']} d={e['distance']}" for e in orig["evidence"]))
    for r in report["probable_remints"]:
        print("\n--- probable REMINT (minted later) ---")
        print(f"  collection : {r['collection']}  (token #{r['token_id']}, {r['chain']})")
        print(f"  minted     : {fmt_ts(r.get('mint_timestamp'))}")
    print("\nNOTE: this report is technical evidence of media duplication and")
    print("temporal mint order; it does not establish legal authorship.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", help="image file to screen")
    ap.add_argument("--data", default=str(Path(__file__).parent.parent / "data"))
    ap.add_argument("--clip", action="store_true", help="also use M10 CLIP in layer 3")
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = ap.parse_args()

    report = screen(Path(args.candidate), Path(args.data), use_clip=args.clip)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
