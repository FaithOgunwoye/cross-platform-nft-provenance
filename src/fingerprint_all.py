"""Fingerprint EVERY image of the camera-ready experiments once (GPU step).

Covers: D1 references, D2 candidates, the 40 original non-duplicates, the
large real registry (data/large), the hard set D3 (data/hard) and the
real copymint pairs (data/copymints).  All downstream analysis
(src/analyze_large.py) runs on these cached fingerprints with NumPy only, so
it can be re-run on a CPU laptop after the GPU machine is wiped.

Stored per image (results/large/fingerprints.npz):
  m1   raw-file SHA-256                     (as models.M1)
  m2   grayscale-256 canonical SHA-256      (as models.M2)
  ph   16 x 256-bit pHash variants, packed  (variant 0 == M4 pHash; all 16 = M7)
  clip OpenCLIP ViT-B/32 laion2b, L2-normed (as models.M10), float16
  sscd SSCD disc_mixup 512-d, L2-normed     (as models.M11), float16
  dino DINOv2 ViT-B/14 CLS, L2-normed       (new learned baseline M12), float16
plus results/large/fp_timing.json (per-image fingerprint cost, GPU-batched
and single-image CPU-style, for the latency discussion).

Run:  python src/fingerprint_all.py [--batch 128] [--workers 16]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))

BASE = Path(__file__).parent.parent
DATA = BASE / "data"
OUT = BASE / "results" / "large"


def collect():
    """(path, set) for every image; set tags let the analysis select subsets."""
    items = []
    for p in sorted((DATA / "registry").glob("*.png")):
        items.append((p, "d1_ref"))
    for r in csv.DictReader(open(DATA / "ground_truth.csv")):
        d = "candidates" if r["label"] == "duplicate" else "nonduplicates"
        items.append((DATA / d / r["candidate"], "d2_dup" if d == "candidates" else "d2_neg"))
    for p in sorted((DATA / "large" / "images").glob("*.png")):
        items.append((p, "large"))
    gt = DATA / "hard" / "ground_truth_hard.csv"
    if gt.exists():
        for r in csv.DictReader(open(gt)):
            if r["candidate"]:
                items.append((DATA / "hard" / "images" / r["candidate"], "d3_" + r["distortion_class"]))
    pj = DATA / "copymints" / "pairs.json"
    if pj.exists():
        seen = set()
        for p in json.loads(pj.read_text()):
            for sub, key in (("original", "orig_file"), ("copy", "copy_file")):
                f = DATA / "copymints" / sub / p[key]
                if f not in seen:
                    seen.add(f); items.append((f, "cm_" + sub))
    return items


def cpu_fp(path_str):
    """M1, M2 and the 16 pHash variants (same code path as models.py)."""
    import imagehash
    from models import M7TransformAwarePHash, HASH_SIZE
    path = Path(path_str)
    raw = hashlib.sha256(path.read_bytes()).hexdigest()
    img = Image.open(path).convert("RGB")
    can = hashlib.sha256(ImageOps.grayscale(img).resize((256, 256), Image.LANCZOS)
                         .tobytes()).hexdigest()
    m7 = M7TransformAwarePHash()
    ph = [np.packbits(imagehash.phash(v, hash_size=HASH_SIZE).hash.flatten().astype(np.uint8))
          for v in m7._variants(img)]
    return raw, can, np.stack(ph)


class Embedder:
    def __init__(self, kind, dev):
        import torch
        self.torch, self.dev, self.kind = torch, dev, kind
        if kind == "clip":
            import open_clip
            m, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
            self.fwd = lambda x: m.encode_image(x)
        elif kind == "sscd":
            from models import M11SSCD
            m, pre, _ = M11SSCD._ensure()
            self.fwd = lambda x: m(x)
        elif kind == "dino":
            from torchvision import transforms
            m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            pre = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            self.fwd = lambda x: m(x)  # CLS embedding
        self.m, self.pre = m.eval().to(dev), pre

    def __call__(self, paths, batch):
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(paths), batch):
                x = torch.stack([self.pre(Image.open(p).convert("RGB")) for p in paths[i:i + batch]])
                e = self.fwd(x.to(self.dev)).float()
                e = e / e.norm(dim=1, keepdim=True)
                out.append(e.cpu().numpy().astype(np.float16))
        return np.concatenate(out)

    def single_ms(self, paths, n=30, dev="cpu"):
        """Per-image latency, batch=1, to compare with the paper's CPU numbers."""
        torch = self.torch
        self.m.to(dev)
        t = []
        with torch.no_grad():
            for p in paths[:n]:
                t0 = time.perf_counter()
                self.fwd(self.pre(Image.open(p).convert("RGB")).unsqueeze(0).to(dev))
                if dev == "cuda":
                    torch.cuda.synchronize()
                t.append(time.perf_counter() - t0)
        self.m.to(self.dev)
        return float(np.median(t) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip", default="", help="comma list of clip,sscd,dino to skip")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev} ({torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'})")

    items = collect()
    # de-duplicate paths (a file may appear in two sets only by mistake)
    uniq, tags = {}, []
    for p, s in items:
        if p not in uniq:
            uniq[p] = len(uniq); tags.append(s)
    paths = list(uniq)
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} missing files, e.g. {missing[:3]}")
    print(f"{len(paths)} images; sets: " +
          ", ".join(f"{s}={tags.count(s)}" for s in sorted(set(tags))))

    timing = {}
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        res = list(pool.map(cpu_fp, [str(p) for p in paths], chunksize=32))
    timing["cpu_hashes_total_s"] = time.time() - t0
    m1 = np.array([r[0] for r in res]); m2 = np.array([r[1] for r in res])
    ph = np.stack([r[2] for r in res])
    print(f"hashes done in {timing['cpu_hashes_total_s']:.0f}s")

    emb = {}
    for kind in ("clip", "sscd", "dino"):
        if kind in args.skip:
            continue
        e = Embedder(kind, dev)
        t0 = time.time()
        emb[kind] = e(paths, args.batch)
        timing[f"{kind}_batched_ms_per_img"] = (time.time() - t0) / len(paths) * 1000
        timing[f"{kind}_single_ms_{dev}"] = e.single_ms(paths, dev=dev)
        timing[f"{kind}_single_ms_cpu"] = e.single_ms(paths, n=15, dev="cpu")
        print(f"{kind}: {timing[f'{kind}_batched_ms_per_img']:.2f} ms/img batched")
        del e; torch.cuda.empty_cache() if dev == "cuda" else None

    rel = np.array([str(p.relative_to(BASE)).replace("\\", "/") for p in paths])
    np.savez_compressed(OUT / "fingerprints.npz", path=rel, set=np.array(tags),
                        m1=m1, m2=m2, ph=ph, **emb)
    timing["device"] = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    (OUT / "fp_timing.json").write_text(json.dumps(timing, indent=1))
    print(f"-> {OUT/'fingerprints.npz'}")


if __name__ == "__main__":
    main()
