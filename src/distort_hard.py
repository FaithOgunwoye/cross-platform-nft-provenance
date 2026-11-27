"""Hard candidate set D3 (camera-ready experiment, reviewers R2/R3).

The original D2 applies ONE controlled distortion per candidate.  D3 adds the
three attack families the reviewers asked about, all derived from the same
100 D1 references so results are directly comparable with D2:

  compound     2-3 chained edits drawn from an extended pool (arbitrary-angle
               rotation, perspective warp, hue shift, 10-20% crop, blur,
               sticker overlay, flip, JPEG, resize, border)          [CPU]
  adaptive     threshold-aware evasion: for each reference, search the
               smallest geometric/colour edit budget that pushes the
               transform-aware pHash (M7) past its decision threshold, i.e.
               an attacker who knows tier 2 of the cascade              [CPU]
  generative   Stable Diffusion img2img at strengths 0.3/0.5/0.7 and
               InstructPix2Pix semantic edits ("watercolor", "add a hat",
               ...) - re-synthesised copies that share no pixels      [GPU]

Output: data/hard/images/*, data/hard/ground_truth_hard.csv
  label = duplicate for every row (derived from a D1 source); `param` holds
  the strength / budget so detection can be reported as a function of it.

Run:  python src/distort_hard.py --parts compound,adaptive      (CPU)
      python src/distort_hard.py --parts generative             (GPU)
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).parent))

BASE = Path(__file__).parent.parent
REG = BASE / "data" / "registry"
OUT = BASE / "data" / "hard"
IMG = OUT / "images"
GT = OUT / "ground_truth_hard.csv"
FIELDS = ["candidate", "source", "distortion", "distortion_class", "label", "param"]


# ---------------------------------------------------------------- edit pool
def rot_any(img, rng, mag=1.0):
    ang = rng.choice([-1, 1]) * rng.uniform(5, 25) * mag
    return img.rotate(ang, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255)), f"rot{ang:.0f}"

def perspective(img, rng, mag=1.0):
    w, h = img.size
    d = 0.08 * mag
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(x + rng.uniform(-d, d) * w, y + rng.uniform(-d, d) * h) for x, y in src]
    # solve homography coefficients for PIL
    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); B.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); B.append(v)
    coeffs = np.linalg.solve(np.array(A, float), np.array(B, float))
    return img.transform((w, h), Image.PERSPECTIVE, tuple(coeffs), Image.BICUBIC,
                         fillcolor=(255, 255, 255)), "persp"

def hue_shift(img, rng, mag=1.0):
    sh = int(rng.choice([-1, 1]) * rng.uniform(15, 60) * mag)
    hsv = np.array(img.convert("HSV")).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + sh) % 256
    return Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB"), f"hue{sh}"

def crop_big(img, rng, mag=1.0):
    f = rng.uniform(0.10, 0.20) * mag
    w, h = img.size
    l, t = rng.uniform(0, f) * w, rng.uniform(0, f) * h
    return img.crop((round(l), round(t), round(w - (f * w - l)), round(h - (f * h - t)))), f"crop{f:.2f}"

def blur(img, rng, mag=1.0):
    r = rng.uniform(1.0, 2.5) * mag
    return img.filter(ImageFilter.GaussianBlur(r)), f"blur{r:.1f}"

def sticker(img, rng, mag=1.0):
    im = img.copy(); d = ImageDraw.Draw(im)
    w, h = im.size; s = int(min(w, h) * rng.uniform(0.12, 0.25) * mag)
    x, y = rng.randrange(0, max(1, w - s)), rng.randrange(0, max(1, h - s))
    col = tuple(rng.randrange(256) for _ in range(3))
    (d.ellipse if rng.random() < 0.5 else d.rectangle)((x, y, x + s, y + s), fill=col)
    return im, "sticker"

def flip(img, rng, mag=1.0):
    return ImageOps.mirror(img), "hflip"

def jpeg(img, rng, mag=1.0):
    import io
    q = rng.randint(40, 75); b = io.BytesIO(); img.save(b, "JPEG", quality=q)
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB"), f"jpeg{q}"

def resize(img, rng, mag=1.0):
    s = rng.uniform(0.4, 0.8)
    return img.resize((max(8, round(img.width * s)), max(8, round(img.height * s))), Image.LANCZOS), f"rs{s:.2f}"

def border(img, rng, mag=1.0):
    b = round(min(img.size) * rng.uniform(0.04, 0.12))
    return ImageOps.expand(img, border=b, fill=tuple(rng.randrange(256) for _ in range(3))), "border"

POOL = [rot_any, perspective, hue_shift, crop_big, blur, sticker, flip, jpeg, resize, border]


def compound(refs, rng, per_ref, rows):
    for ref in refs:
        img = Image.open(ref).convert("RGB")
        for k in range(per_ref):
            n = rng.choice([2, 3])
            ops = rng.sample(POOL, n)
            out, tags = img, []
            for op in ops:
                out, t = op(out, rng)
                tags.append(t)
            name = f"{ref.stem}__cmp{k}.png"
            out.save(IMG / name)
            rows.append(dict(candidate=name, source=ref.name, distortion=f"compound{n}",
                             distortion_class="compound", label="duplicate", param="+".join(tags)))


# ------------------------------------------------------- adaptive evasion
def adaptive(refs, rng, rows):
    """Smallest edit budget (mag in 0.1..1.0 over rotation+crop+hue+blur) that
    lifts M7 distance above its threshold.  Keeps the FIRST evading image."""
    from models import M7TransformAwarePHash
    m7 = M7TransformAwarePHash()
    thr = m7.threshold
    budget_ops = [rot_any, crop_big, hue_shift, blur]
    for ref in refs:
        img = Image.open(ref).convert("RGB")
        ref_fp = m7.fingerprint(ref)
        found = None
        for mag in np.round(np.arange(0.1, 1.01, 0.1), 1):
            for trial in range(3):
                r2 = random.Random(f"{ref.name}-{mag}-{trial}")  # str seed: reproducible across runs
                out = img
                for op in budget_ops:
                    out, _ = op(out, r2, mag=float(mag))
                tmp = IMG / f"_probe.png"
                out.save(tmp)
                d = m7.distance(ref_fp, m7.fingerprint(tmp))
                if d > thr:
                    found = (mag, out, d)
                    break
            if found:
                break
        name = f"{ref.stem}__adapt.png"
        if found:
            found[1].save(IMG / name)
            rows.append(dict(candidate=name, source=ref.name, distortion="adaptive_m7",
                             distortion_class="adaptive", label="duplicate",
                             param=f"mag={found[0]};m7={found[2]:.3f}"))
        else:
            rows.append(dict(candidate="", source=ref.name, distortion="adaptive_m7",
                             distortion_class="adaptive", label="not_evaded", param="mag>1.0"))
    (IMG / "_probe.png").unlink(missing_ok=True)


# ------------------------------------------------------------- generative
IP2P_EDITS = [
    "make it a watercolor painting", "turn it into a pencil sketch",
    "change the background to a sunset", "add sunglasses", "make it look like a 3D render",
    "make it winter with snow", "turn it into pixel art", "make it look like an oil painting",
]


def generative(refs, rng, rows, strengths=(0.3, 0.5, 0.7), n_ip2p=3):
    import torch
    from diffusers import (StableDiffusionImg2ImgPipeline,
                           StableDiffusionInstructPix2PixPipeline)
    dev = "cuda"
    gen = torch.Generator(dev).manual_seed(42)

    def square512(img):
        return ImageOps.pad(img, (512, 512), color=(255, 255, 255))

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False).to(dev)
    pipe.set_progress_bar_config(disable=True)
    for i, ref in enumerate(refs):
        img = square512(Image.open(ref).convert("RGB"))
        for s in strengths:
            out = pipe(prompt="a digital artwork, high quality", image=img, strength=s,
                       guidance_scale=7.0, num_inference_steps=30, generator=gen).images[0]
            name = f"{ref.stem}__i2i{int(s*100)}.png"
            out.save(IMG / name)
            rows.append(dict(candidate=name, source=ref.name, distortion=f"img2img_{s}",
                             distortion_class="generative", label="duplicate", param=f"strength={s}"))
        if i % 10 == 0:
            print(f"  img2img {i}/{len(refs)}", flush=True)
    del pipe; torch.cuda.empty_cache()

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False).to(dev)
    pipe.set_progress_bar_config(disable=True)
    for i, ref in enumerate(refs):
        img = square512(Image.open(ref).convert("RGB"))
        for j, edit in enumerate(rng.sample(IP2P_EDITS, n_ip2p)):
            out = pipe(edit, image=img, num_inference_steps=30, image_guidance_scale=1.5,
                       guidance_scale=7.5, generator=gen).images[0]
            name = f"{ref.stem}__ip2p{j}.png"
            out.save(IMG / name)
            rows.append(dict(candidate=name, source=ref.name, distortion="instruct_pix2pix",
                             distortion_class="generative", label="duplicate", param=edit))
        if i % 10 == 0:
            print(f"  ip2p {i}/{len(refs)}", flush=True)
    del pipe; torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="compound,adaptive")
    ap.add_argument("--compound-per-ref", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="only first N refs (smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    IMG.mkdir(parents=True, exist_ok=True)
    refs = sorted(REG.glob("*.png"))
    if args.limit:
        refs = refs[:args.limit]
    parts = [p.strip() for p in args.parts.split(",")]

    # one CSV per part so parts can run concurrently (CPU and GPU shells);
    # the merged ground_truth_hard.csv is rebuilt from all part files.
    for p in parts:
        rng = random.Random(f"{args.seed}-{p}")
        rows = []
        {"compound": lambda: compound(refs, rng, args.compound_per_ref, rows),
         "adaptive": lambda: adaptive(refs, rng, rows),
         "generative": lambda: generative(refs, rng, rows)}[p]()
        with open(OUT / f"gt_{p}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
        print(f"{p}: {len(rows)} rows -> gt_{p}.csv")

    merged = []
    for f in sorted(OUT.glob("gt_*.csv")):
        merged += list(csv.DictReader(open(f)))
    with open(GT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(merged)
    print(f"-> {GT} ({len(merged)} rows, parts: {[f.stem for f in sorted(OUT.glob('gt_*.csv'))]})")


if __name__ == "__main__":
    main()
