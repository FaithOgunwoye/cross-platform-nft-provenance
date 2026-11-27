"""Generate the D2 distorted candidate dataset from the D1 registry.

Implements every distortion class from the paper skeleton (Section VI):
  A. file-level        : metadata strip, JPEG recompression, WebP conversion
  B. pixel-level       : one-pixel change, few-pixel changes, low-level noise
  C. visual-preserving : resize down/up, brightness, contrast, grayscale,
                         screenshot simulation
  D. evasion           : horizontal/vertical flip, color inversion, rotation
  E. hard near-dup     : 5% crop, border, watermark text overlay

Each candidate is written to data/candidates/ and logged in
data/ground_truth.csv with its true source image and distortion label.

Run:  python src/distort.py [--data data] [--per-image 8] [--seed 42]
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

JPEG = dict(format="JPEG", quality=70)


def save(img: Image.Image, path: Path, **kw):
    img.save(path, **(kw or {"format": "PNG"}))


# --- A. file-level -----------------------------------------------------------

def d_metadata_edit(img, path):
    """Same pixels, different file bytes: inject PNG text metadata chunks."""
    from PIL.PngImagePlugin import PngInfo
    info = PngInfo()
    info.add_text("Title", "totally original artwork")
    info.add_text("Author", "definitely-not-a-copyminter")
    img.save(path.with_suffix(".png"), format="PNG", pnginfo=info)

def d_jpeg_q70(img, path):
    save(img, path.with_suffix(".jpg"), **JPEG)

def d_webp(img, path):
    save(img, path.with_suffix(".webp"), format="WEBP", quality=90)

# --- B. pixel-level ----------------------------------------------------------

def d_one_pixel(img, path):
    arr = np.array(img)
    v = int(arr[0, 0, 0])
    arr[0, 0, 0] = v - 1 if v == 255 else v + 1  # guaranteed to change
    save(Image.fromarray(arr), path.with_suffix(".png"))

def d_few_pixels(img, path, rng):
    arr = np.array(img)
    h, w = arr.shape[:2]
    for _ in range(50):
        y, x = rng.randrange(h), rng.randrange(w)
        arr[y, x] = np.clip(arr[y, x].astype(int) + rng.choice([-2, -1, 1, 2]), 0, 255)
    save(Image.fromarray(arr), path.with_suffix(".png"))

def d_noise(img, path, rng):
    arr = np.array(img).astype(np.int16)
    noise = np.random.default_rng(rng.randrange(2**31)).integers(-4, 5, arr.shape)
    save(Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8)), path.with_suffix(".png"))

# --- C. visual-preserving ----------------------------------------------------

def d_resize_50(img, path):
    save(img.resize((max(1, img.width // 2), max(1, img.height // 2)), Image.LANCZOS),
         path.with_suffix(".png"))

def d_resize_150(img, path):
    save(img.resize((round(img.width * 1.5), round(img.height * 1.5)), Image.LANCZOS),
         path.with_suffix(".png"))

def d_brightness(img, path):
    save(ImageEnhance.Brightness(img).enhance(1.2), path.with_suffix(".png"))

def d_contrast(img, path):
    save(ImageEnhance.Contrast(img).enhance(1.25), path.with_suffix(".png"))

def d_grayscale(img, path):
    save(ImageOps.grayscale(img).convert("RGB"), path.with_suffix(".png"))

def d_screenshot(img, path, rng):
    """Resize + recompress + place on a background, like a marketplace screenshot."""
    scale = rng.uniform(0.6, 0.9)
    inner = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    bg = Image.new("RGB", (round(inner.width * 1.15), round(inner.height * 1.15)), (245, 245, 247))
    bg.paste(inner, ((bg.width - inner.width) // 2, (bg.height - inner.height) // 2))
    save(bg, path.with_suffix(".jpg"), format="JPEG", quality=80)

# --- D. evasion --------------------------------------------------------------

def d_hflip(img, path):
    save(ImageOps.mirror(img), path.with_suffix(".png"))

def d_vflip(img, path):
    save(ImageOps.flip(img), path.with_suffix(".png"))

def d_invert(img, path):
    save(ImageOps.invert(img), path.with_suffix(".png"))

def d_rotate_90(img, path):
    save(img.transpose(Image.ROTATE_90), path.with_suffix(".png"))

def d_rotate_180(img, path):
    save(img.transpose(Image.ROTATE_180), path.with_suffix(".png"))

# --- E. hard near-duplicates -------------------------------------------------

def d_crop_5(img, path):
    m = round(min(img.size) * 0.05)
    save(img.crop((m, m, img.width - m, img.height - m)), path.with_suffix(".png"))

def d_border(img, path):
    save(ImageOps.expand(img, border=round(min(img.size) * 0.06), fill=(20, 20, 20)),
         path.with_suffix(".png"))

def d_watermark(img, path):
    marked = img.copy()
    draw = ImageDraw.Draw(marked)
    text = "© mint"
    x, y = img.width - 90, img.height - 28
    draw.text((x + 1, y + 1), text, fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255))
    save(marked, path.with_suffix(".png"))


DISTORTIONS = {
    # name: (function, class, needs_rng)
    "metadata_edit":  (d_metadata_edit,  "file_level", False),
    "jpeg_q70":       (d_jpeg_q70,       "file_level", False),
    "webp":           (d_webp,           "file_level", False),
    "one_pixel":      (d_one_pixel,      "pixel_level", False),
    "few_pixels":     (d_few_pixels,     "pixel_level", True),
    "noise":          (d_noise,          "pixel_level", True),
    "resize_50":      (d_resize_50,      "visual_preserving", False),
    "resize_150":     (d_resize_150,     "visual_preserving", False),
    "brightness":     (d_brightness,     "visual_preserving", False),
    "contrast":       (d_contrast,       "visual_preserving", False),
    "grayscale":      (d_grayscale,      "visual_preserving", False),
    "screenshot":     (d_screenshot,     "visual_preserving", True),
    "hflip":          (d_hflip,          "evasion", False),
    "vflip":          (d_vflip,          "evasion", False),
    "invert":         (d_invert,         "evasion", False),
    "rotate_90":      (d_rotate_90,      "evasion", False),
    "rotate_180":     (d_rotate_180,     "evasion", False),
    "crop_5":         (d_crop_5,         "hard_near_dup", False),
    "border":         (d_border,         "hard_near_dup", False),
    "watermark":      (d_watermark,      "hard_near_dup", False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).parent.parent / "data"))
    ap.add_argument("--per-image", type=int, default=8,
                    help="distortions sampled per reference image (0 = all 20)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = Path(args.data)
    reg_dir, cand_dir = data / "registry", data / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    refs = sorted(reg_dir.glob("*.png"))
    if not refs:
        raise SystemExit(f"no reference images in {reg_dir}; run fetch_dataset.py first")

    names = list(DISTORTIONS)
    rows = []
    for ref in refs:
        img = Image.open(ref).convert("RGB")
        chosen = names if args.per_image == 0 else rng.sample(names, k=min(args.per_image, len(names)))
        for name in chosen:
            fn, dclass, needs_rng = DISTORTIONS[name]
            out = cand_dir / f"{ref.stem}__{name}"
            fn(img, out, rng) if needs_rng else fn(img, out)
            actual = next(cand_dir.glob(f"{ref.stem}__{name}.*"))
            rows.append({"candidate": actual.name, "source": ref.name,
                         "distortion": name, "distortion_class": dclass, "label": "duplicate"})

    # genuine non-duplicates: registered with no source
    for f in sorted((data / "nonduplicates").glob("*")):
        rows.append({"candidate": f.name, "source": "", "distortion": "none",
                     "distortion_class": "non_duplicate", "label": "non_duplicate"})

    gt = data / "ground_truth.csv"
    with open(gt, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["candidate", "source", "distortion",
                                           "distortion_class", "label"])
        w.writeheader()
        w.writerows(rows)
    dup = sum(1 for r in rows if r["label"] == "duplicate")
    print(f"D2: {dup} distorted candidates + {len(rows)-dup} non-duplicates -> {gt}")


if __name__ == "__main__":
    main()
