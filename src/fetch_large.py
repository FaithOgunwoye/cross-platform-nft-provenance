"""Build the LARGE real-world registry (camera-ready experiment, reviewers R1/R4).

The original benchmark indexes 100 references and scales with synthetic
distractors.  This script replaces the synthetic distractors with thousands of
REAL NFT images fetched on-chain from many collections, so that registry-scale
behaviour (false positives from stylistic siblings, retrieval accuracy) is
measured on genuine marketplace media.

Output (data/large/):
  images/<slug>_<id>.png     media, longest side capped at 512 px (git-ignored)
  registry_large.json        one provenance record per image (committed)
  fetch_log.csv              per-token success/failure (committed)

Tokens already used in the original benchmark (D1 references and the 40
non-duplicates) are excluded so the large set is disjoint from it.  Images
whose canonical hash repeats (e.g. unrevealed placeholder art) are dropped.

Run:  python src/fetch_large.py --per-collection 200 --workers 32 --max-minutes 45
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
import provenance as pv
from provenance import extract_media_uri, get_token_uri, resolve_metadata, rpc_call

BASE = Path(__file__).parent.parent
OUT = BASE / "data" / "large"
MAX_SIDE = 512

# More gateways than the original fetcher: at this volume single gateways 429.
pv.IPFS_GATEWAYS[:] = [
    "https://ipfs.io/ipfs/",
    "https://dweb.link/ipfs/",
    "https://gateway.pinata.cloud/ipfs/",
    "https://nftstorage.link/ipfs/",
    "https://ipfs.filebase.io/ipfs/",
    "https://w3s.link/ipfs/",
    "https://4everland.io/ipfs/",
    "https://cloudflare-ipfs.com/ipfs/",
]

# (name, category, contract, token-id generator hint)
# id_base/id_max: ids are sampled from [id_base, id_base+supply).  Contracts
# that fail (wrong address, non-standard tokenURI) are simply logged & skipped.
COLLECTIONS = [
    # PFP / avatar
    ("Bored Ape Yacht Club", "pfp_avatar", "0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D", 0, 10000),
    ("Mutant Ape Yacht Club", "pfp_avatar", "0x60E4d786628Fea6478F785A6d7e704777c86a7c6", 0, 20000),
    ("Doodles", "pfp_avatar", "0x8a90CAb2b38dba80c64b7734e58Ee1dB38B8992e", 0, 10000),
    ("CloneX", "pfp_avatar", "0x49cF6f5d44E70224e2E23fDcdd2C053F30aDA28B", 1, 19000),
    ("Cool Cats", "pfp_avatar", "0x1A92f7381B9F03921564a437210bB9396471050C", 0, 9900),
    ("mfers", "pfp_avatar", "0x79FCDEF22feeD20eDDacbB2587640e45491b757f", 0, 10000),
    ("Milady Maker", "pfp_avatar", "0x5Af0D9827E0c53E4799BB226655A1de152A425a5", 0, 10000),
    ("Lazy Lions", "pfp_avatar", "0x8943C7bAC1914C9A7ABa750Bf2B6B09Fd21037E0", 0, 10000),
    ("Bored Ape Kennel Club", "pfp_avatar", "0xba30E5F9Bb24caa003E9f2f0497Ad287FDF95623", 0, 9600),
    ("Deadfellaz", "pfp_avatar", "0x2acAb3DEa77832C09420663b0E1cB386031bA17B", 0, 10000),
    # Pixel art
    ("Moonbirds", "pixel_art", "0x23581767a106ae21c074b2276D25e5C3e136a68b", 0, 10000),
    ("Nouns", "pixel_art", "0x9C8fF314C9Bc7F6e59A9d9225Fb22946427eDC03", 0, 1000),
    ("CrypToadz", "pixel_art", "0x1CB1A5e65610AEFF2551A50f76a87a7d3fB649C6", 1, 6969),
    ("Blitmap", "pixel_art", "0x8d04a8c79cEB0889Bdd12acdF3Fa9D207eD3Ff63", 0, 1700),
    # Generative abstract (Art Blocks curated contract: id = project*1e6 + n)
    ("Art Blocks: Chromie Squiggle", "generative_abstract", "0x059EDD72Cd353dF5106D2B9cC5ab83a52287aC3a", 0, 9000),
    ("Art Blocks: Fidenza", "generative_abstract", "0xa7d8d9ef8D8Ce8992Df33D8b8CF4Aebabd5bD270", 78000000, 999),
    ("Art Blocks: Ringers", "generative_abstract", "0xa7d8d9ef8D8Ce8992Df33D8b8CF4Aebabd5bD270", 13000000, 1000),
    ("Art Blocks: Archetype", "generative_abstract", "0xa7d8d9ef8D8Ce8992Df33D8b8CF4Aebabd5bD270", 23000000, 600),
    ("Art Blocks: Subscapes", "generative_abstract", "0xa7d8d9ef8D8Ce8992Df33D8b8CF4Aebabd5bD270", 53000000, 650),
    # Digital illustration
    ("Azuki", "digital_illustration", "0xED5AF388653567Af2F388E6224dC7C4b3241C544", 0, 10000),
    ("World of Women", "digital_illustration", "0xe785E82358879F061BC3dcAC6f0444462D4b5330", 0, 10000),
    ("0N1 Force", "digital_illustration", "0x3bf2922f4520a8BA0c2eFC3D2a1539678DaD5e9D", 1, 7777),
    ("Goblintown", "digital_illustration", "0xbCe3781ae7Ca1a5e050Bd9C4c77369867eBc307e", 1, 9999),
    ("Kaiju Kingz", "digital_illustration", "0x0c2E57EFddbA8c768147D1fdF9176a0A6EBd5d83", 0, 3333),
    ("Sappy Seals", "digital_illustration", "0x364C828eE171616a39897688A831c2499aD972ec", 0, 10000),
    ("VeeFriends", "digital_illustration", "0xa3AEe8BcE55BEeA1951EF834b99f3Ac60d1ABeeB", 1, 10255),
    # Text / logo
    ("Loot", "text_logo", "0xFF9C1b15B16263C61d017ee9F65C50e4AE0113D7", 1, 8000),
    # 3D / rendered
    ("Pudgy Penguins", "render_3d", "0xBd3531dA5CF5857e7CfAA92426877b022e612cf8", 0, 8888),
    ("Lil Pudgys", "render_3d", "0x524cAB2ec69124574082676e6F654a18df49A048", 0, 22222),
    ("Meebits", "render_3d", "0x7Bd29408f11D2bFC23c34f18275bBf23bB716Bc7", 1, 20000),
    ("Otherdeed", "render_3d", "0x34d85c9CDeB23FA97cb08333b511ac86E1C4E258", 1, 100000),
    ("Invisible Friends", "render_3d", "0x59468516a8259058baD1cA5F8f4BFF190d30E066", 1, 5000),
    ("Creature World", "digital_illustration", "0xc92cedDfb8dd984A89fb494c376f9A48b999aAFc", 1, 10000),
]


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace(":", "").replace("/", "_")


def excluded_tokens() -> set:
    """(contract_lower, token_id) pairs already used by the original benchmark."""
    ex = set()
    for f in ("registry.json", "nonduplicates.json"):
        p = BASE / "data" / f
        if p.exists():
            for r in json.loads(p.read_text()):
                ex.add((r["contract"].lower(), int(r["token_id"])))
    return ex


def ipfs_normalize(uri: str) -> str:
    """https://<any gateway>/ipfs/<cid> -> ipfs://<cid>, so the gateway fallback
    list is used instead of one hard-coded (often rate-limited) gateway."""
    m = re.match(r"^https?://[^/]+/ipfs/(.+)$", uri or "")
    return "ipfs://" + m.group(1) if m else uri


def media_to_image(content: bytes, media_uri: str) -> Image.Image:
    if media_uri.startswith("data:image/svg") or content.lstrip()[:4] in (b"<svg", b"<?xm"):
        import cairosvg
        content = cairosvg.svg2png(bytestring=content, output_width=512, output_height=512)
    img = Image.open(io.BytesIO(content))
    img.seek(0)  # animated GIF/WebP: first frame
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        s = MAX_SIDE / max(img.size)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
    return img


def canonical(img: Image.Image) -> str:
    g = ImageOps.grayscale(img).resize((64, 64), Image.LANCZOS)
    return hashlib.sha256(g.tobytes()).hexdigest()


def fetch_one(spec, tid):
    name, cat, contract, _, _ = spec
    uri = get_token_uri("ethereum", contract, tid)
    if not uri:
        raise RuntimeError("empty tokenURI")
    meta = resolve_metadata(ipfs_normalize(uri))
    media = extract_media_uri(meta)
    if not media:
        raise RuntimeError("no media uri")
    if media.startswith(("http", "ipfs", "ar:", "data:")) is False:
        raise RuntimeError(f"unsupported media uri {media[:40]}")
    content = pv.resolve_uri(ipfs_normalize(media), timeout=20)
    img = media_to_image(content, media)
    rec = dict(chain="ethereum", contract=contract, token_id=tid, collection=name,
               category=cat,
               token_uri=uri[:200] if uri.startswith("data:") else uri,
               media_uri=media[:200] if media.startswith("data:") else media,
               media_sha256=hashlib.sha256(content).hexdigest(),
               name=str(meta.get("name", ""))[:120])
    return rec, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-collection", type=int, default=200)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-minutes", type=float, default=45)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", default=None, help="comma-separated collection-name substrings")
    args = ap.parse_args()

    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    reg_path = OUT / "registry_large.json"
    records = json.loads(reg_path.read_text()) if reg_path.exists() else []
    have = {(r["contract"].lower(), r["token_id"]) for r in records}
    seen_canon = {r["canonical"] for r in records}
    ex = excluded_tokens()
    rng = random.Random(args.seed)

    cols = COLLECTIONS
    if args.only:
        keys = [k.strip().lower() for k in args.only.split(",")]
        cols = [c for c in cols if any(k in c[0].lower() for k in keys)]

    # Oversample ids (x1.6) because some tokens fail / are placeholders.
    jobs = []
    for spec in cols:
        _, _, contract, base, supply = spec
        k = min(supply, int(args.per_collection * 1.6))
        ids = sorted(rng.sample(range(supply), k))
        for off in ids:
            tid = base + off
            key = (contract.lower(), tid)
            if key in ex or key in have:
                continue
            jobs.append((spec, tid))
    rng.shuffle(jobs)  # interleave collections -> spreads gateway load
    print(f"{len(jobs)} token fetches queued over {len(cols)} collections")

    lock = threading.Lock()
    per_col = {c[0]: sum(1 for r in records if r["collection"] == c[0]) for c in cols}
    log_rows = []
    deadline = time.time() + args.max_minutes * 60
    done = 0

    def task(spec, tid):
        if time.time() > deadline:
            return spec, tid, None, None, "deadline"
        with lock:
            if per_col[spec[0]] >= args.per_collection:
                return spec, tid, None, None, "quota"
        try:
            rec, img = fetch_one(spec, tid)
            return spec, tid, rec, img, "ok"
        except Exception as e:
            return spec, tid, None, None, f"fail: {str(e)[:120]}"

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(task, s, t) for s, t in jobs]
        for fut in as_completed(futs):
            spec, tid, rec, img, status = fut.result()
            if status == "ok":
                c = canonical(img)
                with lock:
                    if c in seen_canon:
                        status = "dup_canonical"
                    elif per_col[spec[0]] >= args.per_collection:
                        status = "quota"
                    else:
                        seen_canon.add(c)
                        per_col[spec[0]] += 1
                        fname = f"{slugify(spec[0])}_{tid}.png"
                        img.save(img_dir / fname, format="PNG")
                        rec["canonical"] = c
                        rec["local_file"] = fname
                        records.append(rec)
            if status not in ("quota", "deadline"):
                log_rows.append(dict(collection=spec[0], token_id=tid, status=status))
            done += 1
            if done % 200 == 0:
                ok = sum(per_col.values())
                print(f"  {done}/{len(jobs)} processed, {ok} images kept "
                      f"({(deadline - time.time())/60:.1f} min left)", flush=True)
                reg_path.write_text(json.dumps(records, indent=1))

    reg_path.write_text(json.dumps(records, indent=1))
    log_path = OUT / "fetch_log.csv"  # append across resumed runs
    prev = list(csv.DictReader(open(log_path))) if log_path.exists() else []
    with open(log_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["collection", "token_id", "status"])
        w.writeheader(); w.writerows(prev + log_rows)
    print("\nper-collection counts:")
    for k, v in sorted(per_col.items(), key=lambda kv: -kv[1]):
        print(f"  {v:5d}  {k}")
    print(f"total: {len(records)} images -> {reg_path}")


if __name__ == "__main__":
    main()
