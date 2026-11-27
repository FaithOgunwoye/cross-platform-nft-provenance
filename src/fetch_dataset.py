"""Build the D1 clean reference registry from live blockchain data.

For every sampled token the pipeline:
  1. reads tokenURI(tokenId) from the contract via eth_call,
  2. resolves the metadata (IPFS / HTTP / on-chain data URI),
  3. extracts and downloads the media image,
  4. attaches mint block / timestamp / minter from Blockscout,
  5. stores the image plus a full provenance record.

Run:  python src/fetch_dataset.py [--out data]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from provenance import (TokenProvenance, extract_media_uri, get_mint_events,
                        get_token_uri, resolve_metadata, resolve_uri)

# Collections sampled for the reference registry, one entry per paper category.
# Fallbacks within a category are listed after the primary collection.
COLLECTIONS = [
    # PFP / avatar
    dict(name="Bored Ape Yacht Club", category="pfp_avatar", chain="ethereum",
         contract="0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D", token_ids=list(range(1, 11))),
    dict(name="Doodles", category="pfp_avatar", chain="ethereum",
         contract="0x8a90CAb2b38dba80c64b7734e58Ee1dB38B8992e", token_ids=list(range(1, 11))),
    # Pixel art
    dict(name="Moonbirds", category="pixel_art", chain="ethereum",
         contract="0x23581767a106ae21c074b2276D25e5C3e136a68b", token_ids=list(range(1, 11))),
    dict(name="Nouns", category="pixel_art", chain="ethereum",
         contract="0x9C8fF314C9Bc7F6e59A9d9225Fb22946427eDC03", token_ids=list(range(1, 11))),
    # Generative abstract art
    dict(name="Art Blocks: Fidenza", category="generative_abstract", chain="ethereum",
         contract="0xa7d8d9ef8D8Ce8992Df33D8b8CF4Aebabd5bD270",
         token_ids=list(range(78000001, 78000011))),
    dict(name="Art Blocks: Chromie Squiggle", category="generative_abstract", chain="ethereum",
         contract="0x059EDD72Cd353dF5106D2B9cC5ab83a52287aC3a", token_ids=list(range(1, 11))),
    # Digital illustration
    dict(name="Azuki", category="digital_illustration", chain="ethereum",
         contract="0xED5AF388653567Af2F388E6224dC7C4b3241C544", token_ids=list(range(1, 11))),
    dict(name="World of Women", category="digital_illustration", chain="ethereum",
         contract="0xe785E82358879F061BC3dcAC6f0444462D4b5330", token_ids=list(range(1, 11))),
    # Text / logo based (fully on-chain SVG text)
    dict(name="Loot", category="text_logo", chain="ethereum",
         contract="0xFF9C1b15B16263C61d017ee9F65C50e4AE0113D7", token_ids=list(range(1, 11))),
    # 3D / rendered objects
    dict(name="Pudgy Penguins", category="render_3d", chain="ethereum",
         contract="0xBd3531dA5CF5857e7CfAA92426877b022e612cf8", token_ids=list(range(1, 6))),
    dict(name="Meebits", category="render_3d", chain="ethereum",
         contract="0x7Bd29408f11D2bFC23c34f18275bBf23bB716Bc7", token_ids=list(range(1, 6))),
]

# Extra tokens, never placed in the registry: used as genuine non-duplicates.
# Same collections as the registry, so they are the hardest negative type
# (stylistically similar collection-mates).
_NONDUP_IDS = [500, 501, 502, 600, 601, 602, 603, 604]
NON_DUPLICATES = [
    dict(name="Bored Ape Yacht Club", chain="ethereum",
         contract="0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D", token_ids=_NONDUP_IDS),
    dict(name="Doodles", chain="ethereum",
         contract="0x8a90CAb2b38dba80c64b7734e58Ee1dB38B8992e", token_ids=_NONDUP_IDS),
    dict(name="Azuki", chain="ethereum",
         contract="0xED5AF388653567Af2F388E6224dC7C4b3241C544", token_ids=_NONDUP_IDS),
    dict(name="Moonbirds", chain="ethereum",
         contract="0x23581767a106ae21c074b2276D25e5C3e136a68b", token_ids=_NONDUP_IDS),
    dict(name="Loot", chain="ethereum",
         contract="0xFF9C1b15B16263C61d017ee9F65C50e4AE0113D7", token_ids=_NONDUP_IDS),
]

MAX_SIDE = 1024  # cap stored resolution so the benchmark stays lightweight


def media_to_png(content: bytes, media_uri: str) -> Image.Image:
    """Decode downloaded media (raster or SVG) into an RGB PIL image."""
    if media_uri.startswith("data:image/svg") or content.lstrip()[:4] in (b"<svg", b"<?xm"):
        import cairosvg
        content = cairosvg.svg2png(bytestring=content, output_width=512, output_height=512)
    img = Image.open(io.BytesIO(content)).convert("RGB")
    if max(img.size) > MAX_SIDE:
        scale = MAX_SIDE / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return img


def fetch_token(spec: dict, token_id: int, mints: dict) -> tuple[TokenProvenance, Image.Image]:
    token_uri = get_token_uri(spec["chain"], spec["contract"], token_id)
    if not token_uri:
        raise RuntimeError("empty tokenURI")
    metadata = resolve_metadata(token_uri)
    media_uri = extract_media_uri(metadata)
    if not media_uri:
        raise RuntimeError(f"no media URI in metadata keys={list(metadata)[:8]}")
    content = resolve_uri(media_uri)
    img = media_to_png(content, media_uri)
    mint = mints.get(token_id, {})
    prov = TokenProvenance(
        chain=spec["chain"], contract=spec["contract"], token_id=token_id,
        collection=spec["name"], category=spec.get("category", ""),
        token_uri=token_uri[:200] if token_uri.startswith("data:") else token_uri,
        media_uri=media_uri[:200] if media_uri.startswith("data:") else media_uri,
        mint_block=mint.get("block"), mint_timestamp=mint.get("timestamp"),
        mint_tx=mint.get("tx", ""), minter=mint.get("minter", ""),
        media_sha256=hashlib.sha256(content).hexdigest(),
        extra={"name": metadata.get("name", "")},
    )
    return prov, img


def fetch_set(specs: list, out_dir: Path, prefix: str, existing: list | None = None) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = list(existing or [])
    done = {r["local_file"] for r in records}
    for spec in specs:
        print(f"\n== {spec['name']} ({spec['contract'][:10]}…) ==")
        try:
            mints = get_mint_events(spec["chain"], spec["contract"])
            print(f"   mint events resolved for {len(mints)} early tokens")
        except Exception as e:
            print(f"   mint lookup failed ({e}); provenance will lack timestamps")
            mints = {}
        for tid in spec["token_ids"]:
            slug = spec["name"].lower().replace(" ", "_").replace(":", "")
            fname = f"{prefix}_{slug}_{tid}.png"
            target = out_dir / fname
            if fname in done and target.exists():
                print(f"   token {tid}: cached")
                continue
            try:
                prov, img = fetch_token(spec, tid, mints)
                img.save(target, format="PNG")
                prov.local_file = fname
                records.append(prov.to_dict())
                ts = prov.mint_timestamp
                print(f"   token {tid}: OK  mint_ts={ts}  {prov.media_uri[:60]}")
                time.sleep(0.3)
            except Exception as e:
                print(f"   token {tid}: FAILED ({e})")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent.parent / "data"))
    args = ap.parse_args()
    data = Path(args.out)

    def load(p):
        return json.loads(p.read_text()) if p.exists() else []

    print("### Building D1 reference registry (live on-chain fetch) ###")
    registry = fetch_set(COLLECTIONS, data / "registry", "ref",
                         existing=load(data / "registry.json"))
    (data / "registry.json").write_text(json.dumps(registry, indent=2))
    print(f"\nD1 registry: {len(registry)} images with provenance -> {data/'registry.json'}")

    print("\n### Fetching genuine non-duplicate images ###")
    nondup = fetch_set(NON_DUPLICATES, data / "nonduplicates", "nondup",
                       existing=load(data / "nonduplicates.json"))
    (data / "nonduplicates.json").write_text(json.dumps(nondup, indent=2))
    print(f"Non-duplicates: {len(nondup)} images")


if __name__ == "__main__":
    main()
