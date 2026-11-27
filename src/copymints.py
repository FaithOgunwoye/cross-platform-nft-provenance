"""Real-world copymint pairs (camera-ready experiment, reviewers R2/R3).

Two documented cases where an existing collection's art was re-minted under a
different contract:

  RR/BAYC      (0x2EE6...691e)  re-mint of Bored Ape Yacht Club images (the
               subject of Yuga Labs v. Ripps).  Media is re-encoded JPEG on a
               different IPFS CID.
  CryptoPhunks (0xf074...F402)  horizontally mirrored CryptoPunks.  Punk art is
               read fully on-chain from the CryptopunksData contract.

For each sampled id we store the ORIGINAL image, the COPY image, and mint
provenance for both, so the screener can be tested on genuine copymints and
the who-minted-first resolver on real mint orderings.

The id correspondence (copy #n <-> original #n) is taken from the copy's
metadata name when it carries a number, else assumed equal; `verify` in the
output records the M7 pHash distance of each pair so mismatched pairs are
visible (and should be spot-checked by eye before being reported).

Output (data/copymints/):
  original/<case>_<id>.png, copy/<case>_<id>.png, pairs.json
Run:  python src/copymints.py --n 150
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import provenance as pv
from provenance import (TRANSFER_TOPIC, extract_media_uri, get_token_uri,
                        resolve_metadata, rpc_call)
from fetch_large import ipfs_normalize, media_to_image

BASE = Path(__file__).parent.parent
OUT = BASE / "data" / "copymints"

BAYC = "0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D"
RRBAYC = "0x2EE6AF0dFf3A1CE3F7E3414C52c48fd50d73691e"
PHUNKS = "0xf07468eAd8cf26c752C676E43C814FEe9c8CF402"
PUNKS = "0xb47e3cd837dDF8e4c57F05d70Ab865de6e193BBB"       # original 2017 contract
PUNKS_DATA = "0x16F5A35647D6F03D5D3da7b35409D65ba03aF3B2"  # on-chain art (2021)
ZERO32 = "0x" + "0" * 64


def keccak_selector(sig: str) -> str:
    try:
        from Crypto.Hash import keccak  # pycryptodome
        k = keccak.new(digest_bits=256); k.update(sig.encode())
        return "0x" + k.hexdigest()[:8]
    except ImportError:
        from eth_hash.auto import keccak as kk
        return "0x" + kk(sig.encode()).hex()[:8]


def punk_image(idx: int) -> Image.Image:
    sel = keccak_selector("punkImageSvg(uint16)")
    res = rpc_call("ethereum", "eth_call",
                   [{"to": PUNKS_DATA, "data": sel + format(idx, "064x")}, "latest"])
    svg = pv._decode_string_return(res)
    if svg.startswith("data:"):
        svg = svg.partition(",")[2]
    import cairosvg
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=480, output_height=480)
    # The SVG has a transparent background; .convert("RGB") would turn it black
    # (merging with black hair).  Composite onto the standard CryptoPunks display
    # background #638596, which is also what the Phunk copies carry.
    rgba = Image.open(io.BytesIO(png)).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (0x63, 0x85, 0x96, 255))
    return Image.alpha_composite(bg, rgba).convert("RGB")


def token_image(contract: str, tid: int):
    uri = get_token_uri("ethereum", contract, tid)
    meta = resolve_metadata(ipfs_normalize(uri))
    media = extract_media_uri(meta)
    content = pv.resolve_uri(ipfs_normalize(media), timeout=25)
    return media_to_image(content, media), dict(token_uri=uri, media_uri=media,
                                                  name=str(meta.get("name", "")),
                                                  media_sha256=hashlib.sha256(content).hexdigest())


# The default public RPCs refuse wide eth_getLogs (publicnode: archive needs a
# token; 1rpc: 50 blocks; drpc: 10k blocks).  These two served the full-range
# queries during the camera-ready run.
LOG_RPCS = ["https://mainnet.gateway.tenderly.co", "https://rpc.mevblocker.io"]


def logs_rpc(flt: dict, timeout: int = 90):
    import requests
    err = None
    for url in LOG_RPCS:
        try:
            j = requests.post(url, json={"jsonrpc": "2.0", "id": 1, "method": "eth_getLogs",
                                         "params": [flt]}, timeout=timeout).json()
            if isinstance(j.get("result"), list):
                return j["result"]
            err = j.get("error")
        except Exception as e:
            err = e
    raise RuntimeError(f"eth_getLogs failed on LOG_RPCS: {err}")


def mint_event(contract: str, tid: int):
    """First Transfer(0x0 -> x, tid) via eth_getLogs (indexed tokenId); falls
    back to Blockscout v2 transfer history.  Returns dict or None."""
    try:
        logs = logs_rpc({
            "address": contract, "fromBlock": "0x0", "toBlock": "latest",
            "topics": [TRANSFER_TOPIC, ZERO32, None, "0x" + format(tid, "064x")]}, timeout=40)
        if logs:
            lg = min(logs, key=lambda l: int(l["blockNumber"], 16))
            blk = int(lg["blockNumber"], 16)
            return dict(block=blk, timestamp=block_timestamp(blk),
                        tx=lg["transactionHash"], minter="0x" + lg["topics"][2][-40:])
    except Exception:
        pass
    try:
        import requests
        url = f"https://eth.blockscout.com/api/v2/tokens/{contract}/instances/{tid}/transfers"
        items, params = [], {}
        for _ in range(20):
            j = requests.get(url, params=params, timeout=30).json()
            items += j.get("items", [])
            if not j.get("next_page_params"):
                break
            params = j["next_page_params"]
        mints = [i for i in items if (i.get("from") or {}).get("hash", "").lower() == "0x" + "0" * 40]
        if mints:
            m = min(mints, key=lambda i: i["block_number"])
            from datetime import datetime
            ts = int(datetime.fromisoformat(m["timestamp"].replace("Z", "+00:00")).timestamp())
            return dict(block=m["block_number"], timestamp=ts, tx=m["transaction_hash"],
                        minter=m["to"]["hash"])
    except Exception:
        pass
    return None


def keccak_topic(sig: str) -> str:
    try:
        from Crypto.Hash import keccak
        k = keccak.new(digest_bits=256); k.update(sig.encode())
        return "0x" + k.hexdigest()
    except ImportError:
        from eth_hash.auto import keccak as kk
        return "0x" + kk(sig.encode()).hex()


def punk_assign_all():
    """All Assign(address indexed to, uint256 punkIndex) events of the 2017
    Punks contract.  CryptoPunks predate ERC-721, so the claim is this custom
    event, not a Transfer from 0x0.  Returns {punk_index: mint dict}.
    punkIndex is not indexed, so all Assign logs are pulled (one wide getLogs on
    an RPC that allows it; see LOG_RPCS)."""
    topic = keccak_topic("Assign(address,uint256)")
    try:
        logs = logs_rpc({"address": PUNKS, "fromBlock": hex(3914495), "toBlock": hex(4000000),
                         "topics": [topic]})
    except Exception as e:
        print(f"  punk Assign fetch failed: {e}")
        return {}
    out = {}
    for r in logs:
        idx = int(r["data"][2:66], 16)
        blk = int(r["blockNumber"], 16)
        if idx not in out or blk < out[idx]["block"]:
            out[idx] = dict(block=blk, timestamp=None, tx=r["transactionHash"],
                            minter="0x" + r["topics"][1][-40:])
    return out


_block_ts = {}


def block_timestamp(blk: int) -> int:
    if blk not in _block_ts:
        b = rpc_call("ethereum", "eth_getBlockByNumber", [hex(blk), False])
        _block_ts[blk] = int(b["timestamp"], 16)
    return _block_ts[blk]


def number_in(name: str):
    m = re.search(r"#\s*(\d+)", name or "")
    return int(m.group(1)) if m else None


def build_case(case, ids, copy_contract, orig_fn, orig_contract, workers=12):
    (OUT / "original").mkdir(parents=True, exist_ok=True)
    (OUT / "copy").mkdir(parents=True, exist_ok=True)

    def one(tid):
        try:
            cimg, cmeta = token_image(copy_contract, tid)
            oid = number_in(cmeta["name"])
            oid = tid if oid is None else oid
            oimg, ometa = orig_fn(oid)
            cf, of = f"{case}_{tid}.png", f"{case}_{oid}.png"
            cimg.save(OUT / "copy" / cf); oimg.save(OUT / "original" / of)
            return dict(case=case, copy_contract=copy_contract, copy_token=tid,
                        copy_file=cf, copy_meta=cmeta, copy_mint=mint_event(copy_contract, tid),
                        orig_contract=orig_contract, orig_token=oid, orig_file=of,
                        orig_meta=ometa,
                        orig_mint=(mint_event(orig_contract, oid) if orig_contract != PUNKS else None))
        except Exception as e:
            print(f"  {case} {tid}: FAILED {str(e)[:100]}")
            return None

    with ThreadPoolExecutor(workers) as pool:
        res = [r for r in pool.map(one, ids) if r]
    print(f"{case}: {len(res)}/{len(ids)} pairs")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="pairs per case")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resume", action="store_true",
                    help="keep pairs.json and retry only the sampled ids that failed (same sample)")
    ap.add_argument("--prefer-gateway", default=None, help="IPFS gateway to try first")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    pairs = []
    if args.prefer_gateway:
        pv.IPFS_GATEWAYS[:] = [args.prefer_gateway] + [g for g in pv.IPFS_GATEWAYS if g != args.prefer_gateway]
    prev = json.loads((OUT / "pairs.json").read_text()) if args.resume and (OUT / "pairs.json").exists() else []
    done = {(p["case"], p["copy_token"]) for p in prev}
    if prev:
        print(f"resume: keeping {len(prev)} pairs from pairs.json")

    def todo(case, ids):
        return [i for i in ids if (case, i) not in done]

    def bayc(oid):
        img, meta = token_image(BAYC, oid)
        return img, meta
    pairs += [p for p in prev if p["case"] == "rrbayc"]
    pairs += build_case("rrbayc", todo("rrbayc", sorted(rng.sample(range(0, 10000), args.n))),
                        RRBAYC, bayc, BAYC)

    def punk(oid):
        return punk_image(oid), dict(token_uri=f"punkImageSvg({oid})",
                                     media_uri=f"eth_call:{PUNKS_DATA}", name=f"CryptoPunk #{oid}")
    ph = [p for p in prev if p["case"] == "phunks"]
    ph += build_case("phunks", todo("phunks", sorted(rng.sample(range(0, 10000), args.n))), PHUNKS, punk, PUNKS)
    assigns = punk_assign_all()
    print(f"punk Assign events recovered: {len(assigns)}")
    for p in ph:
        a = assigns.get(p["orig_token"])
        if a:
            a = dict(a, timestamp=block_timestamp(a["block"]))
        p["orig_mint"] = a
    pairs += ph

    # sanity: M7 distance between each pair (flip-aware), so bad pairings surface
    from models import M7TransformAwarePHash, M4PHash
    m7, m4 = M7TransformAwarePHash(), M4PHash()
    for p in pairs:
        a = OUT / "original" / p["orig_file"]; b = OUT / "copy" / p["copy_file"]
        p["verify"] = dict(m4=round(m4.distance(m4.fingerprint(a), m4.fingerprint(b)), 4),
                           m7=round(m7.distance(m7.fingerprint(a), m7.fingerprint(b)), 4))
    (OUT / "pairs.json").write_text(json.dumps(pairs, indent=1))
    bad = [p for p in pairs if p["verify"]["m7"] > 0.25]
    print(f"wrote {len(pairs)} pairs; {len(bad)} with M7 distance > 0.25 (inspect these by eye)")


if __name__ == "__main__":
    main()
