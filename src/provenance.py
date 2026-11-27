"""On-chain provenance utilities.

Resolves NFT token data directly from public blockchain infrastructure:
  - tokenURI(uint256) / uri(uint256) via JSON-RPC eth_call (no API key needed)
  - mint block / timestamp / minter via Blockscout's free transfer API
  - ipfs:// and ar:// URIs via a fallback list of public gateways
  - data: URIs (fully on-chain NFTs such as Loot and Nouns) decoded locally

Every function is chain-agnostic: pass a chain name registered in CHAINS.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests

USER_AGENT = "nft-provenance-screening/1.0 (academic research)"

CHAINS = {
    "ethereum": {
        "chain_id": 1,
        "rpc": [
            "https://ethereum-rpc.publicnode.com",
            "https://eth.drpc.org",
            "https://1rpc.io/eth",
        ],
        "blockscout": "https://eth.blockscout.com/api",
    },
    "polygon": {
        "chain_id": 137,
        "rpc": [
            "https://polygon-bor-rpc.publicnode.com",
            "https://polygon.drpc.org",
            "https://1rpc.io/matic",
        ],
        "blockscout": "https://polygon.blockscout.com/api",
    },
}

IPFS_GATEWAYS = [
    "https://ipfs.io/ipfs/",
    "https://gateway.pinata.cloud/ipfs/",
    "https://dweb.link/ipfs/",
    "https://ipfs.filebase.io/ipfs/",
    "https://nftstorage.link/ipfs/",
]

# keccak256("Transfer(address,address,uint256)")
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
TOKENURI_SELECTOR = "0xc87b56dd"  # tokenURI(uint256)


@dataclass
class TokenProvenance:
    chain: str
    contract: str
    token_id: int
    collection: str = ""
    category: str = ""
    token_uri: str = ""
    media_uri: str = ""
    mint_block: Optional[int] = None
    mint_timestamp: Optional[int] = None
    mint_tx: str = ""
    minter: str = ""
    media_sha256: str = ""
    local_file: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def rpc_call(chain: str, method: str, params: list, timeout: int = 25):
    """JSON-RPC call with endpoint fallback."""
    last_err = None
    for url in CHAINS[chain]["rpc"]:
        try:
            r = requests.post(
                url,
                json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                headers={"User-Agent": USER_AGENT},
                timeout=timeout,
            )
            body = r.json()
            if "result" in body:
                return body["result"]
            last_err = body.get("error")
        except Exception as e:  # network / JSON errors -> try next endpoint
            last_err = str(e)
    raise RuntimeError(f"RPC {method} failed on all {chain} endpoints: {last_err}")


def _decode_string_return(hexdata: str) -> str:
    """Decode a single ABI-encoded string return value."""
    raw = bytes.fromhex(hexdata[2:] if hexdata.startswith("0x") else hexdata)
    if len(raw) < 64:
        return ""
    offset = int.from_bytes(raw[0:32], "big")
    length = int.from_bytes(raw[offset:offset + 32], "big")
    return raw[offset + 32:offset + 32 + length].decode("utf-8", errors="replace")


def get_token_uri(chain: str, contract: str, token_id: int) -> str:
    """Read tokenURI(tokenId) straight from the contract via eth_call."""
    data = TOKENURI_SELECTOR + format(token_id, "064x")
    result = rpc_call(chain, "eth_call", [{"to": contract, "data": data}, "latest"])
    return _decode_string_return(result)


def get_mint_events(chain: str, contract: str, max_pages: int = 5, page_size: int = 100) -> dict:
    """First mint (Transfer from 0x0) per token id, via Blockscout's free API.

    Returns {token_id: {"block": int, "timestamp": int, "tx": str, "minter": str}}.
    Only covers the earliest `max_pages * page_size` transfers of the collection,
    which is enough for low token ids sampled in this study.
    """
    api = CHAINS[chain]["blockscout"]
    mints: dict[int, dict] = {}
    for page in range(1, max_pages + 1):
        try:
            r = requests.get(
                api,
                params={
                    "module": "account", "action": "tokennfttx",
                    "contractaddress": contract,
                    "startblock": 0, "endblock": 99999999,
                    "page": page, "offset": page_size, "sort": "asc",
                },
                headers={"User-Agent": USER_AGENT},
                timeout=30,
            )
            rows = r.json().get("result") or []
        except Exception:
            break
        if not isinstance(rows, list) or not rows:
            break
        for row in rows:
            if row.get("from") != "0x0000000000000000000000000000000000000000":
                continue  # secondary transfer, not a mint
            tid = int(row["tokenID"])
            if tid not in mints:
                mints[tid] = {
                    "block": int(row["blockNumber"]),
                    "timestamp": int(row.get("timeStamp", 0)) or None,
                    "tx": row.get("hash", ""),
                    "minter": row.get("to", ""),
                }
        if len(rows) < page_size:
            break
        time.sleep(0.4)  # stay polite to the free API
    return mints


def resolve_uri(uri: str, timeout: int = 30) -> bytes:
    """Fetch the content behind a token/media URI (ipfs://, http(s)://, data:)."""
    if uri.startswith("data:"):
        header, _, payload = uri.partition(",")
        if ";base64" in header:
            return base64.b64decode(payload)
        return requests.utils.unquote(payload).encode()
    if uri.startswith("ipfs://"):
        path = uri[len("ipfs://"):]
        if path.startswith("ipfs/"):
            path = path[len("ipfs/"):]
        last_err = None
        for gw in IPFS_GATEWAYS:
            try:
                r = requests.get(gw + path, headers={"User-Agent": USER_AGENT}, timeout=timeout)
                if r.status_code == 200 and r.content:
                    return r.content
                last_err = f"HTTP {r.status_code}"
            except Exception as e:
                last_err = str(e)
        raise RuntimeError(f"all IPFS gateways failed for {uri}: {last_err}")
    if uri.startswith("ar://"):
        uri = "https://arweave.net/" + uri[len("ar://"):]
    r = requests.get(uri, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    return r.content


def resolve_metadata(token_uri: str) -> dict:
    """Fetch and parse the JSON metadata behind a tokenURI."""
    return json.loads(resolve_uri(token_uri).decode("utf-8", errors="replace"))


def extract_media_uri(metadata: dict) -> str:
    """Pull the media URL out of collection-specific metadata layouts."""
    for key in ("image", "image_url", "imageUrl", "image_data", "animation_url"):
        val = metadata.get(key)
        if isinstance(val, str) and val:
            # image_data is raw SVG markup rather than a URI
            if key == "image_data" and not val.startswith(("http", "ipfs", "data:")):
                return "data:image/svg+xml;utf8," + val
            return val
    return ""
