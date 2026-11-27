"""Demonstrate who-minted-first provenance resolution on a remint scenario.

A copymint is simulated by adding a second registry entry that reuses the
media of a real token (default: the first registry entry with a known mint
timestamp) under a fictitious contract with a mint timestamp one year later.
The candidate image (a JPEG-recompressed copy of the media) then matches BOTH
entries, and the resolver must attribute originality to the earlier mint.

The injected entry is clearly marked "SIMULATED-REMINT" — real cross-contract
copymints exist on-chain, but hard-coding one would make the benchmark depend
on a specific scam contract staying resolvable.

Run:  python src/demo_remint.py
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import screen as screening


def main():
    data = Path(__file__).parent.parent / "data"
    registry = json.loads((data / "registry.json").read_text())
    original = next(e for e in registry if e.get("mint_timestamp"))

    remint = dict(original)
    remint.update({
        "collection": original["collection"] + " (SIMULATED-REMINT)",
        "contract": "0x000000000000000000000000000000000000dEaD",
        "chain": "polygon",
        "token_id": 777,
        "mint_timestamp": original["mint_timestamp"] + 365 * 24 * 3600,
        "mint_block": None,
        "mint_tx": "(simulated)",
        "local_file": "remint__" + original["local_file"],
    })

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "registry").mkdir()
        # same media under both entries
        for e in (original, remint):
            src = data / "registry" / original["local_file"]
            (tmp / "registry" / e["local_file"]).write_bytes(src.read_bytes())
        (tmp / "registry.json").write_text(json.dumps([original, remint]))

        # candidate: recompressed copy of the media, as a buyer would receive it
        img = Image.open(data / "registry" / original["local_file"]).convert("RGB")
        cand = tmp / "candidate.jpg"
        img.save(cand, format="JPEG", quality=80)

        report = screening.screen(cand, tmp)
        screening.print_report(report)

        assert report["presumed_original"]["contract"] == original["contract"], \
            "resolver failed to pick the earlier mint!"
        print("\n[demo] resolver correctly attributed the earliest mint as original.")


if __name__ == "__main__":
    main()
