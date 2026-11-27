# Cross-Platform NFT Provenance Inference for Duplicate and Near-Duplicate Digital Media

Implementation for the paper *"Cross-Platform NFT Provenance Inference for Duplicate
and Near-Duplicate Digital Media"* (F. Ogunwoye, Y. Madhwal, K. Rehman,
Y. Yanovich; IEEE-Blockchain 2026). The system answers a practical buyer-side
question: **is the NFT image I'm about to buy a duplicate of media that was already
minted, and if so, what is the earliest observable matching mint within the
indexed registry?** It does not prove authorship: an earlier mint may exist outside
the registry, and the artwork may predate every mint.

Blockchains prove *ownership* of a token, not *originality* of its media: the same
image can be re-minted under a different contract, token ID, or storage URL, and a
single re-saved JPEG already breaks raw file hashing. This repo builds the full
experimental pipeline and adds an on-chain provenance layer on top of it.

## Camera-ready benchmarks

The controlled benchmark uses 100 reference NFTs, 800 transformed candidates, and
40 genuine negatives. The CLIP cascade achieves F1 0.987 at a mean 166 ms per query.
The registry-scale benchmark uses 3,438 entries, 800 transformed candidates, and
877 genuine negatives. The results below use fixed thresholds.

| System | Visual only: F1 / FPR | Collection-aware: F1 / FPR |
|---|---|---|
| M7 transform-aware pHash | 0.860 / 0.106 | 0.914 / 0.000 |
| M10 CLIP | 0.728 / 0.339 | 0.880 / 0.000 |
| M11 SSCD | 0.641 / 0.927 | 0.967 / 0.014 |
| **Cascade, CLIP final tier** | 0.817 / 0.376 | **0.987 / 0.000** (0/877) |
| Cascade, SSCD final tier | 0.663 / 0.927 | **0.993** / 0.014 |

The filter excludes the candidate's contract, so within-contract duplicates fall
outside its search scope. In two real copymint series (137 RR/BAYC and 131
CryptoPhunks pairs), the earliest matching mint after filtering belonged to the
original collection in all 268 cases. It was the exact source token in 247 cases.
The D3 benchmark separately measures compound, adaptive, and generative edits.

## Replot the benchmark results

From the repository root:

```sh
python -m pip install -r requirements-figures.txt
python src/reproduce_paper_figures.py
```

This writes the controlled-distortion heatmap, registry-scale false-positive
plot, and D3 detection heatmap to `results/figures/`. Plots use the committed
measurements. The D3 figure omits the paper's NFT example thumbnails.

## Data and code

| Path | Contents |
|---|---|
| `src/` | Registry collection, fingerprinting, evaluation, and plotting code |
| `data/` | Token provenance, media references, and ground-truth manifests |
| `results/` | Controlled-benchmark measurements and tables |
| `results/large/` | Registry-scale, D3, and copymint results and cached fingerprints |

The public snapshot excludes third-party artwork, transformed images, and
paper PDFs. The manifests retain remote references and media hashes when
available. Inline data URIs are replaced with `redacted-inline-media:sha256:`
references. To recover on-chain inline media, resolve the token again from its
recorded chain, contract, and token ID. Network availability, token metadata
changes, and rights in the source artwork may affect independent media retrieval.
Cached fingerprints and the published measurements are included. The live
fetching and media-dependent benchmark scripts require separately obtained inputs.

The [MIT License](LICENSE) covers the project code. References to third-party NFT
artwork grant no permission to reuse that artwork.
