"""Image-matching models M1-M10 (Table I of the paper skeleton).

Every model implements the same interface:

    fingerprint(path) -> object            # precomputable representation
    distance(fp_a, fp_b) -> float          # 0 = identical; larger = less similar
    threshold: float                       # distance <= threshold  => duplicate

so evaluate.py can index all reference images once and screen candidates
uniformly.  Distances are normalised per model (hash Hamming distances are
divided by the number of bits, SSIM/cosine similarities become 1 - s).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageOps

HASH_SIZE = 16          # 256-bit perceptual hashes
NBITS = HASH_SIZE ** 2


def _load(path) -> Image.Image:
    return Image.open(path).convert("RGB")


class BaseModel:
    name = "base"
    threshold = 0.0

    def fingerprint(self, path):
        raise NotImplementedError

    def distance(self, a, b) -> float:
        raise NotImplementedError


class M1RawSHA256(BaseModel):
    """Exact-file duplicate baseline: hash of the raw bytes."""
    name = "M1 Raw SHA-256"
    threshold = 0.0

    def fingerprint(self, path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def distance(self, a, b):
        return 0.0 if a == b else 1.0


class M2CanonicalHash(BaseModel):
    """Grayscale + fixed resize + SHA-256 of decoded pixels.

    Robust to metadata and container changes; brittle to any pixel change.
    """
    name = "M2 Canonical Hash"
    threshold = 0.0

    def fingerprint(self, path):
        img = ImageOps.grayscale(_load(path)).resize((256, 256), Image.LANCZOS)
        return hashlib.sha256(img.tobytes()).hexdigest()

    def distance(self, a, b):
        return 0.0 if a == b else 1.0


class M3ColorQuantHash(BaseModel):
    """Coarse color quantisation (4 bits/channel) + resize + SHA-256."""
    name = "M3 Color Quant."
    threshold = 0.0

    def fingerprint(self, path):
        img = _load(path).resize((64, 64), Image.LANCZOS)
        arr = (np.array(img) >> 4).astype(np.uint8)  # 16 levels per channel
        return hashlib.sha256(arr.tobytes()).hexdigest()

    def distance(self, a, b):
        return 0.0 if a == b else 1.0


class M4PHash(BaseModel):
    """DCT perceptual hash; normalised Hamming distance."""
    name = "M4 pHash"
    threshold = 0.15

    def fingerprint(self, path):
        return imagehash.phash(_load(path), hash_size=HASH_SIZE)

    def distance(self, a, b):
        return (a - b) / NBITS


class M5DHash(BaseModel):
    """Difference hash: fast gradient-based structural hash."""
    name = "M5 dHash"
    threshold = 0.15

    def fingerprint(self, path):
        return imagehash.dhash(_load(path), hash_size=HASH_SIZE)

    def distance(self, a, b):
        return (a - b) / NBITS


class M6EdgePHash(BaseModel):
    """Canny edge map + pHash: shape-based matching, color-invariant."""
    name = "M6 Edge + pHash"
    threshold = 0.22

    def fingerprint(self, path):
        import cv2
        gray = np.array(ImageOps.grayscale(_load(path)).resize((256, 256), Image.LANCZOS))
        edges = cv2.Canny(gray, 100, 200)
        return imagehash.phash(Image.fromarray(edges), hash_size=HASH_SIZE)

    def distance(self, a, b):
        return (a - b) / NBITS


class M7TransformAwarePHash(BaseModel):
    """pHash computed over the dihedral group + color inversion of the query.

    A candidate matches if ANY of its 16 variants (4 rotations x mirror x
    inversion) is close to the reference hash, catching flip/rotate/invert
    evasion that plain pHash misses.
    """
    name = "M7 Transform-aware pHash"
    threshold = 0.15
    _TRANSPOSES = [None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]

    def _variants(self, img):
        for base in (img, ImageOps.mirror(img)):
            for base2 in (base, ImageOps.invert(base)):
                for t in self._TRANSPOSES:
                    yield base2 if t is None else base2.transpose(t)

    def fingerprint(self, path):
        img = _load(path)
        return [imagehash.phash(v, hash_size=HASH_SIZE) for v in self._variants(img)]

    def distance(self, a, b):
        # a, b are variant lists; compare canonical (first) hash of the
        # reference against every variant of the candidate.
        return min((a[0] - vb) / NBITS for vb in b)


class M8SSIM(BaseModel):
    """Structural similarity on grayscale 256x256; distance = 1 - SSIM."""
    name = "M8 SSIM"
    threshold = 0.25

    def fingerprint(self, path):
        img = ImageOps.grayscale(_load(path)).resize((256, 256), Image.LANCZOS)
        return np.array(img)

    def distance(self, a, b):
        from skimage.metrics import structural_similarity
        return 1.0 - structural_similarity(a, b, data_range=255)


class M9ORB(BaseModel):
    """ORB keypoints + ratio-test matching; distance = 1 - inlier match ratio."""
    name = "M9 ORB"
    threshold = 0.85

    def fingerprint(self, path):
        import cv2
        gray = np.array(ImageOps.grayscale(_load(path)).resize((256, 256), Image.LANCZOS))
        orb = cv2.ORB_create(nfeatures=500)
        _, desc = orb.detectAndCompute(gray, None)
        return desc

    def distance(self, a, b):
        import cv2
        if a is None or b is None or len(a) < 2 or len(b) < 2:
            return 1.0
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.knnMatch(a, b, k=2)
        good = sum(1 for pair in matches if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance)
        return 1.0 - good / max(min(len(a), len(b)), 1)


class M10CLIP(BaseModel):
    """OpenCLIP ViT-B/32 image embeddings; distance = 1 - cosine similarity."""
    name = "M10 CLIP"
    threshold = 0.10
    _model = None

    @classmethod
    def _ensure(cls):
        if cls._model is None:
            import open_clip
            import torch
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k")
            model.eval()
            cls._model = (model, preprocess, torch)
        return cls._model

    def fingerprint(self, path):
        model, preprocess, torch = self._ensure()
        with torch.no_grad():
            t = preprocess(_load(path)).unsqueeze(0)
            emb = model.encode_image(t)[0]
            emb = emb / emb.norm()
        return emb.numpy().astype(np.float32)

    def distance(self, a, b):
        return float(1.0 - np.dot(a, b))


class M11SSCD(BaseModel):
    """SSCD (Self-Supervised Copy Detection) ResNet-50 descriptors.

    Uses the released `sscd_disc_mixup` TorchScript model (Pizzi et al.,
    CVPR 2022), a ResNet-50 trained specifically for copy detection with
    entropy-regularized embeddings.  Descriptors are 512-d, L2-normalised;
    distance = 1 - cosine similarity.  Weights are fetched once into
    models_cache/ if absent.
    """
    name = "M11 SSCD"
    threshold = 0.50
    _model = None
    _URL = ("https://dl.fbaipublicfiles.com/sscd-copy-detection/"
            "sscd_disc_mixup.torchscript.pt")

    @classmethod
    def _ensure(cls):
        if cls._model is None:
            import torch
            from torchvision import transforms
            path = Path(__file__).parent.parent / "models_cache" / "sscd_disc_mixup.torchscript.pt"
            if not path.exists():
                import urllib.request
                path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(cls._URL, path)
            model = torch.jit.load(str(path))
            model.eval()
            preprocess = transforms.Compose([
                transforms.Resize((288, 288)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            cls._model = (model, preprocess, torch)
        return cls._model

    def fingerprint(self, path):
        model, preprocess, torch = self._ensure()
        with torch.no_grad():
            t = preprocess(_load(path)).unsqueeze(0)
            emb = model(t)[0]
            emb = emb / emb.norm()
        return emb.numpy().astype(np.float32)

    def distance(self, a, b):
        return float(1.0 - np.dot(a, b))


ALL_MODELS = [M1RawSHA256, M2CanonicalHash, M3ColorQuantHash, M4PHash, M5DHash,
              M6EdgePHash, M7TransformAwarePHash, M8SSIM, M9ORB, M10CLIP,
              M11SSCD]


def build_models(include_clip: bool = True):
    models = []
    for cls in ALL_MODELS:
        if cls is M10CLIP and not include_clip:
            continue
        models.append(cls())
    return models
