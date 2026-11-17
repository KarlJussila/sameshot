"""Candidate retrieval: SSCD copy-detection embeddings with a deliberately loose threshold."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MODEL = 'sscd_disc_mixup'
URL = f'https://dl.fbaipublicfiles.com/sscd-copy-detection/{MODEL}.torchscript.pt'
VERSION = f'{MODEL}-320-mirror-v1'
THRESHOLD = .70
MEAN = np.array([.485, .456, .406], np.float32)[:, None, None]
STD = np.array([.229, .224, .225], np.float32)[:, None, None]


def views(path) -> np.ndarray:
    """The image and its mirror, resized and normalized for SSCD."""
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert('RGB').resize((320, 320), Image.Resampling.BILINEAR)
    pixels = (np.asarray(rgb, np.float32).transpose(2, 0, 1) / 255 - MEAN) / STD
    return np.stack([pixels, pixels[:, :, ::-1]])


def resolve_device(device='auto') -> str:
    import torch
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def load_model(device):
    """The TorchScript model, downloaded once into torch's hub directory."""
    import torch
    weights = Path(torch.hub.get_dir()) / 'checkpoints' / f'{MODEL}.torchscript.pt'
    if not weights.exists():
        weights.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(URL, str(weights))
    return torch.jit.load(str(weights), map_location=device).eval()


def embed(images: dict, *, device='auto', batch_size=16):
    """Yield (sha, vectors, error) for {sha: path}; vectors are (2, 512) unit rows: image, mirror."""
    import torch
    device = resolve_device(device)
    model = load_model(device)

    def prepare(item):
        sha, path = item
        try:
            return sha, views(path), None
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            return sha, None, f'{type(error).__name__}: {error}'

    remaining = iter(images.items())
    try:
        with ThreadPoolExecutor(8) as pool, torch.inference_mode():
            while batch := list(islice(remaining, batch_size)):
                prepared = list(pool.map(prepare, batch))
                for sha, _, error in prepared:
                    if error:
                        yield sha, None, error
                ready = [(sha, pixels) for sha, pixels, _ in prepared if pixels is not None]
                if not ready:
                    continue
                inputs = torch.from_numpy(np.concatenate([pixels for _, pixels in ready])).to(device)
                vectors = torch.nn.functional.normalize(model(inputs), dim=1).cpu().numpy()
                for (sha, _), vector in zip(ready, vectors.reshape(len(ready), 2, -1)):
                    yield sha, vector, None
    finally:
        del model
        if device.startswith('cuda'):
            torch.cuda.empty_cache()


def candidates(vectors: np.ndarray, threshold=THRESHOLD) -> list[tuple[float, int, int]]:
    """(score, i, j) for i < j whose best image/mirror cosine reaches the threshold, best first."""
    edges = []
    # Score in row blocks so the temporary matrix stays small for large folders.
    for start in range(0, len(vectors), 256):
        block = vectors[start:start + 256]
        scores = np.maximum.reduce([block[:, a] @ vectors[:, b].T for a in range(2) for b in range(2)])
        rows, columns = np.where(scores >= threshold)
        edges.extend((float(scores[row, column]), start + row, int(column))
                     for row, column in zip(rows, columns) if start + row < column)
    return sorted(edges, reverse=True)
