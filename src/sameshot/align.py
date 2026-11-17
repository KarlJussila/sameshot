"""Verification: do two images share the same pixels under one similarity transform?

SIFT matches fit a single translation, rotation, and uniform scale (also tried
mirrored). The image showing more of the scene is warped onto the one showing
less, and the latter's central content is cut into tiles. Two images are the same photo when the warp covers
nearly all of that content and most tiles correlate almost perfectly. Crops,
resizing, recompression, watermarks, and color edits pass; a different shot
fails because the subject or camera moved.
"""

from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageOps

from .store import default_cache_dir, file_hash

FEATURES = 'sift2500-1400-v1'
CHECK = 'smaller-ref-center80-tiles8x6-corr099-matched-v2'
VERSION = f'{FEATURES}/{CHECK}'
MAX_EDGE = 1400
MIN_OVERLAP = .95
MIN_MATCHING_TILES = .75


def resize(image, scale):
    height, width = image.shape[:2]
    return cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


@lru_cache(maxsize=8)
def load(path) -> tuple[np.ndarray, tuple[int, int]]:
    """RGB pixels with the long edge capped at MAX_EDGE, and the original (width, height)."""
    with Image.open(path) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
    height, width = rgb.shape[:2]
    if max(width, height) > MAX_EDGE:
        rgb = resize(rgb, MAX_EDGE / max(width, height))
    return rgb, (width, height)


def pixels(path, mirror=False) -> np.ndarray:
    rgb = load(str(path))[0]
    return cv2.flip(rgb, 1) if mirror else rgb


def compute_features(path, mirror=False) -> dict:
    rgb = pixels(path, mirror)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    keys, descriptors = cv2.SIFT_create(nfeatures=2500).detectAndCompute(gray, None)
    width, height = load(str(path))[1]
    return {'points': np.array([key.pt for key in keys], np.float32).reshape(-1, 2),
            'descriptors': descriptors.astype(np.uint8) if descriptors is not None else np.empty((0, 128), np.uint8),
            'width': width, 'height': height}


def features(cache_dir, sha, path, mirror=False) -> dict:
    """SIFT keypoints and the original size, cached by content hash."""
    directory = Path(cache_dir) / 'features' / FEATURES
    file = directory / f'{sha}{".mirror" if mirror else ""}.npz'
    if not file.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, suffix='.npz', delete=False) as temporary:
            np.savez_compressed(temporary, **compute_features(path, mirror))
        Path(temporary.name).replace(file)
    with np.load(file) as saved:
        value = {key: saved[key] for key in saved.files}
    return {**value, 'width': int(value['width']), 'height': int(value['height'])}


def similarity_transform(left, right):
    """Map right's points onto left's with one translation, rotation, and uniform scale."""
    if len(left['descriptors']) < 6 or len(right['descriptors']) < 6:
        return None, 0
    matches = cv2.BFMatcher().knnMatch(right['descriptors'].astype(np.float32),
                                      left['descriptors'].astype(np.float32), k=2)
    good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < .7 * pair[1].distance]
    if len(good) < 6:
        return None, 0
    dst = np.array([left['points'][match.trainIdx] for match in good], np.float32)
    src = np.array([right['points'][match.queryIdx] for match in good], np.float32)
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=1.5)
    count = int(inliers.sum()) if inliers is not None else 0
    if matrix is None or count < 6 or not 1 / 20 < math.sqrt(abs(np.linalg.det(matrix[:, :2]))) < 20:
        return None, count
    return matrix, count


def matched_resolution(other, reference, matrix):
    """Compare at the coarser of the two resolutions, so a small copy isn't faulted for missing detail.

    A copy the warp would upscale is compared against a reference degraded the
    same way; a copy the warp would shrink is first downsampled without aliasing.
    """
    scale = math.sqrt(abs(np.linalg.det(matrix[:, :2])))
    if scale > 1.05:
        height, width = reference.shape[:2]
        return other, cv2.resize(resize(reference, 1 / scale), (width, height), interpolation=cv2.INTER_LINEAR), matrix
    if scale < 1 / 1.05:
        small = resize(other, scale)
        # Map the shrunken image's pixels back to the original's exactly, including
        # resize's half-pixel offset; even a 0.15px error costs tiles their 0.99.
        sx, sy = small.shape[1] / other.shape[1], small.shape[0] / other.shape[0]
        warp = matrix.copy()
        warp[:, 0] /= sx
        warp[:, 1] /= sy
        warp[:, 2] += matrix[:, :2] @ np.array([.5 / sx - .5, .5 / sy - .5])
        return small, reference, warp
    return other, reference, matrix


def tile_correlations(original, compared, mask) -> list[float]:
    """Best per-channel correlation for each 8x6 tile that is at least half valid content."""
    height, width = mask.shape
    rows, columns = np.linspace(0, height, 9, dtype=int), np.linspace(0, width, 7, dtype=int)
    correlations = []
    for top, bottom in zip(rows[:-1], rows[1:]):
        for start, end in zip(columns[:-1], columns[1:]):
            tile = mask[top:bottom, start:end]
            if tile.sum() < tile.size * .5:
                continue
            x = original[top:bottom, start:end][tile].astype(np.float64)
            y = compared[top:bottom, start:end][tile].astype(np.float64)
            # Flat channels carry no evidence either way.
            usable = (x.std(0) >= 5) & (y.std(0) >= 5)
            if not usable.any():
                continue
            x, y = x[:, usable] - x[:, usable].mean(0), y[:, usable] - y[:, usable].mean(0)
            correlations.append(float(((x * y).sum(0) / np.sqrt((x * x).sum(0) * (y * y).sum(0))).max()))
    return correlations


def content_mask(gray) -> np.ndarray:
    """The central 80%, minus flat dark areas such as letterboxing."""
    height, width = gray.shape
    central = np.zeros(gray.shape, bool)
    central[round(.1 * height):round(.9 * height), round(.1 * width):round(.9 * width)] = True
    level = gray.astype(np.float32)
    texture = cv2.GaussianBlur(level ** 2, (15, 15), 0) - cv2.GaussianBlur(level, (15, 15), 0) ** 2
    return central & ((gray > 25) | (texture > 36))


def check(reference, other, matrix) -> tuple[float, list[float]]:
    """Warp other onto reference: the share of reference's content covered, and the tile correlations."""
    height, width = reference.shape[:2]
    content = content_mask(cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY))
    other, reference, warp = matched_resolution(other, reference, matrix)
    warped = cv2.warpAffine(other, warp, (width, height))
    valid = cv2.warpAffine(np.ones(other.shape[:2], np.uint8), warp, (width, height))
    mask = (cv2.erode(valid, np.ones((9, 9), np.uint8)) > 0) & content
    correlations = tile_correlations(cv2.GaussianBlur(reference, (3, 3), .6),
                                     cv2.GaussianBlur(warped, (3, 3), .6), mask)
    return float(mask.sum() / max(1, content.sum())), correlations


def measure(cache_dir, left, right) -> list[dict]:
    """Alignment measurements for a pair, as is and with right mirrored. left and right are (sha, path).

    Whichever image shows less of the scene (a crop, or one photo inside a
    collage) is the reference, so the result doesn't depend on which file is
    which.
    """
    left_features, left_color = features(cache_dir, *left), pixels(left[1])
    options = []
    for mirror in (False, True):
        matrix, inliers = similarity_transform(left_features, features(cache_dir, *right, mirror=mirror))
        if matrix is None:
            continue
        right_color = pixels(right[1], mirror)
        # Compare how much of the scene each frame covers, in left's coordinates.
        scale = math.sqrt(abs(np.linalg.det(matrix[:, :2])))
        right_is_smaller = (scale ** 2 * right_color.shape[0] * right_color.shape[1]
                            < left_color.shape[0] * left_color.shape[1])
        if right_is_smaller:
            overlap, correlations = check(right_color, left_color, cv2.invertAffineTransform(matrix))
        else:
            overlap, correlations = check(left_color, right_color, matrix)
        options.append({
            'mirror': mirror, 'reference': 'b' if right_is_smaller else 'a', 'inliers': inliers,
            'tiles': len(correlations),
            'matching_tiles': round(float(np.mean(np.array(correlations) >= .99)) if correlations else 0., 4),
            'overlap': round(overlap, 4), 'matrix': np.round(matrix, 5).tolist()})
    return options


def is_same(options, min_overlap=MIN_OVERLAP, min_matching_tiles=MIN_MATCHING_TILES) -> bool:
    return any(option['overlap'] >= min_overlap and option['matching_tiles'] >= min_matching_tiles
               for option in options)


def init_worker():
    cv2.setNumThreads(1)


def compare(a, b, *, cache_dir=None, min_overlap=MIN_OVERLAP, min_matching_tiles=MIN_MATCHING_TILES) -> dict:
    """Check one pair of files directly, without candidate retrieval."""
    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
    left, right = sorted([(file_hash(a), str(a)), (file_hash(b), str(b))])
    if left[0] == right[0]:
        return {'a': left[1], 'b': right[1], 'same': True, 'identical': True, 'options': []}
    options = measure(cache_dir, left, right)
    return {'a': left[1], 'b': right[1], 'same': is_same(options, min_overlap, min_matching_tiles),
            'identical': False, 'options': options}
