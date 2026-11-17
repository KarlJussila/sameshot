"""Scan files and folders for copies of the same photo."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import multiprocessing
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from . import __version__, align, embed
from .store import Cache

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff', '.gif', '.avif', '.heic', '.heif'}


def log(message):
    print(message, file=sys.stderr, flush=True)


def find_images(paths) -> list[Path]:
    """Files given directly, plus images found recursively in folders, without repeats."""
    readable = EXTENSIONS & set(Image.registered_extensions())
    found = {}
    for path in map(Path, paths):
        if path.is_dir():
            for file in sorted(path.rglob('*')):
                if file.suffix.lower() in readable and file.is_file():
                    found.setdefault(file.resolve(), file)
        elif path.is_file():
            found.setdefault(path.resolve(), path)
        else:
            raise FileNotFoundError(path)
    return list(found.values())


def scan(paths, *, cache_dir=None, device='auto', threshold=embed.THRESHOLD, min_overlap=align.MIN_OVERLAP,
         min_matching_tiles=align.MIN_MATCHING_TILES, workers=None, batch_size=16, log=log) -> dict:
    """Group copies of the same photo. Returns the report that the CLI writes as JSON."""
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must be between 0 and 1')
    workers = workers or min(8, os.cpu_count() or 1)
    cache = Cache(cache_dir)
    files = find_images(paths)
    log(f'{len(files)} images')
    hashes, failures = cache.hashes(files)
    errors = {str(path): error for path, error in failures.items()}
    copies = {}
    for path, sha in hashes.items():
        copies.setdefault(sha, []).append(path)
    photos = {sha: str(paths[0]) for sha, paths in copies.items()}

    vectors = cache.embeddings(photos, embed.VERSION)
    missing = {sha: path for sha, path in photos.items() if sha not in vectors}
    if missing:
        log(f'SSCD: embedding {len(missing)} images ({len(vectors)} cached)')
        for done, (sha, vector, error) in enumerate(embed.embed(missing, device=device, batch_size=batch_size), 1):
            if error:
                errors.update(dict.fromkeys(map(str, copies.pop(sha)), error))
            else:
                vectors[sha] = vector
                cache.add_embedding(sha, embed.VERSION, vector)
            if done % 200 == 0:
                cache.commit()
                log(f'SSCD: {done}/{len(missing)}')
        cache.commit()

    shas = sorted(vectors.keys() & copies.keys())
    edges = [(score, shas[i], shas[j]) for score, i, j in
             (embed.candidates(np.stack([vectors[sha] for sha in shas]), threshold) if len(shas) > 1 else [])]
    measured = cache.pairs(align.VERSION)
    # Sorted so neighboring work shares images and hits each worker's decode cache.
    pending = sorted({(a, b) for _, a, b in edges} - measured.keys())
    log(f'Alignment: {len(edges)} candidate pairs, {len(pending)} to check')
    if pending:
        context = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(workers, mp_context=context, initializer=align.init_worker) as pool:
            results = pool.map(align.measure, repeat(str(cache.directory)),
                               ((a, photos[a]) for a, _ in pending), ((b, photos[b]) for _, b in pending),
                               chunksize=8)
            for done, (pair, options) in enumerate(zip(pending, results), 1):
                measured[pair] = options
                cache.add_pair(*pair, align.VERSION, options)
                if done % 200 == 0:
                    cache.commit()
                    log(f'Alignment: {done}/{len(pending)}')
        cache.commit()

    parent = {sha: sha for sha in copies}

    def root(sha):
        while parent[sha] != sha:
            parent[sha] = parent[parent[sha]]
            sha = parent[sha]
        return sha

    pairs = []
    for score, a, b in edges:
        options = measured[a, b]
        same = align.is_same(options, min_overlap, min_matching_tiles)
        pairs.append({'a': photos[a], 'b': photos[b], 'similarity': round(score, 4), 'same': same,
                      'options': options})
        if same:
            parent[root(b)] = root(a)

    families = {}
    for sha in copies:
        families.setdefault(root(sha), []).append(sha)
    groups = []
    for family in families.values():
        if len(family) == 1 and len(copies[family[0]]) == 1:
            continue
        measurements = {sha: align.features(cache.directory, sha, photos[sha]) for sha in family}
        members = sorted((str(path), sha) for sha in family for path in copies[sha])
        groups.append({'files': [{'path': path, 'sha256': sha, 'width': measurements[sha]['width'],
                                  'height': measurements[sha]['height']} for path, sha in members]})
    groups.sort(key=lambda group: group['files'][0]['path'])
    return {'sameshot': __version__,
            'settings': {'threshold': threshold, 'min_overlap': min_overlap,
                         'min_matching_tiles': min_matching_tiles},
            'images': len(files), 'groups': groups, 'pairs': pairs,
            'errors': [{'path': path, 'error': error} for path, error in sorted(errors.items())]}
