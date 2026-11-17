"""The cache: one SQLite file for file hashes, embeddings, and pair measurements.

SIFT features live beside it as .npz files. Everything here is derived and can
be deleted; it is keyed by content hash and algorithm version.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import numpy as np

SCHEMA = '''
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, size INTEGER, mtime_ns INTEGER, sha256 TEXT);
CREATE TABLE IF NOT EXISTS embeddings (sha256 TEXT, version TEXT, vectors BLOB, PRIMARY KEY (sha256, version));
CREATE TABLE IF NOT EXISTS pairs (a TEXT, b TEXT, version TEXT, options TEXT, PRIMARY KEY (a, b, version));
'''


def default_cache_dir() -> Path:
    return Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'sameshot'


def file_hash(path) -> str:
    with open(path, 'rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


class Cache:
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory else default_cache_dir()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.directory / 'cache.sqlite3')
        self.db.executescript(SCHEMA)

    def hashes(self, paths) -> tuple[dict[Path, str], dict[Path, str]]:
        """SHA-256 for each path, rehashing only files whose size or mtime changed."""
        known = {row[0]: row[1:] for row in self.db.execute('SELECT path, size, mtime_ns, sha256 FROM files')}
        hashes, errors, pending = {}, {}, []
        for path in paths:
            try:
                stat = path.stat()
            except OSError as error:
                errors[path] = str(error)
                continue
            key = str(path.resolve())
            row = known.get(key)
            if row and row[:2] == (stat.st_size, stat.st_mtime_ns):
                hashes[path] = row[2]
            else:
                pending.append((path, key, stat))

        def digest(item):
            path, key, stat = item
            try:
                return item, file_hash(path), None
            except OSError as error:
                return item, None, str(error)

        with ThreadPoolExecutor(8) as pool:
            for (path, key, stat), sha, error in pool.map(digest, pending):
                if error:
                    errors[path] = error
                    continue
                hashes[path] = sha
                self.db.execute('INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?)',
                                (key, stat.st_size, stat.st_mtime_ns, sha))
        self.db.commit()
        return hashes, errors

    def embeddings(self, shas, version) -> dict[str, np.ndarray]:
        wanted = set(shas)
        return {sha: np.frombuffer(blob, np.float32).reshape(2, -1) for sha, blob in self.db.execute(
            'SELECT sha256, vectors FROM embeddings WHERE version = ?', (version,)) if sha in wanted}

    def add_embedding(self, sha, version, vectors):
        self.db.execute('INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)',
                        (sha, version, np.asarray(vectors, np.float32).tobytes()))

    def pairs(self, version) -> dict[tuple[str, str], list]:
        return {(a, b): json.loads(options) for a, b, options in self.db.execute(
            'SELECT a, b, options FROM pairs WHERE version = ?', (version,))}

    def add_pair(self, a, b, version, options):
        self.db.execute('INSERT OR REPLACE INTO pairs VALUES (?, ?, ?, ?)', (a, b, version, json.dumps(options)))

    def commit(self):
        self.db.commit()
