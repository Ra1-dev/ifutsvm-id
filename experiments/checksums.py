"""Content checksums for invalidating fitted-result caches."""
import hashlib
from pathlib import Path


def source_file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
