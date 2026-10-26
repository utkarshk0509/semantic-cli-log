import hashlib
import json
from pathlib import Path
import numpy as np

CACHE_DIR = Path.home() / ".cache" / "sgrep"

def get_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of file contents for cache invalidation."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def load_cache(file_hash: str) -> tuple[list[tuple[int, str]], np.ndarray] | None:
    """Load cached line metadata and embedding matrix if available."""
    cache_path = CACHE_DIR / f"{file_hash}.npz"
    meta_path = CACHE_DIR / f"{file_hash}.json"

    if cache_path.exists() and meta_path.exists():
        data = np.load(cache_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            lines = [tuple(item) for item in json.load(f)]
        return lines, data["embeddings"]
    return None

def save_cache(file_hash: str, lines: list[tuple[int, str]], embeddings: np.ndarray) -> None:
    """Persist line metadata and normalized embeddings to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{file_hash}.npz"
    meta_path = CACHE_DIR / f"{file_hash}.json"

    np.savez_compressed(cache_path, embeddings=embeddings)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(lines, f)