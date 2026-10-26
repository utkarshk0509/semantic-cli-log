from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sgrep.cache import get_file_hash, load_cache, save_cache

DEFAULT_MODEL = "all-MiniLM-L6-v2"

class SemanticSearcher:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model so CLI help/errors don't pay import latency."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def parse_file(self, filepath: Path, min_length: int = 8) -> list[tuple[int, str]]:
        """Extract non-empty, meaningful lines with their 1-based line numbers."""
        results = []
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f, start=1):
                cleaned = line.strip()
                if len(cleaned) >= min_length:
                    results.append((idx, cleaned))
        return results

    def index_file(self, filepath: Path, use_cache: bool = True) -> tuple[list[tuple[int, str]], np.ndarray]:
        """Return (lines, embeddings) from cache or compute from scratch."""
        file_hash = get_file_hash(filepath)

        if use_cache:
            cached = load_cache(file_hash)
            if cached is not None:
                return cached

        lines = self.parse_file(filepath)
        if not lines:
            return [], np.empty((0, 384), dtype=np.float32)

        texts = [text for _, text in lines]
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        if use_cache:
            save_cache(file_hash, lines, embeddings)

        return lines, embeddings

    def search(
        self,
        query: str,
        lines: list[tuple[int, str]],
        embeddings: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.25,
    ) -> list[dict]:
        """Rank lines against the query using fast matrix dot product."""
        if len(lines) == 0 or embeddings.size == 0:
            return []

        query_vec = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        scores = np.dot(embeddings, query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= threshold:
                line_num, text = lines[idx]
                results.append({
                    "line_number": line_num,
                    "score": round(score, 4),
                    "text": text,
                })
        return results