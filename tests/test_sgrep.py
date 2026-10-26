from pathlib import Path
import numpy as np
import pytest
from sgrep.core import SemanticSearcher

@pytest.fixture(scope="module")
def searcher():
    return SemanticSearcher()

def test_semantic_ranking_on_logs(searcher, tmp_path: Path):
    log_file = tmp_path / "server.log"
    log_file.write_text(
        "2026-10-26 10:00:01 INFO User admin logged in successfully from 10.0.0.1\n"
        "2026-10-26 10:01:15 ERROR Upstream socket connection reset by peer after 30000ms\n"
        "2026-10-26 10:02:00 DEBUG Cron job cleanup_temp_files completed in 12ms\n"
        "2026-10-26 10:03:42 FATAL Out of memory: Killed process 4412 (python3)\n",
        encoding="utf-8",
    )

    lines, embeddings = searcher.index_file(log_file, use_cache=False)
    results = searcher.search("network timeout failure", lines, embeddings, top_k=2, threshold=0.1)

    assert len(results) >= 1
    assert results[0]["line_number"] == 2
    assert "connection reset by peer" in results[0]["text"]

def test_cache_roundtrip(searcher, tmp_path: Path, monkeypatch):
    monkeypatch.setattr("sgrep.cache.CACHE_DIR", tmp_path / "cache")

    sample = tmp_path / "notes.txt"
    sample.write_text("Configure SSH forwarding in ~/.ssh/config\n", encoding="utf-8")

    lines_1, emb_1 = searcher.index_file(sample, use_cache=True)
    lines_2, emb_2 = searcher.index_file(sample, use_cache=True)

    assert lines_1 == lines_2
    assert np.allclose(emb_1, emb_2)