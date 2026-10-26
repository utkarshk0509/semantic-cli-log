import argparse
import sys
from pathlib import Path
from sgrep.core import SemanticSearcher

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def collect_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return [
        p for p in target.rglob("*")
        if p.is_file() and not any(part.startswith(".git") or part == ".venv" for part in p.parts)
    ]

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sgrep",
        description="Semantic grep for logs, configs, and local files.",
    )
    parser.add_argument("query", type=str, help="Natural language search query")
    parser.add_argument("path", type=Path, help="File or directory to search")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Max results to return (default: 5)")
    parser.add_argument("-t", "--threshold", type=float, default=0.25, help="Minimum similarity score (0.0 to 1.0)")
    parser.add_argument("--no-cache", action="store_true", help="Bypass local embedding cache")

    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: Path '{args.path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    searcher = SemanticSearcher()
    files = collect_files(args.path)
    all_matches = []

    for filepath in files:
        lines, embeddings = searcher.index_file(filepath, use_cache=not args.no_cache)
        matches = searcher.search(
            query=args.query,
            lines=lines,
            embeddings=embeddings,
            top_k=args.top_k,
            threshold=args.threshold,
        )
        for m in matches:
            m["file"] = str(filepath)
            all_matches.append(m)

    all_matches.sort(key=lambda x: x["score"], reverse=True)

    for match in all_matches[: args.top_k]:
        print(
            f"{CYAN}{match['file']}{RESET}:{YELLOW}{match['line_number']}{RESET} "
            f"[{GREEN}{match['score']:.2f}{RESET}] {match['text']}"
        )

if __name__ == "__main__":
    main()