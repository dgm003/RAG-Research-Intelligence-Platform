import argparse
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

INDEX_FILE = DATA_DIR / "papers.faiss"
METADATA_FILE = DATA_DIR / "embedding_metadata.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticSearcher:
    def __init__(self):
        if not INDEX_FILE.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {INDEX_FILE}. "
                "Run `python -m src.embeddings.vector_store` first."
            )

        self.index = faiss.read_index(str(INDEX_FILE))
        self.model = SentenceTransformer(MODEL_NAME)

        with METADATA_FILE.open("r", encoding="utf-8") as file:
            self.metadata = [json.loads(line) for line in file if line.strip()]

        if self.index.ntotal != len(self.metadata):
            raise ValueError("Index and metadata counts do not match.")

    def search(self, query, top_k=5):
        query_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, ids = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            chunk = dict(self.metadata[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results


def _page(chunk):
    for key in ("page", "page_number", "page_num"):
        if key in chunk:
            return chunk[key]
    return "?"


def main():
    parser = argparse.ArgumentParser(description="Semantic search over paper chunks.")
    parser.add_argument("query", help="Search query")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    args = parser.parse_args()

    searcher = SemanticSearcher()
    results = searcher.search(args.query, args.top_k)

    for rank, chunk in enumerate(results, start=1):
        snippet = " ".join(chunk["text"].split())[:300]
        print(f"\n[{rank}] score={chunk['score']:.3f}")
        print(f"    source: {chunk.get('source_path', '?')} (page {_page(chunk)})")
        print(f"    category: {chunk.get('category', '?')}")
        print(f"    {snippet}...")


if __name__ == "__main__":
    main()