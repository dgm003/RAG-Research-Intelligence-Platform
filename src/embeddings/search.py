
import argparse
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticSearcher:
    def __init__(self, index_name="papers.faiss", metadata_name="embedding_metadata.jsonl"):
        index_file = DATA_DIR / index_name
        metadata_file = DATA_DIR / metadata_name

        if not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError("FAISS index or metadata file not found.")

        self.index = faiss.read_index(str(index_file))
        self.model = SentenceTransformer(MODEL_NAME)

        with metadata_file.open("r", encoding="utf-8") as file:
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
    return chunk.get("page_number", chunk.get("page", "?"))


def main():
    parser = argparse.ArgumentParser(description="Semantic search over paper chunks.")
    parser.add_argument("query", help="Search query")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    parser.add_argument("--index", default="papers.faiss")
    parser.add_argument("--metadata", default="embedding_metadata.jsonl")
    args = parser.parse_args()

    searcher = SemanticSearcher(args.index, args.metadata)
    results = searcher.search(args.query, args.top_k)

    for rank, chunk in enumerate(results, start=1):
        snippet = " ".join(chunk["text"].split())[:300]
        print(f"\n[{rank}] score={chunk['score']:.3f}")
        print(f"    source: {chunk.get('source_file', '?')} (page {_page(chunk)})")
        print(f"    category: {chunk.get('category', '?')}")
        print(f"    {snippet}...")


if __name__ == "__main__":
    main()
