
import argparse
import json
from pathlib import Path

import faiss
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def build_index(embeddings_name: str, metadata_name: str, index_name: str):
    embeddings_file = DATA_DIR / embeddings_name
    metadata_file = DATA_DIR / metadata_name
    index_file = DATA_DIR / index_name

    if not embeddings_file.exists() or not metadata_file.exists():
        raise FileNotFoundError("Embeddings or metadata file not found.")

    embeddings = np.load(embeddings_file).astype("float32")

    with metadata_file.open("r", encoding="utf-8") as file:
        metadata = [json.loads(line) for line in file if line.strip()]

    if len(embeddings) != len(metadata):
        raise ValueError("Embedding and metadata counts do not match.")

    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError("Embeddings must be a finite 2D array.")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(index_file))

    print(f"Indexed vectors: {index.ntotal}")
    print(f"Vector dimensions: {index.d}")
    print(f"FAISS index saved: {index_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="paper_embeddings.npy")
    parser.add_argument("--metadata", default="embedding_metadata.jsonl")
    parser.add_argument("--index", default="papers.faiss")
    args = parser.parse_args()

    build_index(args.embeddings, args.metadata, args.index)
