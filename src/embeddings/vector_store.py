
import json
from pathlib import Path

import faiss
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

EMBEDDINGS_FILE = DATA_DIR / "paper_embeddings.npy"
METADATA_FILE = DATA_DIR / "embedding_metadata.jsonl"
INDEX_FILE = DATA_DIR / "papers.faiss"


def build_index():
    embeddings = np.load(EMBEDDINGS_FILE).astype("float32")

    with METADATA_FILE.open("r", encoding="utf-8") as file:
        metadata = [json.loads(line) for line in file if line.strip()]

    if len(embeddings) != len(metadata):
        raise ValueError("Embedding and metadata counts do not match.")

    # Embeddings are normalized, so inner product equals cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_FILE))

    print(f"Indexed vectors: {index.ntotal}")
    print(f"Vector dimensions: {index.d}")
    print(f"FAISS index saved: {INDEX_FILE}")


if __name__ == "__main__":
    build_index()
