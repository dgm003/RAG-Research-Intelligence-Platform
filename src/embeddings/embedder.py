
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "paper_chunks.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_FILE = OUTPUT_DIR / "paper_embeddings.npy"
METADATA_FILE = OUTPUT_DIR / "embedding_metadata.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


def main():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file if line.strip()]

    if not chunks:
        raise ValueError("No chunks found.")

    texts = [chunk["text"] for chunk in chunks]

    print(f"Loaded {len(texts)} chunks.")
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(EMBEDDINGS_FILE, embeddings)

    with METADATA_FILE.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("\nEmbedding generation complete.")
    print(f"Chunks embedded: {len(embeddings)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"Embeddings saved: {EMBEDDINGS_FILE}")
    print(f"Metadata saved: {METADATA_FILE}")


if __name__ == "__main__":
    main()
