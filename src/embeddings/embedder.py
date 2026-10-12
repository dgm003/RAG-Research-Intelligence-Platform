
import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="paper_chunks.jsonl",
        help="Input chunks JSONL filename inside data/processed",
    )
    parser.add_argument(
        "--output-prefix",
        default="paper",
        help="Prefix for output embeddings and metadata",
    )
    args = parser.parse_args()

    chunks_file = OUTPUT_DIR / args.input
    embeddings_file = OUTPUT_DIR / f"{args.output_prefix}_embeddings.npy"
    metadata_file = OUTPUT_DIR / f"{args.output_prefix}_embedding_metadata.jsonl"

    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

    with chunks_file.open("r", encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file if line.strip()]

    if not chunks:
        raise ValueError("No chunks found.")

    texts = [chunk["text"] for chunk in chunks]

    print(f"Input: {chunks_file.name}")
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

    np.save(embeddings_file, embeddings)

    with metadata_file.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("\nEmbedding generation complete.")
    print(f"Chunks embedded: {len(embeddings)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"Embeddings saved: {embeddings_file.name}")
    print(f"Metadata saved: {metadata_file.name}")


if __name__ == "__main__":
    main()
