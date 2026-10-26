from huggingface_hub import hf_hub_download
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

INDEX_FILE = DATA_DIR / "papers_cleaned.faiss"
METADATA_FILE = DATA_DIR / "paper_cleaned_embedding_metadata.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HF_REPO_ID = "dgm003/rag-research-artifacts"


def ensure_artifacts():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not INDEX_FILE.exists():
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="papers_cleaned.faiss",
            repo_type="model",
            local_dir=str(DATA_DIR),
        )

    if not METADATA_FILE.exists():
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="paper_cleaned_embedding_metadata.jsonl",
            repo_type="model",
            local_dir=str(DATA_DIR),
        )



class ResearchRetriever:
    def __init__(self):
        ensure_artifacts()

        self.index = faiss.read_index(str(INDEX_FILE))
        with METADATA_FILE.open("r", encoding="utf-8") as file:
            self.metadata = [
                json.loads(line) for line in file if line.strip()
            ]

        if self.index.ntotal != len(self.metadata):
            raise ValueError("Index and metadata counts do not match.")

        self.model = SentenceTransformer(MODEL_NAME)

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20.")

        query_embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(
            query_embedding, min(top_k, self.index.ntotal)
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            chunk = self.metadata[int(idx)]
            results.append({
                "text": chunk["text"],
                "source_file": chunk.get("source_file", "Unknown"),
                "page_number": chunk.get("page_number", "Unknown"),
                "category": chunk.get("category", "Unknown"),
                "score": float(score),
            })

        return results


def assemble_context(results: list[dict]) -> str:
    """Format retrieved passages with source information for RAG."""
    passages = []

    for i, result in enumerate(results, start=1):
        passages.append(
            f"[Source {i}: {result['source_file']}, "
            f"page {result['page_number']}]\n"
            f"{result['text']}"
        )

    return "\n\n".join(passages)
