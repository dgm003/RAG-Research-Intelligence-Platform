import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.document_processing.pdf_loader import load_corpus
from src.embeddings.embedder import TextEmbedder


def build_paper_table(corpus):
    rows = []
    for paper in corpus:
        # For the first baseline, represent a paper using its full extracted text.
        # We will later test title+abstract, section-aware text, and chunk-level representations.
        text = " ".join(paper["text"].split())
        rows.append(
            {
                "paper_id": paper["paper_id"],
                "filename": paper["filename"],
                "num_pages": paper["num_pages"],
                "text_chars": len(text),
                "text": text,
            }
        )
    return pd.DataFrame(rows)


def run(input_dir: str, output_dir: str, k: int, model_name: str):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(input_dir)
    if not corpus:
        raise SystemExit(f"No PDF files found in {input_dir}")

    papers = build_paper_table(corpus)
    papers = papers[papers["text_chars"] > 100].reset_index(drop=True)
    if len(papers) < k:
        raise SystemExit(f"Need at least {k} usable papers, found {len(papers)}")

    embedder = TextEmbedder(model_name)
    embeddings = embedder.encode(papers["text"].tolist())

    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = model.fit_predict(embeddings)
    silhouette = silhouette_score(embeddings, labels) if k > 1 else float("nan")

    # PCA is only for visualization/export; clustering uses the original embeddings.
    coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)

    papers["cluster"] = labels
    papers["pca_x"] = coords[:, 0]
    papers["pca_y"] = coords[:, 1]

    papers[["paper_id", "filename", "num_pages", "text_chars", "cluster", "pca_x", "pca_y"]].to_csv(
        output / "clusters.csv", index=False
    )
    papers.drop(columns=["text"]).to_csv(output / "papers.csv", index=False)
    np.save(output / "paper_embeddings.npy", embeddings)

    print(f"Papers: {len(papers)}")
    print(f"Embedding model: {model_name}")
    print(f"K: {k}")
    print(f"Silhouette score: {silhouette:.4f}")
    print("Cluster counts:")
    print(papers["cluster"].value_counts().sort_index().to_string())
    print(f"\nSaved results to {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, args.k, args.model)
