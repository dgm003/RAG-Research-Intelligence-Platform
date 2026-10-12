
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

EMBEDDINGS_FILE = DATA_DIR / "paper_embeddings.npy"
METADATA_FILE = DATA_DIR / "embedding_metadata.jsonl"
OUTPUT_DIR = DATA_DIR / "clustering"

DEFAULT_K = 6
RANDOM_STATE = 42


def main(k: int = DEFAULT_K):
    if k < 2:
        raise ValueError("K must be at least 2.")

    # Load existing embeddings and metadata.
    embeddings = np.load(EMBEDDINGS_FILE).astype("float32")

    with METADATA_FILE.open("r", encoding="utf-8") as file:
        metadata = [json.loads(line) for line in file if line.strip()]

    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Embedding count ({len(embeddings)}) does not match "
            f"metadata count ({len(metadata)})."
        )

    if len(embeddings) <= k:
        raise ValueError("Number of chunks must be greater than K.")

    print(f"Loaded chunks: {len(embeddings)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"Running K-Means with K={k}...")

    # Cluster using the existing embeddings.
    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10,
    )
    labels = model.fit_predict(embeddings)

    # Evaluate on a reproducible sample to keep computation manageable.
    sample_size = min(2000, len(embeddings))
    score = silhouette_score(
        embeddings,
        labels,
        sample_size=sample_size if sample_size < len(embeddings) else None,
        random_state=RANDOM_STATE,
    )

    # PCA is only for 2D visualization; it does not affect clustering.
    coordinates = PCA(
        n_components=2,
        random_state=RANDOM_STATE,
    ).fit_transform(embeddings)

    results = pd.DataFrame({
        "chunk_id": [item.get("chunk_id") for item in metadata],
        "source_file": [item.get("source_file") for item in metadata],
        "source_path": [item.get("source_path") for item in metadata],
        "page_number": [item.get("page_number") for item in metadata],
        "category": [item.get("category") for item in metadata],
        "cluster": labels,
        "pca_x": coordinates[:, 0],
        "pca_y": coordinates[:, 1],
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(OUTPUT_DIR / "chunk_clusters.csv", index=False)

    summary = (
        results.groupby(["cluster", "category"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    summary.to_csv(OUTPUT_DIR / "cluster_category_summary.csv")

    report = {
        "chunks": len(embeddings),
        "embedding_dimensions": embeddings.shape[1],
        "k": k,
        "silhouette_score": float(score),
        "cluster_counts": {
            str(int(cluster)): int(count)
            for cluster, count in results["cluster"].value_counts().items()
        },
    }

    with (OUTPUT_DIR / "clustering_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print("\nClustering complete.")
    print(f"Chunks clustered: {len(embeddings)}")
    print(f"Silhouette score: {score:.4f}")
    print("\nCluster counts:")
    print(results["cluster"].value_counts().sort_index().to_string())
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
