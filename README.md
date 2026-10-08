# Research Intelligence Platform

An end-to-end research/technical document intelligence platform combining:

- ML-based document clustering/classification
- Semantic embeddings
- Vector search / RAG
- LLM grounded question answering with citations
- Paper similarity and potential-overlap detection
- Streamlit UI
- FastAPI backend
- Docker
- AWS deployment

## Current milestone: Phase 1 — Document embeddings + clustering

The first version intentionally uses **user-provided PDFs**. We will not use filenames as labels. The pipeline extracts paper text, creates semantic embeddings, clusters papers with K-Means, and saves metadata/results for inspection.

## Planned architecture

```text
PDF corpus
   |
   v
Text extraction / cleaning
   |
   v
Embedding model
   |
   +--------------------+
   |                    |
   v                    v
K-Means clustering   Vector index
   |                    |
   v                    v
Topic metadata       RAG retrieval
                        |
                        v
                    Reranking
                        |
                        v
                       LLM
                        |
                        v
                 Answer + citations

Similarity module: paper-to-paper and passage-to-passage similarity

Deployment: Streamlit + FastAPI + Docker + AWS
```

## Local setup

Python 3.11+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Put research PDFs in:

```text
data/raw/
```

Run the Phase 1 pipeline:

```bash
python -m src.ml.cluster_papers --input-dir data/raw --output-dir data/processed --k 6
```

Outputs:

- `data/processed/papers.csv`
- `data/processed/paper_embeddings.npy`
- `data/processed/clusters.csv`

The value of `k=6` is only a starting point. We will later compare multiple K values using clustering metrics and human inspection rather than assuming six topics are correct.
