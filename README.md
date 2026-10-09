# Research Intelligence Platform

An end-to-end research paper intelligence platform combining Machine Learning, semantic search, vector databases, and Retrieval-Augmented Generation (RAG).

## Project Overview

The platform processes research papers, converts their content into vector embeddings, and indexes those embeddings for efficient semantic retrieval. The goal is to enable research discovery, paper similarity analysis, and question answering grounded in academic sources.

## Current Progress

- [x] OpenAlex research paper collection
- [x] PDF text extraction using PyMuPDF
- [x] Page-aware text chunking
- [x] Embedding generation using Sentence Transformers
- [x] FAISS vector index creation

### Current Dataset

| Metric | Value |
|---|---:|
| Research papers processed | 168 |
| Text chunks generated | 15,412 |
| Embedding dimensions | 384 |
| Embedding model | all-MiniLM-L6-v2 |
| Vector index | FAISS IndexFlatIP |

## Technology Stack

- **Language:** Python
- **Data source:** OpenAlex API
- **PDF processing:** PyMuPDF
- **Embeddings:** Sentence Transformers
- **Vector search:** FAISS
- **Data formats:** JSONL, NumPy, CSV

## Architecture

1. **Paper ingestion:** Download open-access research papers and metadata.
2. **PDF extraction:** Extract text page-by-page while preserving source information.
3. **Chunking:** Divide extracted text into smaller overlapping sections.
4. **Embedding generation:** Convert each chunk into a 384-dimensional vector.
5. **Vector indexing:** Store vectors in a FAISS index for similarity-based retrieval.

### Pipeline

```text
Research Papers (PDF)
        ↓
PDF Text Extraction
        ↓
Text Chunking
        ↓
Sentence Transformer Embeddings
        ↓
FAISS Vector Index
        ↓
Semantic Search and RAG (Planned)
```

## Project Structure

```text
RAG-Research-Intelligence-Platform/
├── api/
├── app/
├── data/
│   ├── raw/
│   ├── metadata/
│   └── processed/
├── scripts/
│   ├── download_papers.py
│   └── process_paper.py
├── src/
│   ├── document_processing/
│   │   ├── pdf_loader.py
│   │   └── chunker.py
│   ├── embeddings/
│   │   ├── embedder.py
│   │   └── vector_store.py
│   ├── ml/
│   ├── rag/
│   └── similarity/
├── tests/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── PROJECT_PLAN.md
```

## Running the Pipeline

Run commands from the project root.

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install pymupdf sentence-transformers faiss-cpu
```

### 2. Extract and chunk papers

```bash
python scripts/process_paper.py
```

This generates `data/processed/paper_chunks.jsonl`.

### 3. Generate embeddings

```bash
python -m src.embeddings.embedder
```

This generates the embedding vectors and their associated metadata.

### 4. Build the FAISS index

```bash
python -m src.embeddings.vector_store
```

This creates the FAISS index for similarity search.

**Note:** The raw PDFs, generated chunks, embeddings, and FAISS index are stored locally and are not included in the Git repository.

## Roadmap

- [ ] Implement semantic search
- [ ] Evaluate and visualize ML paper clustering
- [ ] Build research paper similarity analysis
- [ ] Implement RAG-based question answering
- [ ] Add source and page citations
- [ ] Develop the Streamlit interface
- [ ] Build and deploy the FastAPI backend
- [ ] Containerize and deploy to a cloud platform

## Project Goal

Build a practical research intelligence system demonstrating document processing, machine learning, vector retrieval, LLM integration, and cloud deployment.
