# Project Updates — RAG Research Intelligence Platform

This file tracks day-to-day development progress, decisions, issues, and pending tasks. All contributors should update it when completing meaningful work.

## Project Overview

Build a Research Intelligence Platform that processes research papers, enables semantic search, and eventually answers questions using Retrieval-Augmented Generation (RAG), with citations grounded in the source papers.

## Current Status — 12 October 2026

### Completed

**1. PDF processing and chunking**
- Extracted text from 168 research PDFs.
- Generated 15,412 text chunks.
- Preserved source filenames, page numbers, categories, and source paths.

**2. Text cleaning**
- Integrated `ftfy` for general Unicode and encoding repair.
- Added Unicode normalization and cleanup of control characters and line-break hyphenation.
- Created `paper_chunks_cleaned.jsonl`.
- Validation: 15,412 chunks, zero empty chunks, zero missing required metadata fields, and zero occurrences of the checked mojibake markers.

**3. Embedding generation**
- Model: `sentence-transformers/all-MiniLM-L6-v2`.
- Embedding dimensions: 384.
- Generated separate cleaned embeddings while preserving the original baseline.

**4. FAISS vector indexing**
- Built the original and cleaned FAISS indexes.
- Both indexes contain 15,412 vectors.
- Uses normalized embeddings with inner-product similarity.

**5. Semantic search**
- Implemented configurable index and metadata selection.
- Tested machine-learning research discovery, cloud/distributed architectures, and NLP/text classification queries.
- Initial results were relevant across all three queries. Cleaning caused only small score and ranking changes; the cloud-computing query returned identical top-five results.

**6. Exploratory clustering**
- Applied K-Means with six clusters to chunk embeddings.
- Generated cluster assignments, category summaries, and a clustering report.
- Silhouette score: approximately 0.0416, indicating substantial overlap between clusters.

### Current Artifacts

Original baseline:
- `paper_embeddings.npy`
- `embedding_metadata.jsonl`
- `papers.faiss`

Cleaned pipeline:
- `paper_chunks_cleaned.jsonl`
- `paper_cleaned_embeddings.npy`
- `paper_cleaned_embedding_metadata.jsonl`
- `papers_cleaned.faiss`

Generated artifacts and research PDFs are local and should not be committed to Git.

### Known Issues and Decisions

- Some PDF extraction artifacts remain, such as occasional misplaced accents. Avoid manually patching individual words.
- Citation and author metadata may appear in early chunks; header/footer and reference filtering remain future improvements.
- Clustering results are exploratory and are not yet validated as a document-classification model.
- TensorFlow/protobuf compatibility warnings appear during model loading, but the embedding and search commands currently complete successfully.
- Preliminary retrieval comparisons are not a formal benchmark. More representative queries and relevance judgments are needed.

## Next Steps

### Priority 1 — RAG
- [ ] Retrieve relevant chunks from the cleaned FAISS index.
- [ ] Assemble context while preserving source and page metadata.
- [ ] Integrate an LLM to generate answers grounded in retrieved passages.
- [ ] Display citations to the source paper and page.
- [ ] Handle questions where retrieved evidence is insufficient.

### Priority 2 — Evaluation
- [ ] Create a fixed test set of research questions.
- [ ] Compare original and cleaned retrieval quality.
- [ ] Evaluate answer correctness and citation support.
- [ ] Record latency and resource usage.

### Priority 3 — Data and ML Improvements
- [ ] Improve chunk boundaries and metadata extraction.
- [ ] Investigate repeated headers, footers, and reference sections.
- [ ] Compare chunk-level and paper-level clustering.
- [ ] Evaluate automatic document classification.

### Priority 4 — Application and Deployment
- [ ] Connect the Streamlit interface to semantic search and RAG.
- [ ] Complete and test FastAPI endpoints.
- [ ] Improve logging, error handling, and configuration.
- [ ] Validate Docker deployment.
- [ ] Explore cloud deployment.

## Daily Update Template

Copy this section for each meaningful development session.

### YYYY-MM-DD — Contributor Name

**Completed**
- 

**Files or components changed**
- 

**Validation and results**
- 

**Issues or decisions**
- 

**Next steps**
- 

---

## Contribution Guidelines

- Keep `README.md` focused on project overview and setup.
- Use this file for ongoing development updates and shared decisions.
- Record commands, validation results, and relevant output summaries.
- Coordinate before changing shared pipeline interfaces or replacing baseline artifacts.
- Use feature branches and pull requests for substantial changes.
- Never commit PDFs, generated embeddings, FAISS indexes, secrets, or environment files.
