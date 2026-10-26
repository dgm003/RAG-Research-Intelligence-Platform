# Research Intelligence Platform

An end-to-end research paper intelligence platform combining machine learning, semantic search, FAISS vector indexing, and Retrieval-Augmented Generation (RAG). The platform enables users to search academic literature and ask questions grounded in retrieved research passages.

## Project Overview

The platform collects open-access research papers, extracts and cleans PDF text, splits documents into overlapping chunks, generates vector embeddings, and indexes them for semantic retrieval. A RAG pipeline retrieves relevant passages and uses a Groq-hosted LLM to generate answers with source references.

A Streamlit interface provides an interactive research assistant, backed by a FastAPI service.

## Current Features

- [x] Research paper collection using OpenAlex
- [x] PDF text extraction using PyMuPDF
- [x] Page-aware text chunking
- [x] Text cleaning and normalization
- [x] Embedding generation using Sentence Transformers
- [x] FAISS vector index creation
- [x] Semantic search with source metadata and similarity scores
- [x] RAG retrieval and context assembly
- [x] LLM-powered answer generation using Groq
- [x] FastAPI endpoints for health checks, retrieval, and question answering
- [x] Streamlit research chat interface
- [x] Basic validation of generated source citation labels
- [x] Citation-validation tests

## Dataset

| Metric | Value |
|---|---:|
| Research PDFs discovered | 168 |
| Text chunks generated | 15,412 |
| Embedding dimensions | 384 |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector index | FAISS inner-product index |
| LLM provider | Groq |

The counts describe the current local dataset and may change when papers are added or reprocessed.

## Technology Stack

- **Language:** Python
- **Paper discovery:** OpenAlex API
- **PDF extraction:** PyMuPDF
- **Text processing:** ftfy, Unicode normalization
- **Embeddings:** Sentence Transformers
- **Vector search:** FAISS
- **Data processing:** NumPy, pandas, JSONL
- **Backend:** FastAPI, Uvicorn, Pydantic
- **LLM integration:** Groq API
- **Frontend:** Streamlit
- **Testing:** Python test scripts

## Architecture

```text
OpenAlex / Research PDFs
          |
          v
PDF Text Extraction
          |
          v
Text Cleaning and Chunking
          |
          v
Sentence Transformer Embeddings
          |
          v
FAISS Vector Index + Metadata
          |
          v
User Question
          |
          v
Semantic Retrieval (Top-K)
          |
          v
Retrieved Context with Source Labels
          |
          v
Groq LLM Answer Generation
          |
          v
Answer + Source Metadata
          |
          v
Streamlit UI
```

## Project Structure

```text
RAG-Research-Intelligence-Platform/
├── api/
│   └── main.py
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   ├── download_papers.py
│   ├── process_paper.py
│   ├── clean_papers.py
│   ├── test_retriever.py
│   └── test_generator.py
├── src/
│   ├── document_processing/
│   ├── embeddings/
│   ├── ml/
│   ├── rag/
│   │   ├── retriever.py
│   │   └── generator.py
│   └── similarity/
├── tests/
│   └── test_citations.py
├── requirements.txt
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── PROJECT_PLAN.md
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/dgm003/RAG-Research-Intelligence-Platform.git
cd RAG-Research-Intelligence-Platform
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

Ensure the requirements include the packages used by the project, including `pymupdf`, `faiss-cpu`, `sentence-transformers`, `ftfy`, `fastapi`, `uvicorn`, `streamlit`, `groq`, and `python-dotenv`.

### 4. Configure environment variables

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

Keep `.env` out of version control. Never commit API keys or secrets.

### 5. Prepare the local research index

The RAG retriever expects these generated files:

```text
data/processed/papers_cleaned.faiss
data/processed/paper_cleaned_embedding_metadata.jsonl
```

If these files are not present, run the document cleaning, embedding, and index-building pipeline using the scripts and command-line options available in the repository. The raw PDFs and generated artifacts are not committed to Git.

## Running the Application

Start the FastAPI backend in the first terminal:

```powershell
python -m uvicorn api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

Open the interactive API documentation at `http://127.0.0.1:8000/docs`.

Start Streamlit in a second terminal:

```powershell
python -m streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

The frontend currently calls the local FastAPI backend. For a public deployment, configure the frontend to use the deployed API URL.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check API health |
| POST | `/retrieve` | Retrieve relevant research passages |
| POST | `/ask` | Generate an answer using retrieved context |

Example request to `/ask`:

```json
{
  "question": "What are the main challenges in distributed cloud computing?",
  "top_k": 3
}
```

The response includes the question, generated answer, and source metadata such as filenames, page numbers, categories, and retrieval scores.

## Testing

Run the citation-validation tests:

```powershell
python tests/test_citations.py
```

Additional manual checks:

```powershell
python scripts/test_retriever.py
python scripts/test_generator.py
```

The retriever and generator checks require the corresponding local index files and a configured Groq API key.

Citation validation checks whether referenced source labels exist in the retrieved context. It does not independently verify that every cited passage fully supports the associated claim.

## Data and Security

- Raw research PDFs, generated embeddings, and FAISS indexes are kept locally and excluded from Git.
- API keys belong in `.env` or the hosting provider's secret configuration.
- Do not expose API keys in source code, logs, or public repositories.
- Public deployment requires appropriate API access controls, resource limits, and configuration.

## Roadmap

- [x] Research paper ingestion and preprocessing
- [x] Semantic search with FAISS
- [x] RAG-based question answering
- [x] FastAPI backend
- [x] Streamlit frontend
- [x] Basic source-label validation
- [ ] Evaluate retrieval quality across a representative query set
- [ ] Improve PDF extraction and chunk-boundary quality
- [ ] Evaluate answer faithfulness and citation correctness
- [ ] Improve paper similarity analysis and clustering evaluation
- [ ] Configure and test public cloud deployment
- [ ] Add deployment monitoring and API usage controls

## Project Goal

Build a practical research intelligence system demonstrating document processing, machine learning, semantic retrieval, LLM integration, API development, and cloud deployment.