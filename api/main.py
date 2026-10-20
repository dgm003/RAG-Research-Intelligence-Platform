
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag.generator import AnswerGenerator
from src.rag.retriever import assemble_context

from src.rag.retriever import ResearchRetriever

app = FastAPI(
    title="Research Intelligence API",
    version="0.2.0",
)

retriever = None


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


generator = None

@app.on_event("startup")
def load_retriever():
    global retriever, generator
    retriever = ResearchRetriever()
    generator = AnswerGenerator()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/retrieve")
def retrieve_documents(request: RetrieveRequest):
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Retriever is not initialized.",
        )

    try:
        results = retriever.retrieve(
            request.question,
            top_k=request.top_k,
        )
        return {
            "question": request.question,
            "count": len(results),
            "results": results,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    
@app.post("/ask")
def ask_question(request: RetrieveRequest):
    if retriever is None or generator is None:
        raise HTTPException(
            status_code=503,
            detail="RAG components are not initialized.",
        )

    try:
        results = retriever.retrieve(
            request.question,
            top_k=request.top_k,
        )

        if not results:
            return {
                "question": request.question,
                "answer": "No relevant research passages were found.",
                "sources": [],
            }

        context = assemble_context(results)
        answer = generator.generate_answer(
            request.question,
            context,
        )

        sources = [
            {
                "source_file": result["source_file"],
                "page_number": result["page_number"],
                "category": result["category"],
                "score": result["score"],
            }
            for result in results
        ]

        return {
            "question": request.question,
            "answer": answer,
            "sources": sources,
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))