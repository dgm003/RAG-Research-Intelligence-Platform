import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.retriever import ResearchRetriever, assemble_context

retriever = ResearchRetriever()

question = "What are the main challenges in distributed cloud computing?"
results = retriever.retrieve(question, top_k=3)

print(f"Question: {question}")
print(f"Retrieved passages: {len(results)}")

for result in results:
    print(
        f"\nSource: {result['source_file']}"
        f"\nPage: {result['page_number']}"
        f"\nScore: {result['score']:.3f}"
        f"\nText: {' '.join(result['text'].split())[:300]}"
    )

print("\n--- ASSEMBLED CONTEXT PREVIEW ---")
print(assemble_context(results)[:1500])
