
from typing import Any


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[dict[str, Any]]:
    """
    Split extracted page text into overlapping chunks.

    chunk_size and overlap are measured in characters.
    Page numbers and source filenames are preserved.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"].strip()
        if not text:
            continue

        # Keep paragraphs together where possible.
        paragraphs = [
            p.strip() for p in text.split("\n\n") if p.strip()
        ]

        current = ""

        for paragraph in paragraphs:
            # Split unusually long paragraphs into smaller pieces.
            while len(paragraph) > chunk_size:
                if current:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "source_file": page["source_file"],
                        "page_number": page["page_number"],
                        "text": current.strip(),
                    })
                    chunk_id += 1
                    current = ""

                chunks.append({
                    "chunk_id": chunk_id,
                    "source_file": page["source_file"],
                    "page_number": page["page_number"],
                    "text": paragraph[:chunk_size].strip(),
                })
                chunk_id += 1
                paragraph = paragraph[chunk_size - overlap:]

            candidate = f"{current}\n\n{paragraph}".strip()

            if current and len(candidate) > chunk_size:
                chunks.append({
                    "chunk_id": chunk_id,
                    "source_file": page["source_file"],
                    "page_number": page["page_number"],
                    "text": current.strip(),
                })
                chunk_id += 1

                # Carry a small overlap from the previous chunk.
                current = current[-overlap:] if overlap else ""
                current = current.strip()

            current = f"{current}\n\n{paragraph}".strip()

        if current:
            chunks.append({
                "chunk_id": chunk_id,
                "source_file": page["source_file"],
                "page_number": page["page_number"],
                "text": current.strip(),
            })
            chunk_id += 1

    return chunks


if __name__ == "__main__":
    from pathlib import Path
    from src.document_processing.pdf_loader import extract_pdf

    pdf_files = list(Path("data/raw").rglob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in data/raw/")
    else:
        pages = extract_pdf(pdf_files[0])
        chunks = chunk_pages(pages)

        print(f"PDF: {pdf_files[0].name}")
        print(f"Pages extracted: {len(pages)}")
        print(f"Chunks created: {len(chunks)}")

        for chunk in chunks[:3]:
            print(
                f"\nChunk {chunk['chunk_id']} "
                f"| Page {chunk['page_number']}"
            )
            print(chunk["text"][:250])
