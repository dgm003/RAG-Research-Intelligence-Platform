
from pathlib import Path

import pymupdf  # PyMuPDF


def extract_pdf(pdf_path: str | Path) -> list[dict]:
    """Extract text page-by-page, preserving page numbers."""
    pdf_path = Path(pdf_path)
    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "source_file": pdf_path.name,
                    "page_number": page_number,
                    "text": text,
                })

    return pages


if __name__ == "__main__":
    pdf_files = list(
        Path("data/raw").rglob("*.pdf")
    )

    if not pdf_files:
        print("No PDFs found in data/raw/")
    else:
        sample_pdf = pdf_files[0]
        pages = extract_pdf(sample_pdf)

        print(f"PDF: {sample_pdf.name}")
        print(f"Pages with text: {len(pages)}")

        if pages:
            print("\nFirst page preview:\n")
            print(pages[0]["text"][:1000])
        else:
            print("No extractable text found. This PDF may require OCR.")
