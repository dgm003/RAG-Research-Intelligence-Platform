from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader


def extract_pdf(path: str | Path) -> Dict:
    """Extract text and basic metadata from one PDF."""
    path = Path(path)
    reader = PdfReader(str(path))

    pages: List[Dict] = []
    text_parts: List[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        pages.append({"page": page_number, "text": text})
        if text:
            text_parts.append(text)

    return {
        "paper_id": path.stem,
        "filename": path.name,
        "num_pages": len(reader.pages),
        "text": "\n\n".join(text_parts),
        "pages": pages,
    }


def load_corpus(input_dir: str | Path) -> List[Dict]:
    input_dir = Path(input_dir)
    pdfs = sorted(input_dir.glob("*.pdf"))
    return [extract_pdf(pdf) for pdf in pdfs]
