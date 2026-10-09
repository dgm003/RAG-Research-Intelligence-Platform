
import json
from pathlib import Path

from tqdm import tqdm

from src.document_processing.pdf_loader import extract_pdf
from src.document_processing.chunker import chunk_pages


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "paper_chunks.jsonl"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(RAW_DIR.rglob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {RAW_DIR}")
        return

    total_chunks = 0
    failed_pdfs = []

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                pages = extract_pdf(pdf_path)
                chunks = chunk_pages(pages)

                for chunk in chunks:
                    chunk["category"] = pdf_path.parent.name
                    chunk["source_path"] = str(
                        pdf_path.relative_to(PROJECT_ROOT)
                    )
                    output.write(
                        json.dumps(chunk, ensure_ascii=False) + "\n"
                    )

                total_chunks += len(chunks)

            except Exception as exc:
                failed_pdfs.append((pdf_path.name, str(exc)))

    print("\nProcessing complete.")
    print(f"PDFs discovered: {len(pdf_files)}")
    print(f"Chunks saved: {total_chunks}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Failed PDFs: {len(failed_pdfs)}")

    for filename, error in failed_pdfs:
        print(f"- {filename}: {error}")


if __name__ == "__main__":
    main()
