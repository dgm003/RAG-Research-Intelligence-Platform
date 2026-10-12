import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
from pathlib import Path
from tqdm import tqdm

from src.document_processing.text_cleaner import clean_text


INPUT_FILE = ROOT / "data/processed/paper_chunks.jsonl"
OUTPUT_FILE = ROOT / "data/processed/paper_chunks_cleaned.jsonl"


def main():
    count = 0

    with INPUT_FILE.open("r", encoding="utf-8") as source, \
         OUTPUT_FILE.open("w", encoding="utf-8") as output:

        for line in tqdm(source, desc="Cleaning chunks"):
            if not line.strip():
                continue

            chunk = json.loads(line)
            chunk["text"] = clean_text(chunk["text"])

            if chunk["text"]:
                output.write(
                    json.dumps(chunk, ensure_ascii=False) + "\n"
                )
                count += 1

    print(f"Cleaned chunks: {count}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
