
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.generator import validate_citations


def test_valid_citations():
    answer = "Consistency matters [Source 1]. Security matters [Source 2]."
    assert validate_citations(answer, 2) == []


def test_invalid_citation():
    answer = "This claim cites [Source 3]."
    assert validate_citations(answer, 2) == [3]


def test_no_citations():
    assert validate_citations("No citations here.", 3) == []


if __name__ == "__main__":
    test_valid_citations()
    test_invalid_citation()
    test_no_citations()
    print("All citation validation tests passed.")
