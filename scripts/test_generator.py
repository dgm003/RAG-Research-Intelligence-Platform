
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.generator import AnswerGenerator

generator = AnswerGenerator()

context = """
[Source 1: distributed_cloud_review.pdf, page 8]
Distributed systems involve multiple autonomous computers working together.
Cloud computing provides shared computing resources over the internet.
Both can be scalable, but distributed systems may require more complex
resource management.

[Source 2: cloud_architecture.pdf, page 9]
Consistency, fault tolerance, security, and resource efficiency remain
important challenges in distributed cloud computing.
"""

question = "What are the main challenges in distributed cloud computing?"

answer = generator.generate_answer(question, context)

print("\nQUESTION:")
print(question)
print("\nGENERATED ANSWER:")
print(answer)
