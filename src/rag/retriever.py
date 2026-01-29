# src/rag/retriever.py
from pathlib import Path
from src.rag.vector_store import VectorStore


KB_PATH = Path("data") / "kb.json"


class Retriever:
    def __init__(self):
        self.store = VectorStore(KB_PATH)

    def get_context(self, question: str, top_k: int = 4) -> str:
        chunks = self.store.search(question, top_k=top_k)

        if not chunks:
            return ""

        return "\n\n".join(chunks)
