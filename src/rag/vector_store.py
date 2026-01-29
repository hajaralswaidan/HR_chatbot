# src/rag/vector_store.py
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    def __init__(self, kb_path: Path):
        if not kb_path.exists():
            raise FileNotFoundError(f"KB not found at {kb_path.resolve()}")

        self.kb_path = kb_path
        self.docs = []
        self.texts = []

        self._load_kb()
        self._build_vectors()

    def _load_kb(self):
        with open(self.kb_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.docs = payload["docs"]
        self.texts = [d["text"] for d in self.docs]

    def _build_vectors(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query: str, top_k: int = 4) -> list[str]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]

        ranked = sorted(
            zip(scores, self.texts),
            key=lambda x: x[0],
            reverse=True
        )

        return [text for score, text in ranked[:top_k] if score > 0]
