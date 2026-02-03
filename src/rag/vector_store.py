import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import save_npz, load_npz


class VectorStore:
    """
    TF-IDF + Cosine Vector Store with disk persistence (external folder).
    """

    def __init__(self, kb_path: Path, persist_dir: Path):
        kb_path = Path(kb_path)
        persist_dir = Path(persist_dir)

        if not kb_path.exists():
            raise FileNotFoundError(f"KB not found at {kb_path.resolve()}")

        self.kb_path = kb_path
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.persist_dir / "kb_meta.json"
        self.vectorizer_path = self.persist_dir / "tfidf_vectorizer.pkl"
        self.matrix_path = self.persist_dir / "tfidf_matrix.npz"

        self.docs = []
        self.texts = []
        self.vectorizer = None
        self.matrix = None

        if self._persisted_ready() and self._kb_unchanged():
            self._load_from_disk()
        else:
            self._load_kb()
            self._build_vectors()
            self._save_to_disk()

    def _persisted_ready(self) -> bool:
        return self.meta_path.exists() and self.vectorizer_path.exists() and self.matrix_path.exists()

    def _kb_signature(self) -> dict:
        stat = self.kb_path.stat()
        return {
            "kb_path": str(self.kb_path.resolve()),
            "kb_size": stat.st_size,
            "kb_mtime": int(stat.st_mtime),
        }

    def _kb_unchanged(self) -> bool:
        if not self.meta_path.exists():
            return False
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            return meta.get("signature") == self._kb_signature()
        except Exception:
            return False

    def _load_kb(self):
        with open(self.kb_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.docs = payload.get("docs", [])
        self.texts = [d.get("text", "") for d in self.docs if (d.get("text", "") or "").strip()]

    def _build_vectors(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def _save_to_disk(self):
        meta = {
            "signature": self._kb_signature(),
            "count": len(self.texts),
            "texts": self.texts,
        }
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        with open(self.vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        save_npz(self.matrix_path, self.matrix)

    def _load_from_disk(self):
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.texts = meta.get("texts", [])

        with open(self.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

        self.matrix = load_npz(self.matrix_path)

    def search(self, query: str, top_k: int = 4) -> list[str]:
        if not query or not query.strip():
            return []

        if self.vectorizer is None or self.matrix is None or not self.texts:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]

        ranked = sorted(
            zip(scores, self.texts),
            key=lambda x: x[0],
            reverse=True
        )

        return [text for score, text in ranked[:top_k] if score > 0]
