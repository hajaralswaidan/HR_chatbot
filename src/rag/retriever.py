from pathlib import Path

from src.rag.vector_store import VectorStore
from src.config import EXTERNAL_STORE_DIR

BASE_DIR = Path(__file__).resolve().parents[2]
KB_PATH = BASE_DIR / "data" / "kb.json"


class Retriever:
    def __init__(self):
        EXTERNAL_STORE_DIR.mkdir(parents=True, exist_ok=True)
        self.store = VectorStore(kb_path=KB_PATH, persist_dir=EXTERNAL_STORE_DIR)

    def get_context(self, question: str, top_k: int = 4) -> str:
        chunks = self.store.search(question, top_k=top_k)
        if not chunks:
            return ""

        # ✅ Filter out row-level dumps
        clean = []
        for c in chunks:
            t = (c or "").strip()
            if not t:
                continue
            if t.startswith("Employee records"):
                continue
            if "EmployeeNumber=" in t and "Attrition=" in t and "YearsAtCompany=" in t:
                continue
            clean.append(t)

        return "\n\n".join(clean) if clean else ""
