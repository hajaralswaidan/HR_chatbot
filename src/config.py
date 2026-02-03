# src/config.py
from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parents[1]

# Data & Database
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "hr.sqlite"

# Vector Store (RAG)

EXTERNAL_STORE_DIR = DATA_DIR / "vector_store"
EXTERNAL_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Models
MODELS_DIR = BASE_DIR / "models"

# Local model path (Qwen 1.5)
LOCAL_MODEL_PATH = MODELS_DIR / "qwen1.5"

# Allow override via environment variable
_env_path = os.getenv("LOCAL_MODEL_PATH")
if _env_path:
    LOCAL_MODEL_PATH = Path(_env_path)

# Generation parameters (Local
LOCAL_MAX_NEW_TOKENS = 256
LOCAL_TEMPERATURE = 0.2
LOCAL_TOP_P = 0.9
