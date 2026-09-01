import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent

DB_PATH = Path(os.getenv("DB_PATH", REPO_ROOT.parent / "DB"))

RAW_DIR = DB_PATH / "raw"
EMBEDDINGS_DIR = DB_PATH / "embeddings"
CHROMA_DIR = DB_PATH / "chroma"
BM25_DIR = DB_PATH / "bm-25"

TESTS_DIR = REPO_ROOT / "tests"
PROMPTS_PATH = Path(__file__).resolve().parent / "prompts.yaml"
