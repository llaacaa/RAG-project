from pathlib import Path


EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:3b"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = str(ROOT / "docs")
CHROMA_DIR = str(ROOT / "chroma_db")
COLLECTION_NAME = "docs"