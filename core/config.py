"""Shared configuration: paths, model choices, API key handling."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PAGES_PATH = DATA_DIR / "s1_pages.json"
CHUNKS_PATH = DATA_DIR / "s1_chunks.json"
VECTORS_PATH = DATA_DIR / "s1_vectors.npy"

S1_URL = (
    "https://www.sec.gov/Archives/edgar/data/2130357/000162828026068676/"
    "anthropicpbc-sx1ax1.htm"
)
S1_LABEL = "Anthropic, PBC — Form S-1/A (Amendment No. 1), filed Nov 2, 2026"

# Look for a .env here first, then fall back to the class streamlit folder's .env
load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parent / "streamlit" / ".env")

# Same family of models used in the class streamlit templates (chatbot_example.py)
CHAT_MODELS = ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"]
EMBEDDING_MODEL = "text-embedding-3-small"


def get_api_key() -> str | None:
    """Template convention: accept OPENAI_API_KEY or OPENAI_API_TOKEN."""
    return os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_TOKEN")


def set_api_key(key: str) -> None:
    os.environ["OPENAI_API_KEY"] = key
