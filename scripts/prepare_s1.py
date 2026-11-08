"""Download Anthropic's S-1/A from SEC EDGAR, extract per-page text, and
(optionally) build the embedding index used for RAG.

    python scripts/prepare_s1.py            # download + extract
    python scripts/prepare_s1.py --index    # also embed (needs OPENAI_API_KEY)
"""
import json
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import DATA_DIR, PAGES_PATH, S1_URL  # noqa: E402

RAW = DATA_DIR / "anthropic_s1a.htm"
# SEC asks automated clients to identify themselves in the User-Agent
HEADERS = {"User-Agent": "Anthropic-S1-Explorer class project research@example.com"}


def download():
    if RAW.exists():
        print(f"Already downloaded: {RAW.name}")
        return
    DATA_DIR.mkdir(exist_ok=True)
    print("Downloading S-1/A from EDGAR…")
    resp = requests.get(S1_URL, headers=HEADERS, timeout=120)
    resp.raise_for_status()
    RAW.write_bytes(resp.content)


def extract():
    soup = BeautifulSoup(RAW.read_text(encoding="utf-8", errors="ignore"), "lxml")
    # The EDGAR HTML renders each printed page as a fixed 612x792pt div
    pages = soup.find_all("div", style=lambda s: s and "height:792pt" in s)
    out = [{"page": i + 1, "text": p.get_text(" ", strip=True)} for i, p in enumerate(pages)]
    out = [o for o in out if len(o["text"]) > 50]
    PAGES_PATH.write_text(json.dumps(out))
    print(f"Extracted {len(out)} text pages -> {PAGES_PATH.name}")


if __name__ == "__main__":
    download()
    extract()
    if "--index" in sys.argv:
        from core.rag import S1Index

        idx = S1Index.build(progress=lambda p: print(f"\rembedding {p:.0%}", end=""))
        print(f"\nIndexed {len(idx.chunks)} chunks")
