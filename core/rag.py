"""Retrieval-augmented generation over the Anthropic S-1.

Pipeline
--------
1. `data/s1_pages.json` holds the plain text of every prospectus page
   (built by scripts/prepare_s1.py from the EDGAR HTML).
2. Pages are split into overlapping chunks; each chunk keeps its printed page
   number and the prospectus section it belongs to (from the table of contents).
3. Chunks are embedded with OpenAI `text-embedding-3-small` and cached to disk.
4. Retrieval is *hybrid*: dense cosine similarity + BM25 keyword scoring,
   merged with reciprocal-rank fusion. Keyword search matters for a filing
   full of exact terms ("Bartz settlement", "Class N", "SAFEs").
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import CHUNKS_PATH, EMBEDDING_MODEL, PAGES_PATH, VECTORS_PATH, get_api_key

# Section start pages, straight from the S-1 table of contents (printed page numbers)
TOC = [
    (1, "Select Defined Terms"),
    (4, "Prospectus Summary"),
    (27, "Risk Factors"),
    (121, "Forward-Looking Statements"),
    (124, "Industry, Market, and Other Data"),
    (125, "Use of Proceeds"),
    (126, "Dividend Policy"),
    (127, "Capitalization"),
    (130, "Dilution"),
    (133, "MD&A"),
    (182, "Business"),
    (246, "Governance Structure"),
    (258, "Management"),
    (268, "Executive Compensation"),
    (286, "Related Person Transactions"),
    (292, "Principal and Selling Stockholders"),
    (296, "Description of Capital Stock"),
    (311, "Shares Eligible for Future Sale"),
    (316, "Tax Consequences to Non-U.S. Holders"),
    (321, "Underwriting"),
    (338, "Legal Matters"),
]
SECTIONS = [name for _, name in TOC] + ["Financial Statements", "Cover / Front Matter"]

_PAGE_RE = re.compile(r"^\s*(F-\d+|\d+|[ivx]+)\s+Table of Contents", re.I)
_TOKEN_RE = re.compile(r"[a-z0-9$%][a-z0-9$%.\-']*", re.I)


def _page_label_and_section(text: str) -> tuple[str, str]:
    m = _PAGE_RE.match(text)
    if not m:
        return "cover", "Cover / Front Matter"
    label = m.group(1)
    if label.upper().startswith("F-"):
        return label.upper(), "Financial Statements"
    if not label.isdigit():
        return label, "Cover / Front Matter"
    n = int(label)
    section = TOC[0][1]
    for start, name in TOC:
        if n >= start:
            section = name
    return label, section


def build_chunks(chunk_size: int = 1400, overlap: int = 200) -> list[dict]:
    pages = json.loads(PAGES_PATH.read_text())
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for p in pages:
        label, section = _page_label_and_section(p["text"])
        body = _PAGE_RE.sub("", p["text"], count=1).strip()
        for piece in splitter.split_text(body):
            chunks.append({"id": len(chunks), "page": label, "section": section, "text": piece})
    return chunks


def _tokenize(text: str) -> list[str]:
    return [t.lower().strip(".'") for t in _TOKEN_RE.findall(text)]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(_tokenize(d)) for d in docs]
        self.lens = np.array([sum(tf.values()) for tf in self.tfs], dtype=float)
        self.avg = self.lens.mean()
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = set(_tokenize(query))
        out = np.zeros(len(self.tfs))
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            out[i] = s
        return out


@dataclass
class Hit:
    page: str
    section: str
    text: str
    score: float

    def cite(self) -> str:
        return f"[S-1 p. {self.page} · {self.section}]"


class S1Index:
    def __init__(self, chunks: list[dict], vectors: np.ndarray):
        self.chunks = chunks
        self.vectors = vectors
        self.bm25 = BM25([c["text"] for c in chunks])
        self.embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=get_api_key() or "sk-missing")

    # ---- construction -------------------------------------------------
    @classmethod
    def exists(cls) -> bool:
        return CHUNKS_PATH.exists() and VECTORS_PATH.exists()

    @classmethod
    def load(cls) -> "S1Index":
        return cls(json.loads(CHUNKS_PATH.read_text()), np.load(VECTORS_PATH))

    @classmethod
    def build(cls, progress=None, batch_size: int = 128) -> "S1Index":
        chunks = build_chunks()
        embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=get_api_key() or "sk-missing")
        vecs = []
        for i in range(0, len(chunks), batch_size):
            batch = [c["text"] for c in chunks[i : i + batch_size]]
            vecs.extend(embedder.embed_documents(batch))
            if progress:
                progress(min(1.0, (i + batch_size) / len(chunks)))
        arr = np.array(vecs, dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)
        CHUNKS_PATH.write_text(json.dumps(chunks))
        np.save(VECTORS_PATH, arr)
        return cls(chunks, arr)

    # ---- retrieval ------------------------------------------------------
    def search(self, query: str, k: int = 6, section: str | None = None, rrf_k: int = 60) -> list[Hit]:
        q = np.array(self.embedder.embed_query(query), dtype=np.float32)
        q /= np.linalg.norm(q)
        dense = self.vectors @ q
        sparse = self.bm25.scores(query)

        mask = np.ones(len(self.chunks), dtype=bool)
        if section and section in SECTIONS:
            mask = np.array([c["section"] == section for c in self.chunks])

        def ranks(scores):
            s = np.where(mask, scores, -np.inf)
            order = np.argsort(-s)
            r = np.empty_like(order)
            r[order] = np.arange(len(order))
            return r

        fused = 1 / (rrf_k + ranks(dense)) + 1 / (rrf_k + ranks(sparse))
        fused = np.where(mask, fused, -np.inf)
        top = np.argsort(-fused)[:k]
        return [
            Hit(self.chunks[i]["page"], self.chunks[i]["section"], self.chunks[i]["text"], float(fused[i]))
            for i in top
        ]

    def section_counts(self) -> Counter:
        return Counter(c["section"] for c in self.chunks)


def format_hits(hits: list[Hit]) -> str:
    return "\n\n".join(f"{h.cite()}\n{h.text}" for h in hits)
