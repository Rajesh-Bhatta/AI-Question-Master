from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedChunk:
    text: str
    score: float


def chunk_text(text: str, chunk_size: int = 220, overlap: int = 40) -> List[str]:
    """Split text into overlapping word chunks while preserving useful context."""
    words = " ".join(text.split()).split()
    if not words:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    return [
        " ".join(words[start:start + chunk_size])
        for start in range(0, len(words), step)
        if words[start:start + chunk_size]
    ]


class RAGStore:
    def __init__(self, embedder: SentenceTransformer, chunks: List[str], index=None):
        if not chunks:
            raise ValueError("Cannot build a retrieval store from empty text")
        self.embedder = embedder
        self.chunks = chunks
        if index is None:
            vectors = self._encode(chunks)
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
        self.index = index

    def _encode(self, texts: List[str]) -> np.ndarray:
        vectors = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        return np.ascontiguousarray(vectors)

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        if not query.strip():
            query = "important concepts, definitions, facts, and relationships"
        limit = min(max(top_k, 1), len(self.chunks))
        scores, indexes = self.index.search(self._encode([query]), limit)
        return [
            RetrievedChunk(self.chunks[index], float(score))
            for score, index in zip(scores[0], indexes[0])
            if index >= 0
        ]


def build_vector_store(
    text: str,
    embedder: SentenceTransformer,
    cache_dir: str = "rag_cache",
) -> RAGStore:
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("Cannot build a retrieval store from empty text")

    cache_key = hashlib.sha256(
        json.dumps({"text": text, "model": "all-MiniLM-L6-v2"}, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cache_path = Path(cache_dir)
    index_path = cache_path / f"{cache_key}.faiss"
    chunks_path = cache_path / f"{cache_key}.json"

    if index_path.is_file() and chunks_path.is_file():
        try:
            cached_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            if cached_chunks == chunks:
                return RAGStore(embedder, cached_chunks, faiss.read_index(str(index_path)))
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            pass

    store = RAGStore(embedder, chunks)
    cache_path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(store.index, str(index_path))
    chunks_path.write_text(json.dumps(chunks), encoding="utf-8")
    return store