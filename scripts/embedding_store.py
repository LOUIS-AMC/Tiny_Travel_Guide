"""Simple embedding helper using Ollama for RAG scoring."""
from __future__ import annotations

import math
import os
from typing import Iterable, List, Sequence


class EmbeddingClient:
    """Wraps Ollama embeddings with basic caching."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv(
            "OLLAMA_EMBED_MODEL", "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest"
        )
        self._cache: dict[str, List[float]] = {}

    def embed(self, text: str) -> List[float]:
        """Embed a single string, with memoization."""
        if text in self._cache:
            return self._cache[text]
        try:
            import ollama
        except ModuleNotFoundError as exc:  # pragma: no cover - env guard
            raise RuntimeError(
                "Python package 'ollama' is not installed. Run `pip install ollama`."
            ) from exc
        try:
            resp = ollama.embeddings(model=self.model, prompt=text)
            vec = resp.get("embedding") if isinstance(resp, dict) else None
        except Exception as exc:  # pragma: no cover - passthrough for caller
            raise RuntimeError(
                f"Ollama embeddings call failed using model '{self.model}'."
            ) from exc
        if not vec:
            raise RuntimeError("Ollama returned no embedding vector.")
        self._cache[text] = vec
        return vec

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must be same length for cosine similarity.")
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def top_k_by_embedding(
    query: str, items: Iterable[str], embedder: EmbeddingClient, k: int = 10
) -> List[int]:
    """
    Return indices of top-k items sorted by cosine similarity to the query.
    """
    query_vec = embedder.embed(query)
    item_list = list(items)
    if not item_list:
        return []
    item_vecs = embedder.embed_many(item_list)
    scores = [
        (idx, cosine_similarity(query_vec, vec)) for idx, vec in enumerate(item_vecs)
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scores[:k]]
