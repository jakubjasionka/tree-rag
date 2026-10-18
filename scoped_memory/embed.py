"""CPU-offloaded embedding abstraction (0 MB GPU VRAM) and sentence-boundary text chunking."""

from __future__ import annotations

import hashlib
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
DEFAULT_EMBED_DIM = 384
DEFAULT_MODEL_NAME = "compact-multilingual-cpu-100m"
DEFAULT_EMBED_URL = "http://embed.example:8089/v1"


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize a 2D float32 array row-wise so inner product equals cosine similarity."""
    arr = np.asarray(vectors, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype("float32")


class Embedder(ABC):
    """Abstract interface for embedding models (CPU offload or deterministic mock)."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector dimensionality produced by this embedder."""

    @abstractmethod
    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a sequence of document passages into an `(N, dim)` L2-normalized float32 matrix."""

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query into a `(dim,)` L2-normalized float32 vector."""


class CpuHttpEmbedder(Embedder):
    """OpenAI-compatible HTTP client for a compact multilingual CPU embedding model (~100M params).

    Offloads embedding computation entirely to host/edge CPU (0 MB GPU VRAM), reserving
    100% of GPU memory for the primary generative LLM KV-cache.
    """

    def __init__(
        self,
        endpoint_url: str | None = None,
        model_name: str | None = None,
        dim: int = DEFAULT_EMBED_DIM,
        passage_prefix: str = "passage: ",
        query_prefix: str = "query: ",
        batch_size: int = 128,
        timeout_sec: float = 15.0,
    ) -> None:
        self._endpoint_url = (
            endpoint_url
            or os.environ.get("EMBED_URL")
            or os.environ.get("TREE_RAG_EMBED_URL")
            or DEFAULT_EMBED_URL
        ).rstrip("/")
        self._model_name = (
            model_name
            or os.environ.get("TREE_RAG_EMBED_MODEL")
            or DEFAULT_MODEL_NAME
        )
        self._dim = dim
        self._passage_prefix = passage_prefix
        self._query_prefix = query_prefix
        self._batch_size = max(1, batch_size)
        self._timeout_sec = timeout_sec

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def endpoint_url(self) -> str:
        return self._endpoint_url

    @property
    def model_name(self) -> str:
        return self._model_name

    def _request_embeddings(self, prefixed_texts: list[str]) -> np.ndarray:
        if not prefixed_texts:
            return np.zeros((0, self._dim), dtype="float32")
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("The 'requests' library is required for CpuHttpEmbedder") from exc

        batches: list[np.ndarray] = []
        for i in range(0, len(prefixed_texts), self._batch_size):
            batch = prefixed_texts[i : i + self._batch_size]
            resp = requests.post(
                f"{self._endpoint_url}/embeddings",
                json={"input": batch, "model": self._model_name},
                timeout=self._timeout_sec,
            )
            resp.raise_for_status()
            payload = resp.json()
            data = sorted(payload["data"], key=lambda item: item["index"])
            vecs = np.array([item["embedding"] for item in data], dtype="float32")
            batches.append(l2_normalize(vecs))
        merged = np.vstack(batches)
        self._dim = int(merged.shape[1])
        return merged

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        prefixed = [f"{self._passage_prefix}{t}" for t in texts]
        return self._request_embeddings(prefixed)

    def embed_query(self, query: str) -> np.ndarray:
        mat = self._request_embeddings([f"{self._query_prefix}{query}"])
        return mat[0]


class MockEmbedder(Embedder):
    """Deterministic, zero-dependency CPU embedder for unit tests, CI, and offline demos.

    Projects lexical tokens and character n-grams into a fixed-dimensional space via
    SHA-256 feature hashing so semantically overlapping queries and passages yield
    high cosine similarity while unrelated domains remain near-orthogonal.
    """

    def __init__(self, dim: int = DEFAULT_EMBED_DIM) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype="float32")
        cleaned = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
        tokens = [tok for tok in cleaned.split() if len(tok) >= 2]
        if not tokens:
            tokens = ["_empty_"]

        features: list[tuple[str, float]] = []
        for tok in tokens:
            features.append((f"w:{tok}", 2.0))
            if len(tok) >= 4:
                features.append((f"p:{tok[:4]}", 1.0))
            for idx in range(len(tok) - 2):
                features.append((f"g:{tok[idx : idx + 3]}", 0.5))

        for feat, weight in features:
            digest = hashlib.sha256(feat.encode("utf-8")).digest()
            for slot in range(0, 16, 4):
                raw_int = int.from_bytes(digest[slot : slot + 4], "little")
                bucket = raw_int % self._dim
                sign = 1.0 if (raw_int & 1) == 0 else -1.0
                vec[bucket] += sign * weight

        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return (vec / norm).astype("float32")

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype="float32")
        rows = [self._encode_one(t) for t in texts]
        return np.vstack(rows).astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode_one(query)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split long text on natural sentence or word boundaries without mid-word truncation."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []
    if len(cleaned) <= size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    total_len = len(cleaned)
    step_min = max(60, size - overlap)

    while start < total_len:
        end = min(start + size, total_len)
        if end < total_len:
            window_start = start + int(size * 0.65)
            best_cut = -1
            for sep in (". ", "! ", "? ", "; ", " — ", " - "):
                pos = cleaned.rfind(sep, window_start, end)
                if pos != -1 and pos + len(sep) > best_cut:
                    best_cut = pos + len(sep)
            if best_cut == -1:
                space_pos = cleaned.rfind(" ", window_start, end)
                if space_pos != -1:
                    best_cut = space_pos + 1
            if best_cut > start:
                end = best_cut

        chunk_str = cleaned[start:end].strip()
        if chunk_str:
            chunks.append(chunk_str)
        if end >= total_len:
            break

        next_start = max(start + step_min, end - overlap)
        if start < next_start < total_len and cleaned[next_start - 1] != " ":
            sp = cleaned.find(" ", next_start, min(next_start + 45, total_len))
            if sp != -1:
                next_start = sp + 1
        start = next_start

    return chunks


def _chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Internal alias for `chunk_text`."""
    return chunk_text(text, size=size, overlap=overlap)
