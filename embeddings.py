"""
embeddings.py — Embedding generation + vector store

Uses sentence-transformers locally (all-MiniLM-L6-v2).
No external API calls — the model runs in-process.
Falls back to the HuggingFace Inference API only if the
`use_hf_api=True` flag is passed (kept for legacy compat).

Pure Python cosine similarity — no numpy required.
Persists to JSON for reproducibility.
"""

import json
import logging
import math
import os
import time
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
STORE_PATH = Path("vector_store.json")

# ── Local embedding model (loaded once, reused) ───────────────────────────────

@lru_cache(maxsize=1)
def _load_model():
    """
    Load and cache the SentenceTransformer model.
    First call downloads the model (~80 MB) if not already cached.
    Subsequent calls return the cached instance instantly.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Run:  pip install sentence-transformers"
        )
    logger.info(f"Loading embedding model '{MODEL_NAME}' (one-time download if not cached)...")
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Embedding model loaded.")
    return model


# ── Embedding generation ──────────────────────────────────────────────────────

def get_embeddings(
    texts: list[str],
    hf_token: str = "",          # kept for API compatibility — unused locally
    retries: int = 3,
    batch_size: int = 64,
    use_hf_api: bool = False,
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    By default, runs the model locally via sentence-transformers.
    Pass use_hf_api=True to use the HuggingFace Inference API instead
    (requires a valid hf_token and a live endpoint).

    Returns a list of float vectors, one per input text.
    """
    if use_hf_api:
        return _get_embeddings_hf_api(texts, hf_token, retries)

    return _get_embeddings_local(texts, batch_size)


def get_embed_text(chunk: dict) -> str:
    """
    Return the text to embed for a chunk.
    Uses lowercased 'text_normalized' if present (ingestion v2+)
    to eliminate case-sensitivity mismatches during retrieval.
    Falls back to plain 'text' for legacy chunks.
    """
    return chunk.get("text_normalized") or chunk.get("text", "")


def _get_embeddings_local(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Embed texts using the locally loaded SentenceTransformer model."""
    if not texts:
        return []

    model = _load_model()

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # encode() returns a numpy array; convert to plain Python lists
        vecs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.extend(v.tolist() for v in vecs)

    return all_embeddings


def _get_embeddings_hf_api(
    texts: list[str],
    hf_token: str,
    retries: int = 3,
) -> list[list[float]]:
    """
    Legacy: generate embeddings via HuggingFace Inference API.
    Only use this if you have a confirmed live endpoint.
    """
    import requests

    HF_API_URL = (
        "https://api-inference.huggingface.co/pipeline/feature-extraction/"
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    headers = {"Authorization": f"Bearer {hf_token}"}
    all_embeddings: list[list[float]] = []
    batch_size = 32

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        for attempt in range(retries):
            try:
                response = requests.post(
                    HF_API_URL,
                    headers=headers,
                    json={"inputs": batch, "options": {"wait_for_model": True}},
                    timeout=60,
                )

                if response.status_code == 503:
                    wait = int(response.headers.get("X-Wait-For-Model", "20"))
                    logger.warning(f"HF model loading, waiting {wait}s...")
                    time.sleep(wait)
                    continue

                if response.status_code == 429:
                    logger.warning("HF rate limited, waiting 10s...")
                    time.sleep(10)
                    continue

                response.raise_for_status()
                result = response.json()

                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        all_embeddings.extend(result)
                    else:
                        all_embeddings.append(result)
                break

            except requests.RequestException as e:
                if attempt == retries - 1:
                    raise RuntimeError(
                        f"Embedding API failed after {retries} attempts: {e}"
                    )
                time.sleep(2**attempt)

        if i + batch_size < len(texts):
            time.sleep(0.5)

    return all_embeddings


# ── Pure Python Cosine Similarity ─────────────────────────────────────────────

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 on zero vectors."""
    na, nb = _norm(a), _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return _dot(a, b) / (na * nb)


# ── Vector Store ──────────────────────────────────────────────────────────────


class VectorStore:
    """
    Simple JSON-backed vector store.
    Each entry: {chunk_id, document, chunk_index, text, embedding, metadata}
    """

    def __init__(self, store_path: Path = STORE_PATH):
        self.store_path = store_path
        self.entries: list[dict] = []
        self._load()

    def _load(self):
        if self.store_path.exists():
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    self.entries = json.load(f)
                logger.info(f"Loaded {len(self.entries)} entries from store")
            except Exception as e:
                logger.error(f"Failed to load store: {e}")
                self.entries = []

    def _save(self):
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False)

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]):
        """Add chunks with their embeddings to the store."""
        for chunk, emb in zip(chunks, embeddings):
            entry = {**chunk, "embedding": emb}
            # Remove existing entry with same chunk_id (dedup)
            self.entries = [
                e for e in self.entries if e["chunk_id"] != chunk["chunk_id"]
            ]
            self.entries.append(entry)
        self._save()
        logger.info(f"Store now has {len(self.entries)} entries")

    def remove_document(self, filename: str):
        """Remove all chunks belonging to a document."""
        before = len(self.entries)
        self.entries = [e for e in self.entries if e["document"] != filename]
        self._save()
        logger.info(f"Removed {before - len(self.entries)} entries for '{filename}'")

    def list_documents(self) -> list[str]:
        """Return unique document names in the store."""
        return sorted(set(e["document"] for e in self.entries))

    def chunk_count(self) -> int:
        return len(self.entries)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.3,
        filter_document: str | None = None,
    ) -> list[dict]:
        """
        Top-k cosine similarity search.
        Returns results sorted by similarity score descending.
        """
        if not self.entries:
            return []

        scored = []
        for entry in self.entries:
            if filter_document and entry["document"] != filter_document:
                continue
            score = cosine_similarity(query_embedding, entry["embedding"])
            if score >= min_score:
                scored.append(
                    {
                        "chunk_id": entry["chunk_id"],
                        "document": entry["document"],
                        "chunk_index": entry["chunk_index"],
                        "text": entry["text"],
                        "similarity_score": round(score, 4),
                    }
                )

        scored.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored[:top_k]

    def clear(self):
        """Wipe the entire store."""
        self.entries = []
        self._save()
        logger.info("Vector store cleared")