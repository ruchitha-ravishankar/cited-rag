"""BM25 + dense + hybrid (reciprocal rank fusion) retrieval with an optional cross-encoder reranker.

The dense side uses a multilingual embedding model (BGE-M3 by default) so Hindi questions can
retrieve English passages. If sentence-transformers / the model is unavailable it falls back to
word-level TF-IDF so the project still runs offline (but cross-lingual retrieval will not work).
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from .ingest import Chunk

log = logging.getLogger("cited")


def tokenize(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Indexes:
    """Builds BM25 + dense indexes once; Retriever objects share them (cheap ablations)."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        texts = [c.index_text() for c in chunks]
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.backend = "tfidf"
        self.model = None
        if config.FORCE_BACKEND != "tfidf":
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(config.EMBED_MODEL)
                self.backend = "st"
            except Exception as e:  # noqa: BLE001
                log.warning("Embedding model unavailable (%s). Falling back to TF-IDF.", e)
        if self.backend == "st":
            self.emb = self._cached_embeddings(texts)
        else:
            self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
            self.emb = self.vec.fit_transform(texts)

    def _cached_embeddings(self, texts: list[str]) -> np.ndarray:
        key = hashlib.md5((config.EMBED_MODEL + "\n".join(texts)).encode()).hexdigest()[:16]
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = config.CACHE_DIR / f"emb_{key}.npy"
        if path.exists():
            return np.load(path)
        emb = self.model.encode(texts, normalize_embeddings=True, batch_size=16, show_progress_bar=True)
        np.save(path, emb)
        return emb

    def bm25_scores(self, q: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(q)), dtype=float)

    def dense_scores(self, q: str) -> np.ndarray:
        if self.backend == "st":
            qv = self.model.encode([q], normalize_embeddings=True)[0]
            return self.emb @ qv
        return (self.emb @ self.vec.transform([q]).T).toarray().ravel()


def _ranks(scores: np.ndarray, n: int) -> list[int]:
    if scores.max() <= 0:  # e.g. Hindi query vs English-only BM25 vocabulary -> no signal
        return []
    return list(np.argsort(-scores)[:n])


class Retriever:
    def __init__(self, indexes: Indexes, mode: str = "hybrid", rerank: bool = False):
        assert mode in {"bm25", "dense", "hybrid"}
        self.ix, self.mode, self.rerank = indexes, mode, rerank
        self._reranker = None
        if rerank:
            try:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(config.RERANK_MODEL)
            except Exception as e:  # noqa: BLE001
                log.warning("Reranker unavailable (%s). Continuing without it.", e)
                self.rerank = False

    @property
    def name(self) -> str:
        return self.mode + ("+rerank" if self.rerank else "")

    def search(self, q: str, k: int = config.TOP_K) -> tuple[list[Hit], float]:
        """Returns (top-k hits, confidence). Confidence = best dense cosine over the whole corpus,
        independent of ranking mode, so refusal behaviour is comparable across ablations."""
        dense = self.ix.dense_scores(q)
        confidence = float(dense.max())
        n = config.CANDIDATES
        if self.mode == "dense":
            order = list(np.argsort(-dense)[:n])
            fused = {i: float(dense[i]) for i in order}
        elif self.mode == "bm25":
            b = self.ix.bm25_scores(q)
            order = _ranks(b, n)
            fused = {i: float(b[i]) for i in order}
        else:
            fused: dict[int, float] = {}
            for ranking in (_ranks(self.ix.bm25_scores(q), n), list(np.argsort(-dense)[:n])):
                for r, i in enumerate(ranking):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (config.RRF_K + r + 1)
        cands = sorted(fused.items(), key=lambda kv: -kv[1])[:n]
        if self.rerank and cands:
            pairs = [(q, self.ix.chunks[i].index_text()) for i, _ in cands]
            rs = self._reranker.predict(pairs)
            cands = sorted(((i, float(s)) for (i, _), s in zip(cands, rs)), key=lambda kv: -kv[1])
        return [Hit(self.ix.chunks[i], s) for i, s in cands[:k]], confidence
