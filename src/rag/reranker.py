"""Optional second-stage rerankers for RAG retrieval results."""

import asyncio
import logging
import math
from dataclasses import replace
from typing import Any, Dict, List, Protocol, Sequence

from src.memory import tokenize
from src.rag.vector_store import RagSearchResult

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    async def rerank(
        self, query: str, results: Sequence[RagSearchResult]
    ) -> List[RagSearchResult]:
        """Return the supplied results reordered and rescored for the query."""


class NoOpReranker:
    async def rerank(
        self, query: str, results: Sequence[RagSearchResult]
    ) -> List[RagSearchResult]:
        return list(results)


class TokenOverlapReranker:
    """Deterministic lexical reranker used for tests and lightweight fallback."""

    async def rerank(
        self, query: str, results: Sequence[RagSearchResult]
    ) -> List[RagSearchResult]:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return list(results)

        rescored = []
        for result in results:
            chunk_tokens = set(tokenize(result.chunk.text))
            score = len(query_tokens.intersection(chunk_tokens)) / len(query_tokens)
            rescored.append(_with_rerank_score(result, score))
        rescored.sort(key=lambda item: -item.score)
        return rescored


class LocalHttpReranker:
    """Reranker for local HTTP cross-encoder services."""

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config or {})
        self.url = str(self.config.get("url") or "").strip()
        self.model = self.config.get("model")
        self.timeout = float(self.config.get("timeout", 30))

    async def rerank(
        self, query: str, results: Sequence[RagSearchResult]
    ) -> List[RagSearchResult]:
        if not self.url:
            raise RuntimeError("RAG rerank URL is not configured")
        import httpx

        payload: Dict[str, Any] = {
            "query": query,
            "documents": [result.chunk.text for result in results],
            "top_n": int(self.config.get("top_n", len(results))),
        }
        if self.model:
            payload["model"] = self.model
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(_rerank_url(self.url), json=payload)
            response.raise_for_status()
            data = response.json()
        return _apply_scores(results, _parse_scores(data, len(results)))


class SentenceTransformersReranker:
    """Local sentence-transformers CrossEncoder reranker."""

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config or {})
        self.model = self.config.get("model") or "BAAI/bge-reranker-v2-m3"
        self._cross_encoder = None

    async def rerank(
        self, query: str, results: Sequence[RagSearchResult]
    ) -> List[RagSearchResult]:
        if self._cross_encoder is None:
            self._cross_encoder = await asyncio.to_thread(self._load_model)
        pairs = [(query, result.chunk.text) for result in results]
        scores = await asyncio.to_thread(self._cross_encoder.predict, pairs)
        return _apply_scores(results, list(scores))

    def _load_model(self):
        from sentence_transformers import CrossEncoder

        return CrossEncoder(str(self.model))


def build_reranker(config: Dict[str, Any]) -> Reranker:
    if not bool(config.get("enabled", False)):
        return NoOpReranker()
    provider = str(config.get("provider") or "sentence_transformers").lower()
    if provider in {"none", "disabled", ""}:
        return NoOpReranker()
    if provider in {"token_overlap", "keyword", "lexical"}:
        return TokenOverlapReranker()
    if provider in {"local_http", "http"}:
        return LocalHttpReranker(config)
    if provider in {"sentence_transformers", "cross_encoder", "local"}:
        return SentenceTransformersReranker(config)
    logger.warning("Unknown RAG reranker provider '%s'; rerank disabled", provider)
    return NoOpReranker()


def _apply_scores(
    results: Sequence[RagSearchResult], scores: Sequence[Any]
) -> List[RagSearchResult]:
    if len(scores) != len(results):
        raise ValueError("reranker returned a score count that does not match results")
    rescored = [
        _with_rerank_score(result, _normalize_score(float(score)))
        for result, score in zip(results, scores)
    ]
    rescored.sort(key=lambda item: -item.score)
    return rescored


def _with_rerank_score(result: RagSearchResult, score: float) -> RagSearchResult:
    source = result.match_source
    if "rerank" not in source:
        source = f"{source}+rerank"
    return replace(result, score=max(0.0, float(score)), match_source=source)


def _normalize_score(score: float) -> float:
    if 0.0 <= score <= 1.0:
        return score
    if score >= 50:
        return 1.0
    if score <= -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-score))


def _rerank_url(url: str) -> str:
    normalized = url.rstrip("/")
    if normalized.endswith(("/rerank", "/score", "/scores")):
        return normalized
    return normalized + "/rerank"


def _parse_scores(data: Any, expected_count: int) -> List[float]:
    if isinstance(data, dict):
        if isinstance(data.get("scores"), list):
            return [float(score) for score in data["scores"]]
        for key in ("results", "data"):
            values = data.get(key)
            if isinstance(values, list):
                return _parse_ranked_items(values, expected_count)
    if isinstance(data, list):
        if all(isinstance(item, (int, float)) for item in data):
            return [float(score) for score in data]
        return _parse_ranked_items(data, expected_count)
    raise ValueError("unsupported reranker response format")


def _parse_ranked_items(items: Sequence[Any], expected_count: int) -> List[float]:
    scores = [0.0] * expected_count
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError("unsupported reranker ranked item format")
        index = int(item.get("index", position))
        if index < 0 or index >= expected_count:
            continue
        scores[index] = float(item.get("score", item.get("relevance_score", 0.0)))
    return scores
