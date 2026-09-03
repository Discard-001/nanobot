"""Reranker service for RAG knowledge base.

Supports:
- ModelScope API (BAAI/bge-reranker-v2-m3) - requires API key
- Local sentence-transformers (BAAI/bge-reranker-v2-m3) - no API key needed

Usage:
    # Local mode (no API key)
    reranker = RerankerService(provider="local")
    results = await reranker.rerank("query", ["doc1", "doc2"])

    # ModelScope API mode
    reranker = RerankerService(provider="modelscope", api_key="your_key")
    results = await reranker.rerank("query", ["doc1", "doc2"])
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from loguru import logger


class RerankerService:
    """Reranker service for improving retrieval precision.

    Takes a query and a list of candidate documents, returns them
    sorted by relevance with confidence scores.

    ModelScope API:
        - Endpoint: https://api-inference.modelscope.cn/v1/rerank
        - Model: BAAI/bge-reranker-v2-m3

    Local mode:
        - Uses sentence-transformers CrossEncoder
        - Downloads model on first use
    """

    def __init__(
        self,
        provider: str = "modelscope",
        model_name: str = "BAAI/bge-reranker-v2-m3",
        api_key: str = "",
        base_url: str = "",
        top_k: int = 5,
    ):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        # Per-provider default when base_url is not configured explicitly
        provider_defaults = {
            "modelscope": "https://api-inference.modelscope.cn/v1",
            "siliconflow": "https://api.siliconflow.cn/v1",
        }
        self.base_url = base_url.rstrip("/") or provider_defaults.get(provider, "")
        self.top_k = top_k
        self._local_model = None

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank documents by relevance to query.

        Args:
            query: Query string.
            documents: List of document strings to rerank.
            top_k: Number of top results to return (default: self.top_k).

        Returns:
            List of dicts with keys:
                - index: Original index in documents list
                - score: Relevance score (0-1)
                - text: The document text
            Sorted by score descending.
        """
        if not documents:
            return []

        k = top_k or self.top_k

        if self.provider in ("modelscope", "siliconflow"):
            # Both expose the same OpenAI-compatible /rerank endpoint
            results = await self._rerank_via_modelscope(query, documents)
        elif self.provider == "local":
            results = await self._rerank_local(query, documents)
        else:
            raise ValueError(f"Unknown reranker provider: {self.provider}")

        # Sort by score descending and take top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    async def _rerank_via_modelscope(
        self, query: str, documents: list[str]
    ) -> list[dict[str, Any]]:
        """Rerank via ModelScope API.

        Args:
            query: Query string.
            documents: List of document strings.

        Returns:
            List of rerank results.
        """
        if not self.api_key:
            raise ValueError(
                "ModelScope API key is required for reranking. "
                "Set MODELSCOPE_API_TOKEN environment variable or "
                "configure rag.reranker.api_key in config."
            )

        url = f"{self.base_url}/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "return_documents": False,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=60.0)
            response.raise_for_status()
            data = response.json()

        results = []
        for item in data.get("results", []):
            idx = item.get("index", 0)
            results.append({
                "index": idx,
                "score": item.get("relevance_score", 0.0),
                "text": documents[idx] if idx < len(documents) else "",
            })

        logger.debug("Reranked {} documents via ModelScope", len(documents))
        return results

    async def _rerank_local(
        self, query: str, documents: list[str]
    ) -> list[dict[str, Any]]:
        """Rerank using local CrossEncoder model.

        Args:
            query: Query string.
            documents: List of document strings.

        Returns:
            List of rerank results.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._rerank_local_sync, query, documents
        )

    def _rerank_local_sync(
        self, query: str, documents: list[str]
    ) -> list[dict[str, Any]]:
        """Synchronous local reranking (runs in thread pool)."""
        if self._local_model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for local reranking. "
                    "Install with: pip install nanobot-ai[rag]"
                ) from e

            logger.info("Loading local reranker model: {}", self.model_name)
            self._local_model = CrossEncoder(
                self.model_name,
                max_length=512,
            )
            logger.info("Reranker model loaded")

        # Prepare pairs for scoring
        pairs = [(query, doc) for doc in documents]

        # Score all pairs
        scores = self._local_model.predict(pairs)

        # Build results
        results = []
        for idx, score in enumerate(scores):
            results.append({
                "index": idx,
                "score": float(score),
                "text": documents[idx],
            })

        logger.debug("Reranked {} documents locally", len(documents))
        return results
