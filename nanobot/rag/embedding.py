"""Embedding service for RAG knowledge base.

Supports:
- ModelScope API (BAAI/bge-m3) - requires API key
- Local sentence-transformers (BAAI/bge-m3) - no API key needed

Usage:
    # Local mode (no API key)
    service = EmbeddingService(provider="local", model_name="BAAI/bge-m3")
    vectors = await service.embed_texts(["hello world"])

    # ModelScope API mode
    service = EmbeddingService(provider="modelscope", api_key="your_key")
    vectors = await service.embed_texts(["hello world"])
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from loguru import logger


class EmbeddingService:
    """Embedding service supporting ModelScope and local models.

    ModelScope API:
        - Endpoint: https://api-inference.modelscope.cn/v1/embeddings
        - Model: BAAI/bge-m3 (1024 dimensions)
        - API Key: Set via env MODELSCOPE_API_TOKEN or config

    Local mode:
        - Uses sentence-transformers library
        - Downloads model on first use
        - Runs on CPU/GPU/MPS based on config
    """

    def __init__(
        self,
        provider: str = "modelscope",
        model_name: str = "BAAI/bge-m3",
        api_key: str = "",
        base_url: str = "https://api-inference.modelscope.cn/v1",
        device: str = "cpu",
        dimensions: int = 1024,
    ):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.device = device
        self.dimensions = dimensions
        self._local_model = None

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (one per input text).
        """
        if not texts:
            return []

        if self.provider == "modelscope":
            return await self._embed_via_modelscope(texts)
        elif self.provider == "local":
            return await self._embed_local(texts)
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query string.

        Args:
            query: Query text to embed.

        Returns:
            Embedding vector.
        """
        results = await self.embed_texts([query])
        return results[0]

    async def _embed_via_modelscope(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via ModelScope API.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.

        Raises:
            ValueError: If API key is not set.
            httpx.HTTPStatusError: If API request fails.
        """
        if not self.api_key:
            raise ValueError(
                "ModelScope API key is required. "
                "Set MODELSCOPE_API_TOKEN environment variable or "
                "configure rag.embedding.api_key in config."
            )

        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "input": texts,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=60.0)
            response.raise_for_status()
            data = response.json()

        # Extract embeddings from response
        embeddings = []
        for item in data.get("data", []):
            embeddings.append(item["embedding"])

        logger.debug("Embedded {} texts via ModelScope", len(texts))
        return embeddings

    async def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using local sentence-transformers model.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_local_sync, texts)

    def _embed_local_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous local embedding (runs in thread pool)."""
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for local embedding. "
                    "Install with: pip install nanobot-ai[rag]"
                ) from e

            logger.info("Loading local embedding model: {}", self.model_name)
            self._local_model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
            logger.info("Model loaded on device: {}", self.device)

        # Encode texts
        embeddings = self._local_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        logger.debug("Embedded {} texts locally", len(texts))
        return embeddings.tolist()
