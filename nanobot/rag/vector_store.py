"""FAISS vector store for RAG knowledge base.

Provides persistent vector storage using Facebook AI Similarity Search (FAISS).

Features:
- Local file-based storage (no external services needed)
- Automatic index creation and management
- Metadata storage alongside vectors
- Similarity search with threshold filtering

Usage:
    store = FAISSVectorStore("data/vector_store", dimensions=1024)
    await store.add(vectors, metadata_list)
    results = await store.search(query_vector, top_k=10)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger


class FAISSVectorStore:
    """FAISS-based vector store with persistent storage.

    Stores vectors in a FAISS index and metadata in a JSON file.
    Supports:
    - Adding vectors with metadata
    - Similarity search (L2 distance)
    - Persistence to disk
    - Loading from disk

    File structure:
        store_path/
            index.faiss      # FAISS index file
            metadata.json    # Vector metadata
    """

    def __init__(self, store_path: str | Path, dimensions: int = 1024):
        """Initialize vector store.

        Args:
            store_path: Directory to store index files.
            dimensions: Vector dimensions (must match embedding model).
        """
        self.store_path = Path(store_path)
        self.dimensions = dimensions
        self._index = None
        self._metadata: list[dict[str, Any]] = []
        self._id_counter: int = 0

    @property
    def index_file(self) -> Path:
        """Path to FAISS index file."""
        return self.store_path / "index.faiss"

    @property
    def metadata_file(self) -> Path:
        """Path to metadata JSON file."""
        return self.store_path / "metadata.json"

    @property
    def count(self) -> int:
        """Number of vectors in the store."""
        return len(self._metadata)

    async def initialize(self):
        """Initialize or load the vector store."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._initialize_sync)

    def _initialize_sync(self):
        """Synchronous initialization (runs in thread pool)."""
        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "FAISS is required for vector storage. "
                "Install with: pip install nanobot-ai[rag]"
            ) from e

        self.store_path.mkdir(parents=True, exist_ok=True)

        if self.index_file.exists() and self.metadata_file.exists():
            # Load existing index
            self._index = faiss.read_index(str(self.index_file))
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._metadata = data.get("metadata", [])
                self._id_counter = data.get("id_counter", len(self._metadata))
            logger.info(
                "Loaded FAISS index: {} vectors, {} dimensions",
                self._index.ntotal,
                self.dimensions,
            )
        else:
            # Create new index
            self._index = faiss.IndexFlatL2(self.dimensions)
            self._metadata = []
            self._id_counter = 0
            logger.info(
                "Created new FAISS index: {} dimensions",
                self.dimensions,
            )

    async def add(
        self,
        vectors: list[list[float]],
        metadata_list: list[dict[str, Any]],
    ) -> list[int]:
        """Add vectors with metadata to the store.

        Args:
            vectors: List of embedding vectors.
            metadata_list: List of metadata dicts (one per vector).

        Returns:
            List of assigned IDs.

        Raises:
            ValueError: If vectors and metadata have different lengths.
        """
        if len(vectors) != len(metadata_list):
            raise ValueError(
                f"Vectors ({len(vectors)}) and metadata ({len(metadata_list)}) "
                "must have the same length"
            )

        if not vectors:
            return []

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._add_sync, vectors, metadata_list
        )

    def _add_sync(
        self,
        vectors: list[list[float]],
        metadata_list: list[dict[str, Any]],
    ) -> list[int]:
        """Synchronous add (runs in thread pool)."""
        import numpy as np

        if self._index is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")

        # Convert to numpy array
        vectors_np = np.array(vectors, dtype=np.float32)

        # Add to FAISS index
        self._index.add(vectors_np)

        # Assign IDs and store metadata
        ids = []
        for meta in metadata_list:
            self._id_counter += 1
            meta["_id"] = self._id_counter
            self._metadata.append(meta)
            ids.append(self._id_counter)

        logger.debug("Added {} vectors to FAISS store", len(vectors))
        return ids

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors.

        Args:
            query_vector: Query embedding vector.
            top_k: Number of results to return.
            threshold: Optional similarity threshold (0-1). Results with
                      score below this are filtered out.

        Returns:
            List of dicts with keys:
                - id: Vector ID
                - score: Similarity score (lower is more similar for L2)
                - metadata: Stored metadata
            Sorted by similarity (best first).
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._search_sync, query_vector, top_k, threshold
        )

    def _search_sync(
        self,
        query_vector: list[float],
        top_k: int,
        threshold: float | None,
    ) -> list[dict[str, Any]]:
        """Synchronous search (runs in thread pool)."""
        import numpy as np

        if self._index is None:
            return []

        # Prepare query
        query_np = np.array([query_vector], dtype=np.float32)

        # Search
        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(query_np, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue

            # Convert L2 distance to similarity score (0-1)
            # Using: similarity = 1 / (1 + distance)
            similarity = 1.0 / (1.0 + float(dist))

            # Apply threshold if specified
            if threshold is not None and similarity < threshold:
                continue

            results.append({
                "id": self._metadata[idx].get("_id", idx),
                "score": similarity,
                "distance": float(dist),
                "metadata": self._metadata[idx],
            })

        return results

    async def delete(self, ids: list[int]) -> int:
        """Delete vectors by ID.

        Note: FAISS IndexFlatL2 doesn't support efficient deletion.
        This method removes metadata but doesn't compact the index.
        For production use, consider rebuilding the index periodically.

        Args:
            ids: List of vector IDs to delete.

        Returns:
            Number of vectors marked for deletion.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._delete_sync, ids)

    def _delete_sync(self, ids: list[int]) -> int:
        """Synchronous delete (runs in thread pool)."""
        id_set = set(ids)
        original_count = len(self._metadata)
        self._metadata = [m for m in self._metadata if m.get("_id") not in id_set]
        deleted = original_count - len(self._metadata)
        logger.debug("Marked {} vectors for deletion", deleted)
        return deleted

    async def save(self):
        """Persist the index and metadata to disk."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._save_sync)

    def _save_sync(self):
        """Synchronous save (runs in thread pool)."""
        import faiss

        if self._index is None:
            return

        self.store_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._index, str(self.index_file))

        # Save metadata
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": self._metadata,
                    "id_counter": self._id_counter,
                    "dimensions": self.dimensions,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.debug("Saved FAISS index: {} vectors", self._index.ntotal)

    async def clear(self):
        """Clear all vectors and metadata."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._clear_sync)

    def _clear_sync(self):
        """Synchronous clear (runs in thread pool)."""
        import faiss

        self._index = faiss.IndexFlatL2(self.dimensions)
        self._metadata = []
        self._id_counter = 0
        logger.info("Cleared FAISS index")

    async def list_documents(self) -> list[dict[str, Any]]:
        """List unique documents in the store.

        Returns:
            List of dicts with document info (source, chunk_count).
        """
        docs: dict[str, dict[str, Any]] = {}
        for meta in self._metadata:
            source = meta.get("source", "unknown")
            if source not in docs:
                docs[source] = {
                    "source": source,
                    "chunk_count": 0,
                    "first_chunk": meta,
                }
            docs[source]["chunk_count"] += 1

        return list(docs.values())
