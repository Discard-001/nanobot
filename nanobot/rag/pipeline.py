"""RAG Pipeline - end-to-end document ingestion and query.

Combines:
- Document parsing (MinerU or pypdf)
- Text chunking
- Embedding generation
- Vector storage (FAISS)
- Reranking
- Context assembly

Usage:
    pipeline = RAGPipeline(config)
    await pipeline.initialize()

    # Ingest a document
    result = await pipeline.ingest("paper.pdf")

    # Query the knowledge base
    answer = await pipeline.query("What is the main contribution?")
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.rag.chunker import Chunk, TextChunker
from nanobot.rag.embedding import EmbeddingService
from nanobot.rag.reranker import RerankerService
from nanobot.rag.vector_store import FAISSVectorStore


@dataclass
class IngestResult:
    """Result of document ingestion."""

    source: str  # Document path
    chunk_count: int  # Number of chunks created
    success: bool  # Whether ingestion succeeded
    error: str | None = None  # Error message if failed


@dataclass
class RAGResult:
    """Result of RAG query."""

    context: str  # Assembled context for LLM
    sources: list[dict[str, Any]]  # Source documents with scores
    query: str  # Original query

    @property
    def has_results(self) -> bool:
        """Whether any relevant results were found."""
        return len(self.sources) > 0

    def to_prompt_context(self) -> str:
        """Format as context for LLM prompt."""
        if not self.has_results:
            return ""

        lines = ["# Retrieved Context\n"]
        for i, source in enumerate(self.sources, 1):
            score = source.get("score", 0)
            text = source.get("text", "")
            source_file = source.get("source", "unknown")
            lines.append(f"## Source {i} (score: {score:.2f})")
            lines.append(f"From: {source_file}\n")
            lines.append(text)
            lines.append("")

        return "\n".join(lines)


class RAGPipeline:
    """End-to-end RAG pipeline for document ingestion and query.

    Orchestrates:
    1. Document parsing (PDF → text)
    2. Text chunking (text → chunks)
    3. Embedding (chunks → vectors)
    4. Storage (vectors → FAISS)
    5. Retrieval (query → similar chunks)
    6. Reranking (chunks → ranked results)
    7. Context assembly (ranked results → prompt context)
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        reranker_service: RerankerService,
        vector_store: FAISSVectorStore,
        chunker: TextChunker | None = None,
        top_k: int = 10,
        similarity_threshold: float = 0.3,
        max_context_length: int = 4096,
        mineru_parser: Any | None = None,
    ):
        """Initialize RAG pipeline.

        Args:
            embedding_service: Embedding generation service.
            reranker_service: Reranking service.
            vector_store: Vector storage backend.
            chunker: Text chunker (default: TextChunker with 512/64).
            top_k: Number of results to retrieve.
            similarity_threshold: Minimum similarity score.
            max_context_length: Max characters in assembled context.
            mineru_parser: Optional MinerU parser for PDFs.
        """
        self.embedding_service = embedding_service
        self.reranker_service = reranker_service
        self.vector_store = vector_store
        self.chunker = chunker or TextChunker()
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.max_context_length = max_context_length
        self.mineru_parser = mineru_parser

    async def initialize(self):
        """Initialize the pipeline (load vector store)."""
        await self.vector_store.initialize()
        logger.info(
            "RAG pipeline initialized: {} vectors in store",
            self.vector_store.count,
        )

    async def ingest(
        self,
        file_path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        """Ingest a document into the knowledge base.

        Supports:
        - PDF files (parsed with MinerU or pypdf)
        - Text files (.txt, .md)
        - Other text formats

        Args:
            file_path: Path to document file.
            metadata: Additional metadata to store with chunks.

        Returns:
            IngestResult with chunk count and status.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return IngestResult(
                source=str(file_path),
                chunk_count=0,
                success=False,
                error=f"File not found: {file_path}",
            )

        meta = metadata or {}
        meta["source"] = str(file_path)
        meta["filename"] = file_path.name

        try:
            # Extract text based on file type
            text = await self._extract_text(file_path)
            if not text.strip():
                return IngestResult(
                    source=str(file_path),
                    chunk_count=0,
                    success=False,
                    error="No text content extracted",
                )

            # Chunk the text
            chunks = self.chunker.chunk(text, meta)
            if not chunks:
                return IngestResult(
                    source=str(file_path),
                    chunk_count=0,
                    success=False,
                    error="No chunks created",
                )

            # Generate embeddings
            texts = [c.text for c in chunks]
            vectors = await self.embedding_service.embed_texts(texts)

            # Store in FAISS
            metadata_list = [c.to_dict() for c in chunks]
            await self.vector_store.add(vectors, metadata_list)
            await self.vector_store.save()

            logger.info(
                "Ingested {}: {} chunks",
                file_path.name,
                len(chunks),
            )

            return IngestResult(
                source=str(file_path),
                chunk_count=len(chunks),
                success=True,
            )

        except Exception as e:
            logger.error("Ingestion failed for {}: {}", file_path.name, e)
            return IngestResult(
                source=str(file_path),
                chunk_count=0,
                success=False,
                error=str(e),
            )

    async def query(
        self,
        question: str,
        top_k: int | None = None,
    ) -> RAGResult:
        """Query the knowledge base.

        Args:
            question: Query string.
            top_k: Override default top_k.

        Returns:
            RAGResult with context and sources.
        """
        k = top_k or self.top_k

        # Generate query embedding
        query_vector = await self.embedding_service.embed_query(question)

        # Search FAISS
        candidates = await self.vector_store.search(
            query_vector,
            top_k=k * 2,  # Get extra candidates for reranking
            threshold=self.similarity_threshold,
        )

        if not candidates:
            return RAGResult(
                context="",
                sources=[],
                query=question,
            )

        # Rerank if enabled
        if self.reranker_service and len(candidates) > 1:
            texts = [c["metadata"].get("text", "") for c in candidates]
            reranked = await self.reranker_service.rerank(
                question, texts, top_k=k
            )

            # Map back to candidates
            results = []
            for item in reranked:
                idx = item["index"]
                if idx < len(candidates):
                    candidate = candidates[idx]
                    candidate["score"] = item["score"]
                    results.append(candidate)
        else:
            results = candidates[:k]

        # Assemble context
        context_parts = []
        sources = []
        total_length = 0

        for result in results:
            text = result["metadata"].get("text", "")
            if total_length + len(text) > self.max_context_length:
                break

            context_parts.append(text)
            sources.append({
                "text": text,
                "score": result["score"],
                "source": result["metadata"].get("source", "unknown"),
                "chunk_index": result["metadata"].get("index", 0),
            })
            total_length += len(text)

        context = "\n\n---\n\n".join(context_parts)

        return RAGResult(
            context=context,
            sources=sources,
            query=question,
        )

    async def _extract_text(self, file_path: Path) -> str:
        """Extract text from a document file.

        Args:
            file_path: Path to document.

        Returns:
            Extracted text content.
        """
        suffix = file_path.suffix.lower()

        # PDF files
        if suffix == ".pdf":
            return await self._extract_pdf(file_path)

        # Text files
        if suffix in (".txt", ".md", ".markdown", ".rst", ".tex"):
            return file_path.read_text(encoding="utf-8", errors="replace")

        # Try to read as text
        try:
            return file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            logger.warning("Unsupported file format: {}", suffix)
            return ""

    async def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from PDF.

        Uses MinerU if available, falls back to pypdf.

        Args:
            file_path: Path to PDF file.

        Returns:
            Extracted text content.
        """
        # Try MinerU first
        if self.mineru_parser:
            try:
                result = await self.mineru_parser.parse(file_path)
                return result.markdown_content
            except Exception as e:
                logger.warning("MinerU parsing failed, falling back to pypdf: {}", e)

        # Fallback to pypdf
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(file_path))
            text_parts = []

            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"--- Page {i + 1} ---\n{page_text}")

            return "\n\n".join(text_parts)

        except Exception as e:
            logger.error("PDF extraction failed: {}", e)
            return ""

    async def delete_document(self, source: str) -> int:
        """Delete all chunks from a document.

        Args:
            source: Document source path.

        Returns:
            Number of chunks deleted.
        """
        # Find all chunk IDs for this source
        ids_to_delete = []
        for meta in self.vector_store._metadata:
            if meta.get("source") == source:
                ids_to_delete.append(meta.get("_id"))

        if not ids_to_delete:
            return 0

        count = await self.vector_store.delete(ids_to_delete)
        await self.vector_store.save()
        return count

    async def list_documents(self) -> list[dict[str, Any]]:
        """List all documents in the knowledge base.

        Returns:
            List of document info dicts.
        """
        return await self.vector_store.list_documents()

    async def clear(self):
        """Clear the entire knowledge base."""
        await self.vector_store.clear()
        await self.vector_store.save()
        logger.info("Knowledge base cleared")
