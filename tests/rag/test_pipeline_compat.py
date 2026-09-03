"""Regression tests for RAGPipeline construction and lazy initialization.

Covers a latent bug: AgentLoop passed chunk_size/chunk_overlap kwargs that
RAGPipeline.__init__ never accepted, so enabling RAG always crashed at startup.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from nanobot.rag.pipeline import RAGPipeline


def _make_pipeline(**kwargs) -> RAGPipeline:
    store = MagicMock()
    store.count = 0
    store.initialize = AsyncMock()
    return RAGPipeline(
        embedding_service=MagicMock(),
        reranker_service=MagicMock(),
        vector_store=store,
        **kwargs,
    )


def test_accepts_chunk_size_and_overlap_kwargs():
    pipeline = _make_pipeline(chunk_size=256, chunk_overlap=32)
    assert pipeline.chunker.chunk_size == 256
    assert pipeline.chunker.chunk_overlap == 32


def test_default_chunker_matches_config_defaults():
    pipeline = _make_pipeline()
    assert pipeline.chunker.chunk_size == 512
    assert pipeline.chunker.chunk_overlap == 64


def test_explicit_chunker_overrides_sizes():
    from nanobot.rag.chunker import TextChunker

    chunker = TextChunker(chunk_size=99, chunk_overlap=9)
    pipeline = _make_pipeline(chunker=chunker, chunk_size=512)
    assert pipeline.chunker is chunker


async def test_ingest_lazy_initializes_store(tmp_path: Path):
    pipeline = _make_pipeline()
    doc = tmp_path / "note.md"
    doc.write_text("# Section\n\nSome text", encoding="utf-8")

    pipeline.embedding_service.embed_texts = AsyncMock(return_value=[[0.0] * 8])
    pipeline.vector_store.add = AsyncMock(return_value=[1])
    pipeline.vector_store.save = AsyncMock()

    result = await pipeline.ingest(doc)

    assert result.success, result.error
    pipeline.vector_store.initialize.assert_awaited_once()
    # Second call must not re-initialize
    await pipeline.ingest(doc)
    assert pipeline.vector_store.initialize.await_count == 1


async def test_header_aware_chunking_on_ingest(tmp_path: Path):
    """Ingested chunks carry the section header in metadata."""
    from nanobot.rag.chunker import TextChunker

    pipeline = _make_pipeline(chunker=TextChunker(chunk_size=64, chunk_overlap=8))
    doc = tmp_path / "paper.md"
    doc.write_text(
        "# Title\n\nintro text\n\n## Methods\n\n" + ("detail " * 30),
        encoding="utf-8",
    )

    captured: dict = {}

    async def fake_embed(texts):
        captured["texts"] = texts
        return [[0.0] * 8 for _ in texts]

    pipeline.embedding_service.embed_texts = fake_embed
    pipeline.vector_store.add = AsyncMock(side_effect=lambda vectors, meta: [1] * len(vectors))
    pipeline.vector_store.save = AsyncMock()

    result = await pipeline.ingest(doc)

    assert result.success, result.error
    assert result.chunk_count >= 2
    # vector_store.add received chunk metadata containing section headers
    add_kwargs = pipeline.vector_store.add.call_args.args[1]
    headers = {m["metadata"].get("header") for m in add_kwargs}
    assert "## Methods" in headers
