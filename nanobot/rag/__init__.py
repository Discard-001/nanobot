"""RAG (Retrieval-Augmented Generation) knowledge base module.

Provides:
- Document ingestion (PDF, text)
- Text chunking
- Embedding generation (ModelScope BAAI/bge-m3 or local models)
- Vector storage (FAISS)
- Reranking (BAAI/bge-reranker-v2-m3)
- Knowledge base query pipeline

Requires: pip install nanobot-ai[rag]
"""

from nanobot.rag.pipeline import RAGPipeline
from nanobot.rag.embedding import EmbeddingService
from nanobot.rag.reranker import RerankerService
from nanobot.rag.vector_store import FAISSVectorStore
from nanobot.rag.chunker import TextChunker

__all__ = [
    "RAGPipeline",
    "EmbeddingService",
    "RerankerService",
    "FAISSVectorStore",
    "TextChunker",
]
