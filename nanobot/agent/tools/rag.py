"""RAG knowledge base tool for document ingestion and query."""

from __future__ import annotations

from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import IntegerSchema, StringSchema, tool_parameters_schema


@tool_parameters(tool_parameters_schema(
    action=StringSchema(
        "Action to perform",
        enum=["ingest", "query", "list", "delete", "clear"],
    ),
    path=StringSchema("Document file path (for ingest/delete)"),
    query=StringSchema("Query text (for query action)"),
    top_k=IntegerSchema(5, description="Number of results to return (for query action)"),
))
class RAGTool(Tool):
    """RAG knowledge base tool for document ingestion and retrieval.

    Manages a local knowledge base powered by:
    - FAISS vector storage
    - BAAI/bge-m3 embeddings
    - BAAI/bge-reranker-v2-m3 reranking

    Actions:
    - ingest: Add a document to the knowledge base
    - query: Search the knowledge base for relevant content
    - list: List all documents in the knowledge base
    - delete: Remove a document from the knowledge base
    - clear: Clear the entire knowledge base

    Supports: PDF, TXT, MD, and other text formats.
    PDF parsing uses MinerU when available for better quality.
    """

    name = "rag"
    description = (
        "Manage RAG knowledge base: ingest documents, query for relevant content, "
        "list/delete/clear documents. Useful for searching through uploaded papers, "
        "documents, and notes."
    )
    _scopes = {"core", "subagent"}

    def __init__(self, pipeline: Any | None = None):
        self._pipeline = pipeline

    @classmethod
    def enabled(cls, ctx: Any) -> bool:
        """Only enable if RAG is configured."""
        return ctx.config.rag.enable

    @classmethod
    def create(cls, ctx: Any) -> "RAGTool":
        """Create tool instance with pipeline from context."""
        return cls(pipeline=ctx.rag_pipeline)

    @property
    def read_only(self) -> bool:
        """Query and list are read-only; others are not."""
        return False

    async def execute(
        self,
        action: str,
        path: str | None = None,
        query: str | None = None,
        top_k: int = 5,
        **kwargs: Any,
    ) -> str:
        """Execute RAG action.

        Args:
            action: Action to perform (ingest/query/list/delete/clear).
            path: Document path for ingest/delete.
            query: Query text for query action.
            top_k: Number of results for query.

        Returns:
            Result string for the agent.
        """
        if not self._pipeline:
            return "Error: RAG pipeline not initialized. Check RAG configuration."

        try:
            if action == "ingest":
                return await self._ingest(path)
            elif action == "query":
                return await self._query(query, top_k)
            elif action == "list":
                return await self._list()
            elif action == "delete":
                return await self._delete(path)
            elif action == "clear":
                return await self._clear()
            else:
                return f"Error: Unknown RAG action '{action}'"
        except Exception as e:
            logger.error("RAG tool error: {}", e)
            return f"Error: {e}"

    async def _ingest(self, path: str | None) -> str:
        """Ingest a document into the knowledge base."""
        if not path:
            return "Error: Document path is required for ingest action."

        result = await self._pipeline.ingest(path)

        if result.success:
            return (
                f"Successfully ingested document: {result.source}\n"
                f"Created {result.chunk_count} chunks"
            )
        else:
            return f"Failed to ingest document: {result.error}"

    async def _query(self, query: str | None, top_k: int) -> str:
        """Query the knowledge base."""
        if not query:
            return "Error: Query text is required for query action."

        result = await self._pipeline.query(query, top_k=top_k)

        if not result.has_results:
            return "No relevant content found in the knowledge base."

        # Format results for the agent
        lines = [f"Found {len(result.sources)} relevant chunks:\n"]

        for i, source in enumerate(result.sources, 1):
            score = source.get("score", 0)
            text = source.get("text", "")
            source_file = source.get("source", "unknown")

            lines.append(f"--- Result {i} (score: {score:.2f}) ---")
            lines.append(f"Source: {source_file}")
            lines.append(text)
            lines.append("")

        return "\n".join(lines)

    async def _list(self) -> str:
        """List all documents in the knowledge base."""
        documents = await self._pipeline.list_documents()

        if not documents:
            return "Knowledge base is empty. Use 'ingest' to add documents."

        lines = [f"Documents in knowledge base ({len(documents)}):\n"]

        for doc in documents:
            source = doc.get("source", "unknown")
            chunk_count = doc.get("chunk_count", 0)
            lines.append(f"- {source} ({chunk_count} chunks)")

        return "\n".join(lines)

    async def _delete(self, path: str | None) -> str:
        """Delete a document from the knowledge base."""
        if not path:
            return "Error: Document path is required for delete action."

        count = await self._pipeline.delete_document(path)

        if count > 0:
            return f"Deleted {count} chunks from document: {path}"
        else:
            return f"Document not found in knowledge base: {path}"

    async def _clear(self) -> str:
        """Clear the entire knowledge base."""
        await self._pipeline.clear()
        return "Knowledge base cleared successfully."
