"""Text chunker for RAG knowledge base.

Splits documents into chunks suitable for embedding and retrieval.

Features:
- Sentence-aware splitting (won't break mid-sentence)
- Configurable chunk size and overlap
- Metadata preservation across chunks
- Support for Markdown structure (headers, lists)

Usage:
    chunker = TextChunker(chunk_size=512, overlap=64)
    chunks = chunker.chunk("Long text...", metadata={"source": "paper.pdf"})
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A text chunk with metadata."""

    text: str  # Chunk content
    index: int  # Chunk index within document
    metadata: dict[str, Any] = field(default_factory=dict)  # Inherited metadata
    start_char: int = 0  # Start character position in original text
    end_char: int = 0  # End character position in original text

    @property
    def char_count(self) -> int:
        """Number of characters in chunk."""
        return len(self.text)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "text": self.text,
            "index": self.index,
            "metadata": self.metadata,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


class TextChunker:
    """Text chunker with sentence-aware splitting.

    Splits text into chunks while respecting:
    - Sentence boundaries
    - Paragraph boundaries
    - Markdown headers
    - Configurable chunk size and overlap

    The chunker tries to create chunks that are:
    - Close to chunk_size characters
    - Never larger than chunk_size + overlap
    - Split at natural boundaries (sentences, paragraphs)
    """

    # Sentence boundary patterns
    _SENTENCE_END = re.compile(r'[.!?。！？]\s+')
    _PARAGRAPH_END = re.compile(r'\n\s*\n')
    _HEADER_PATTERN = re.compile(r'^#{1,6}\s+', re.MULTILINE)

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        """Initialize chunker.

        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between consecutive chunks in characters.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Split text into chunks.

        Args:
            text: Text to chunk.
            metadata: Metadata to attach to all chunks.

        Returns:
            List of Chunk objects.
        """
        if not text or not text.strip():
            return []

        meta = metadata or {}
        text = text.strip()

        # If text is shorter than chunk_size, return as single chunk
        if len(text) <= self.chunk_size:
            return [
                Chunk(
                    text=text,
                    index=0,
                    metadata=meta,
                    start_char=0,
                    end_char=len(text),
                )
            ]

        # Split into chunks
        chunks = []
        current_pos = 0
        chunk_index = 0

        while current_pos < len(text):
            # Find chunk end position
            chunk_end = min(current_pos + self.chunk_size, len(text))

            # If not at end, try to find a natural break point
            if chunk_end < len(text):
                chunk_end = self._find_break_point(text, current_pos, chunk_end)

            # Extract chunk text
            chunk_text = text[current_pos:chunk_end].strip()

            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=chunk_index,
                        metadata=meta,
                        start_char=current_pos,
                        end_char=chunk_end,
                    )
                )
                chunk_index += 1

            # Move to next position with overlap
            next_pos = chunk_end - self.chunk_overlap
            if next_pos <= current_pos:
                # Ensure we make progress
                next_pos = chunk_end
            current_pos = next_pos

        return chunks

    def _find_break_point(
        self,
        text: str,
        start: int,
        end: int,
    ) -> int:
        """Find a natural break point near the end position.

        Looks for (in priority order):
        1. Paragraph break
        2. Sentence end
        3. Word boundary

        Args:
            text: Full text.
            start: Chunk start position.
            end: Preferred chunk end position.

        Returns:
            Adjusted end position at a natural break.
        """
        # Search window: last 20% of chunk
        search_start = max(start, end - self.chunk_size // 5)
        search_text = text[search_start:end]

        # 1. Try paragraph break (highest priority)
        para_matches = list(self._PARAGRAPH_END.finditer(search_text))
        if para_matches:
            last_para = para_matches[-1]
            return search_start + last_para.end()

        # 2. Try sentence end
        sent_matches = list(self._SENTENCE_END.finditer(search_text))
        if sent_matches:
            last_sent = sent_matches[-1]
            return search_start + last_sent.end()

        # 3. Try word boundary
        # Find last space before end
        space_pos = text.rfind(" ", search_start, end)
        if space_pos > search_start:
            return space_pos + 1

        # 4. Fall back to hard cut at end
        return end

    def chunk_markdown(
        self,
        markdown: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Split markdown text with header awareness.

        Preserves markdown structure by:
        - Keeping headers with their content
        - Splitting at header boundaries when possible
        - Maintaining list structure

        Args:
            markdown: Markdown text to chunk.
            metadata: Metadata to attach to all chunks.

        Returns:
            List of Chunk objects.
        """
        if not markdown or not markdown.strip():
            return []

        meta = metadata or {}

        # First, split by headers
        sections = self._split_by_headers(markdown)

        # Then chunk each section
        all_chunks = []
        chunk_index = 0

        for section_text, header in sections:
            # Add header to metadata
            section_meta = {**meta}
            if header:
                section_meta["header"] = header

            # Chunk the section
            section_chunks = self.chunk(section_text, section_meta)

            # Update indices
            for chunk in section_chunks:
                chunk.index = chunk_index
                chunk_index += 1

            all_chunks.extend(section_chunks)

        return all_chunks

    def _split_by_headers(self, markdown: str) -> list[tuple[str, str | None]]:
        """Split markdown by headers.

        Returns:
            List of (text, header) tuples. Header is None for text before
            the first header.
        """
        sections = []
        current_header = None
        current_text = []

        for line in markdown.split("\n"):
            if self._HEADER_PATTERN.match(line):
                # Save previous section
                if current_text:
                    sections.append(("\n".join(current_text), current_header))

                # Start new section
                current_header = line.strip()
                current_text = [line]
            else:
                current_text.append(line)

        # Save last section
        if current_text:
            sections.append(("\n".join(current_text), current_header))

        return sections
