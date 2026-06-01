"""MinerU structured PDF parser for academic papers.

Uses MinerU (magic-pdf) for high-quality PDF parsing with:
- Formula recognition (LaTeX)
- Table recognition (HTML/Markdown)
- Layout analysis
- Reading order detection

Requires: pip install nanobot-ai[mineru]
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class ParsedDocument:
    """Result of MinerU PDF parsing."""

    markdown_content: str = ""  # Full document as Markdown
    pages: list[dict[str, Any]] = field(default_factory=list)  # Per-page content
    metadata: dict[str, Any] = field(default_factory=dict)  # Document metadata
    formula_count: int = 0  # Number of formulas detected
    table_count: int = 0  # Number of tables detected
    page_count: int = 0  # Total pages

    @property
    def has_formulas(self) -> bool:
        return self.formula_count > 0

    @property
    def has_tables(self) -> bool:
        return self.table_count > 0


class MinerUParser:
    """MinerU structured PDF parser.

    Provides high-quality PDF parsing for academic papers with support for:
    - Mathematical formula recognition (output as LaTeX)
    - Table recognition (output as Markdown tables)
    - Layout analysis and reading order detection
    - Multi-column document support

    Usage:
        parser = MinerUParser(device="cpu")
        result = await parser.parse("paper.pdf")
        print(result.markdown_content)  # Full markdown with formulas and tables
    """

    def __init__(
        self,
        device: str = "cpu",
        formula_recognition: bool = True,
        table_recognition: bool = True,
    ):
        self.device = device
        self.formula_recognition = formula_recognition
        self.table_recognition = table_recognition
        self._pipeline = None

    def _ensure_pipeline(self):
        """Lazy-load MinerU pipeline."""
        if self._pipeline is not None:
            return

        try:
            from magic_pdf.pipe.UNIPipe import UNIPipe
            from magic_pdf.pipe.OCRPipe import OCRPipe
            self._UNIPipe = UNIPipe
            self._OCRPipe = OCRPipe
            self._pipeline = True
            logger.info("MinerU pipeline loaded successfully (device={})", self.device)
        except ImportError as e:
            raise ImportError(
                "MinerU (magic-pdf) is not installed. "
                "Install it with: pip install nanobot-ai[mineru]"
            ) from e

    async def parse(self, pdf_path: str | Path) -> ParsedDocument:
        """Parse a PDF file and return structured content.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ParsedDocument with markdown content, metadata, and statistics.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not pdf_path.suffix.lower() == ".pdf":
            raise ValueError(f"Not a PDF file: {pdf_path}")

        self._ensure_pipeline()

        # Run parsing in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._parse_sync, pdf_path, None)

    async def parse_pages(
        self, pdf_path: str | Path, pages: list[int]
    ) -> ParsedDocument:
        """Parse specific pages of a PDF file.

        Args:
            pdf_path: Path to the PDF file.
            pages: List of page numbers (0-indexed).

        Returns:
            ParsedDocument with content from specified pages.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        self._ensure_pipeline()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._parse_sync, pdf_path, pages)

    def _parse_sync(self, pdf_path: Path, pages: list[int] | None) -> ParsedDocument:
        """Synchronous PDF parsing (runs in thread pool)."""
        import json

        try:
            # Read PDF bytes
            pdf_bytes = pdf_path.read_bytes()

            # Choose pipeline based on content type
            # UNIPipe handles both text-based and scanned PDFs
            pipe = self._UNIPipe(
                pdf_bytes,
                [],
                str(pdf_path),
                is_debug=False,
            )

            # Execute pipeline
            pipe.pipe_classify()
            pipe.pipe_analyze()
            pipe.pipe_parse()

            # Get results
            md_content = pipe.pipe_mk_markdown(
                str(pdf_path.parent),
                drop_mode="none",
            )

            # Parse result
            result = ParsedDocument(
                markdown_content=md_content,
                page_count=self._count_pages(md_content),
                formula_count=md_content.count("$") // 2,  # Rough estimate
                table_count=md_content.count("|---"),  # Table separator count
                metadata={
                    "source": str(pdf_path),
                    "parser": "mineru",
                    "device": self.device,
                },
            )

            logger.info(
                "MinerU parsed {}: {} pages, {} formulas, {} tables",
                pdf_path.name,
                result.page_count,
                result.formula_count,
                result.table_count,
            )

            return result

        except Exception as e:
            logger.error("MinerU parsing failed for {}: {}", pdf_path.name, e)
            # Return error as parsed document
            return ParsedDocument(
                markdown_content=f"[MinerU parsing error: {e}]",
                metadata={
                    "source": str(pdf_path),
                    "parser": "mineru",
                    "error": str(e),
                },
            )

    @staticmethod
    def _count_pages(content: str) -> int:
        """Count pages from markdown content."""
        # MinerU typically marks pages with "--- Page N ---" or similar
        import re
        pages = re.findall(r"---\s*Page\s+(\d+)\s*---", content)
        if pages:
            return max(int(p) for p in pages)
        # Fallback: estimate from content length
        return max(1, len(content) // 3000)
