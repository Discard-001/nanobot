"""Literature search tool for academic paper retrieval."""

from __future__ import annotations

import re
from typing import Any

import httpx
from loguru import logger

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import IntegerSchema, StringSchema, tool_parameters_schema


@tool_parameters(tool_parameters_schema(
    query=StringSchema("Search query (keywords, paper title, or research question)"),
    source=StringSchema(
        "Search source",
        enum=["semantic_scholar", "arxiv", "all"],
    ),
    limit=IntegerSchema(5, description="Maximum number of results to return"),
    year_from=IntegerSchema(description="Filter: start year (e.g., 2020)"),
    year_to=IntegerSchema(description="Filter: end year (e.g., 2024)"),
))
class LiteratureSearchTool(Tool):
    """Academic literature search tool.

    Search for academic papers across multiple sources:
    - Semantic Scholar: Comprehensive academic search with citations
    - arXiv: Preprint repository for latest research

    Returns paper metadata including:
    - Title, authors, abstract
    - Publication year, venue
    - Citation count (Semantic Scholar)
    - PDF/download links

    Useful for:
    - Finding related work
    - Exploring research topics
    - Getting paper summaries
    - Finding latest preprints
    """

    name = "literature_search"
    description = (
        "Search academic papers and literature. Supports Semantic Scholar and arXiv. "
        "Returns paper titles, authors, abstracts, and links. "
        "Useful for research, finding related work, and exploring topics."
    )
    _scopes = {"core", "subagent"}
    read_only = True

    # API endpoints
    _SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
    _ARXIV_API = "http://export.arxiv.org/api/query"

    # User agent for requests
    _USER_AGENT = "nanobot-ai/0.2.0 (literature-search)"

    async def execute(
        self,
        query: str,
        source: str = "all",
        limit: int = 5,
        year_from: int | None = None,
        year_to: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Search for academic papers.

        Args:
            query: Search query.
            source: Search source (semantic_scholar/arxiv/all).
            limit: Max results per source.
            year_from: Start year filter.
            year_to: End year filter.

        Returns:
            Formatted search results.
        """
        if not query.strip():
            return "Error: Search query cannot be empty."

        results = []

        try:
            if source in ("semantic_scholar", "all"):
                ss_results = await self._search_semantic_scholar(
                    query, limit, year_from, year_to
                )
                results.extend(ss_results)

            if source in ("arxiv", "all"):
                arxiv_results = await self._search_arxiv(query, limit)
                results.extend(arxiv_results)

            if not results:
                return f"No papers found for query: {query}"

            # Format results
            return self._format_results(results[:limit])

        except Exception as e:
            logger.error("Literature search error: {}", e)
            return f"Error searching literature: {e}"

    async def _search_semantic_scholar(
        self,
        query: str,
        limit: int,
        year_from: int | None,
        year_to: int | None,
    ) -> list[dict[str, Any]]:
        """Search Semantic Scholar API.

        Args:
            query: Search query.
            limit: Max results.
            year_from: Start year filter.
            year_to: End year filter.

        Returns:
            List of paper dicts.
        """
        url = f"{self._SEMANTIC_SCHOLAR_API}/paper/search"
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": "title,authors,abstract,year,venue,citationCount,url,openAccessPdf",
        }

        if year_from or year_to:
            year_range = []
            if year_from:
                year_range.append(str(year_from))
            else:
                year_range.append("")
            if year_to:
                year_range.append(str(year_to))
            else:
                year_range.append("")
            params["year"] = "-".join(year_range)

        headers = {"User-Agent": self._USER_AGENT}

        async with httpx.AsyncClient() as client:
            response = await client.get(
                url, params=params, headers=headers, timeout=30.0
            )
            response.raise_for_status()
            data = response.json()

        results = []
        for paper in data.get("data", []):
            authors = [a.get("name", "") for a in paper.get("authors", [])]
            pdf_url = None
            if paper.get("openAccessPdf"):
                pdf_url = paper["openAccessPdf"].get("url")

            results.append({
                "title": paper.get("title", ""),
                "authors": authors,
                "abstract": paper.get("abstract", ""),
                "year": paper.get("year"),
                "venue": paper.get("venue", ""),
                "citations": paper.get("citationCount", 0),
                "url": paper.get("url", ""),
                "pdf_url": pdf_url,
                "source": "Semantic Scholar",
            })

        logger.debug("Semantic Scholar returned {} results", len(results))
        return results

    async def _search_arxiv(
        self,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Search arXiv API.

        Args:
            query: Search query.
            limit: Max results.

        Returns:
            List of paper dicts.
        """
        # arXiv API uses Atom format
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": min(limit, 100),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        headers = {"User-Agent": self._USER_AGENT}

        async with httpx.AsyncClient() as client:
            response = await client.get(
                self._ARXIV_API, params=params, headers=headers, timeout=30.0
            )
            response.raise_for_status()
            xml_text = response.text

        return self._parse_arxiv_response(xml_text)

    def _parse_arxiv_response(self, xml_text: str) -> list[dict[str, Any]]:
        """Parse arXiv API XML response.

        Args:
            xml_text: XML response text.

        Returns:
            List of paper dicts.
        """
        results = []

        # Simple XML parsing with regex (avoids lxml dependency)
        entries = re.findall(r"<entry>(.*?)</entry>", xml_text, re.DOTALL)

        for entry in entries:
            title = self._extract_xml_field(entry, "title")
            abstract = self._extract_xml_field(entry, "summary")
            published = self._extract_xml_field(entry, "published")
            year = int(published[:4]) if published else None

            # Extract authors
            authors = re.findall(r"<name>(.*?)</name>", entry)

            # Extract links
            pdf_url = None
            links = re.findall(r'<link[^>]*href="([^"]*)"[^>]*/>', entry)
            for link in links:
                if "pdf" in link:
                    pdf_url = link
                    break

            arxiv_url = self._extract_xml_field(entry, "id")

            results.append({
                "title": title.strip().replace("\n", " "),
                "authors": authors,
                "abstract": abstract.strip().replace("\n", " "),
                "year": year,
                "venue": "arXiv",
                "citations": 0,  # arXiv doesn't provide citation counts
                "url": arxiv_url,
                "pdf_url": pdf_url,
                "source": "arXiv",
            })

        logger.debug("arXiv returned {} results", len(results))
        return results

    @staticmethod
    def _extract_xml_field(xml: str, field: str) -> str:
        """Extract field value from XML string."""
        match = re.search(f"<{field}>(.*?)</{field}>", xml, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _format_results(self, results: list[dict[str, Any]]) -> str:
        """Format search results for display.

        Args:
            results: List of paper dicts.

        Returns:
            Formatted string.
        """
        lines = [f"Found {len(results)} papers:\n"]

        for i, paper in enumerate(results, 1):
            title = paper.get("title", "Unknown")
            authors = ", ".join(paper.get("authors", [])[:3])
            if len(paper.get("authors", [])) > 3:
                authors += " et al."
            year = paper.get("year", "")
            venue = paper.get("venue", "")
            citations = paper.get("citations", 0)
            abstract = paper.get("abstract", "")
            url = paper.get("url", "")
            pdf_url = paper.get("pdf_url")
            source = paper.get("source", "")

            lines.append(f"## {i}. {title}")
            lines.append(f"**Authors:** {authors}")
            if year:
                lines.append(f"**Year:** {year}")
            if venue:
                lines.append(f"**Venue:** {venue}")
            if citations > 0:
                lines.append(f"**Citations:** {citations}")
            lines.append(f"**Source:** {source}")
            if url:
                lines.append(f"**URL:** {url}")
            if pdf_url:
                lines.append(f"**PDF:** {pdf_url}")
            if abstract:
                # Truncate abstract if too long
                if len(abstract) > 300:
                    abstract = abstract[:300] + "..."
                lines.append(f"\n**Abstract:** {abstract}")
            lines.append("")

        return "\n".join(lines)
