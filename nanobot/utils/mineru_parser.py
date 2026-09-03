"""MinerU structured PDF parser for academic papers.

Uses the MinerU cloud API (mineru.net) for high-quality PDF parsing with:
- Formula recognition (LaTeX)
- Table recognition (Markdown)
- Layout analysis
- Reading order detection

Requires an API token from https://mineru.net/apiManage/token.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


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


def _count_pages(content: str) -> int:
    """Count pages from markdown content."""
    # MinerU typically marks pages with "--- Page N ---" or similar
    import re
    pages = re.findall(r"---\s*Page\s+(\d+)\s*---", content)
    if pages:
        return max(int(p) for p in pages)
    # Fallback: estimate from content length
    return max(1, len(content) // 3000)


class MinerUApiParser:
    """MinerU cloud API parser (mineru.net Precision Extract API).

    Parses PDFs (and doc/ppt/images) via the hosted MinerU service, keeping
    the same ``parse() -> ParsedDocument`` interface as the local
    :class:`MinerUParser` so the RAG pipeline can use either transparently.

    Requires an API token from https://mineru.net/apiManage/token.
    Flow (batch single-file upload):
        1. POST /file-urls/batch  -> signed upload URL + batch_id
        2. PUT file bytes to the signed URL (no Content-Type header)
        3. Poll GET /extract-results/batch/{batch_id} until state == done
        4. Download full_zip_url, extract the largest *.md
    """

    _DEFAULT_BASE = "https://mineru.net/api/v4"
    _POLL_INTERVAL = 5.0  # seconds between status checks
    _POLL_TIMEOUT = 600.0  # overall parsing budget in seconds

    def __init__(
        self,
        api_token: str,
        model_version: str = "vlm",
        language: str = "",  # empty = let MinerU auto-detect; hint like "ch"/"en"
        formula_recognition: bool = True,
        table_recognition: bool = True,
        api_base: str | None = None,
        poll_interval: float | None = None,
        poll_timeout: float | None = None,
    ):
        if not api_token:
            raise ValueError("MinerU API token is required (get one at mineru.net/apiManage/token)")
        self.api_token = api_token
        self.model_version = model_version
        self.language = language
        self.formula_recognition = formula_recognition
        self.table_recognition = table_recognition
        self.api_base = (api_base or self._DEFAULT_BASE).rstrip("/")
        self.poll_interval = poll_interval or self._POLL_INTERVAL
        self.poll_timeout = poll_timeout or self._POLL_TIMEOUT

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}",
        }

    async def parse(self, pdf_path: str | Path) -> ParsedDocument:
        """Parse a local file via the MinerU cloud API."""
        import io
        import zipfile

        import httpx

        file_path = Path(pdf_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        async with httpx.AsyncClient(timeout=120.0) as client:
            # 1. Request a signed upload URL (batch API accepts a single file too)
            payload: dict[str, Any] = {
                "files": [{"name": file_path.name, "data_id": file_path.stem}],
                "model_version": self.model_version,
                "enable_formula": self.formula_recognition,
                "enable_table": self.table_recognition,
            }
            if self.language:
                # Optional hint; omit to let MinerU auto-detect the document language
                payload["language"] = self.language
            create_resp = await client.post(
                f"{self.api_base}/file-urls/batch",
                headers=self._headers,
                json=payload,
            )
            create_data = self._unwrap(create_resp)
            batch_id = create_data.get("batch_id")
            file_urls = create_data.get("file_urls") or []
            if not batch_id or not file_urls:
                raise RuntimeError(f"MinerU batch create missing batch_id/file_urls: {create_data}")

            # 2. Upload the file bytes (PUT, no Content-Type per docs)
            put_resp = await client.put(file_urls[0], content=file_path.read_bytes())
            if put_resp.status_code not in (200, 201):
                raise RuntimeError(f"MinerU file upload failed: HTTP {put_resp.status_code}")

            # 3. Poll for completion
            deadline = time.monotonic() + self.poll_timeout
            while True:
                await asyncio.sleep(self.poll_interval)
                poll_resp = await client.get(
                    f"{self.api_base}/extract-results/batch/{batch_id}",
                    headers=self._headers,
                )
                poll_data = self._unwrap(poll_resp)
                results = poll_data.get("extract_result") or []
                if not results:
                    raise RuntimeError(f"MinerU batch results empty: {poll_data}")
                entry = results[0]
                state = entry.get("state", "")
                if state == "done":
                    zip_url = entry.get("full_zip_url")
                    if not zip_url:
                        raise RuntimeError(f"MinerU done but no full_zip_url: {entry}")
                    break
                if state == "failed":
                    raise RuntimeError(f"MinerU parsing failed: {entry.get('err_msg', 'unknown')}")
                if time.monotonic() > deadline:
                    raise RuntimeError(f"MinerU parsing timed out after {self.poll_timeout}s (state={state})")

            # 4. Download and extract the markdown from the result zip
            zip_resp = await client.get(zip_url)
            zip_resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
                md_candidates = [n for n in zf.namelist() if n.lower().endswith(".md")]
                if not md_candidates:
                    raise RuntimeError(f"No markdown file in MinerU result zip: {zf.namelist()}")
                # Prefer full.md / *.md at zip root; fall back to the largest md
                best = next(
                    (n for n in md_candidates if Path(n).name == "full.md"),
                    max(md_candidates, key=lambda n: zf.getinfo(n).file_size),
                )
                md_content = zf.read(best).decode("utf-8", errors="replace")

        return ParsedDocument(
            markdown_content=md_content,
            page_count=_count_pages(md_content),
            formula_count=md_content.count("$$") // 2,
            table_count=md_content.count("|---"),
            metadata={
                "source": str(file_path),
                "parser": "mineru-api",
                "model_version": self.model_version,
            },
        )

    @staticmethod
    def _unwrap(resp: httpx.Response) -> dict[str, Any]:
        """Validate the MinerU response envelope and return ``data``.

        Business endpoints return ``{"code": 0, "data": {...}, "msg": "ok"}``;
        the auth/gateway layer returns ``{"success": false, "msgCode": ...}``
        on failure — both shapes are handled here.
        """
        resp.raise_for_status()
        body = resp.json()
        if body.get("success") is False:
            raise RuntimeError(
                f"MinerU API auth error: {body.get('msgCode')} {body.get('msg')}"
            )
        if body.get("code") not in (0, None):
            raise RuntimeError(f"MinerU API error {body.get('code')}: {body.get('msg')}")
        return body.get("data") or {}
