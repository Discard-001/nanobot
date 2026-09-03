"""Tests for MinerUApiParser (mineru.net Precision Extract API) and
the async document extraction path routing PDFs through it."""

import io
import zipfile
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from nanobot.config.schema import MinerUConfig
from nanobot.utils.document import extract_documents_async
from nanobot.utils.mineru_parser import MinerUApiParser, ParsedDocument


def _fake_zip_response(md_content: str) -> httpx.Response:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("full.md", md_content)
        zf.writestr("images/img_0.jpg", "fake")
    return httpx.Response(
        200,
        content=buf.getvalue(),
        request=httpx.Request("GET", "https://cdn.example.com/result.zip"),
    )


def _json_response(payload: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload,
        request=httpx.Request("POST", "https://mineru.net/api/v4/x"),
    )


class TestMinerUApiParserInit:
    def test_requires_token(self):
        with pytest.raises(ValueError, match="token"):
            MinerUApiParser(api_token="")

    def test_default_base_and_headers(self):
        parser = MinerUApiParser(api_token="tok123")
        assert parser.api_base == "https://mineru.net/api/v4"
        assert parser._headers["Authorization"] == "Bearer tok123"
        assert parser.model_version == "vlm"


class TestUnwrap:
    def test_business_ok(self):
        resp = _json_response({"code": 0, "data": {"k": "v"}, "msg": "ok"})
        assert MinerUApiParser._unwrap(resp) == {"k": "v"}

    def test_business_error(self):
        resp = _json_response({"code": 1004, "data": None, "msg": "quota exceeded"})
        with pytest.raises(RuntimeError, match="1004"):
            MinerUApiParser._unwrap(resp)

    def test_auth_error_shape(self):
        resp = _json_response({"success": False, "msgCode": "A0202", "msg": "auth failed"})
        with pytest.raises(RuntimeError, match="A0202"):
            MinerUApiParser._unwrap(resp)


class TestParseFlow:
    def _make_parser(self, **kw) -> MinerUApiParser:
        return MinerUApiParser(api_token="tok", poll_interval=0, **kw)

    async def test_full_flow_returns_markdown(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        parser = self._make_parser()

        create_resp = _json_response(
            {"code": 0, "data": {"batch_id": "b1", "file_urls": ["https://oss/upload"]}}
        )
        put_resp = httpx.Response(
            200, request=httpx.Request("PUT", "https://oss/upload")
        )
        poll_running = _json_response(
            {"code": 0, "data": {"extract_result": [{"state": "running"}]}}
        )
        poll_done = _json_response(
            {"code": 0, "data": {"extract_result": [
                {"state": "done", "full_zip_url": "https://cdn/result.zip"}
            ]}}
        )
        zip_resp = _fake_zip_response("# Paper\n\n$$E=mc^2$$\n\n| a | b |\n|---|---|\n")

        client = MagicMock()
        client.post = AsyncMock(return_value=create_resp)
        client.put = AsyncMock(return_value=put_resp)
        client.get = AsyncMock(side_effect=[poll_running, poll_done, zip_resp])

        # Patch httpx.AsyncClient to return our fake client
        original_async_client = httpx.AsyncClient
        httpx.AsyncClient = lambda **kw: _AsyncClientStub(client)
        try:
            result = await parser.parse(pdf)
        finally:
            httpx.AsyncClient = original_async_client

        assert result.markdown_content.startswith("# Paper")
        assert "$$E=mc^2$$" in result.markdown_content
        assert result.formula_count == 1
        assert result.table_count >= 1  # "|---|---|" separator row
        assert result.metadata["parser"] == "mineru-api"

        # Verify request payloads
        create_call = client.post.call_args
        assert create_call.args[0] == "https://mineru.net/api/v4/file-urls/batch"
        assert create_call.kwargs["json"]["model_version"] == "vlm"
        assert create_call.kwargs["json"]["enable_formula"] is True
        put_call = client.put.call_args
        assert put_call.args[0] == "https://oss/upload"

    async def test_missing_file_raises(self, tmp_path):
        parser = self._make_parser()
        with pytest.raises(FileNotFoundError):
            await parser.parse(tmp_path / "nope.pdf")

    async def test_failed_state_raises(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        parser = self._make_parser()

        create_resp = _json_response(
            {"code": 0, "data": {"batch_id": "b1", "file_urls": ["https://oss/upload"]}}
        )
        put_resp = httpx.Response(200, request=httpx.Request("PUT", "https://oss/upload"))
        poll_failed = _json_response(
            {"code": 0, "data": {"extract_result": [
                {"state": "failed", "err_msg": "bad pdf"}
            ]}}
        )

        client = MagicMock()
        client.post = AsyncMock(return_value=create_resp)
        client.put = AsyncMock(return_value=put_resp)
        client.get = AsyncMock(return_value=poll_failed)

        original_async_client = httpx.AsyncClient
        httpx.AsyncClient = lambda **kw: _AsyncClientStub(client)
        try:
            with pytest.raises(RuntimeError, match="bad pdf"):
                await parser.parse(pdf)
        finally:
            httpx.AsyncClient = original_async_client


class _AsyncClientStub:
    """Minimal async context manager yielding the mocked client."""

    def __init__(self, client):
        self._client = client

    async def __aenter__(self):
        return self._client

    async def __aexit__(self, *exc):
        return False


class TestExtractDocumentsAsync:
    async def test_pdf_routed_to_parser(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        img = tmp_path / "pic.png"
        img.write_bytes(b"\x89PNG fake")

        parser = MagicMock()
        parser.parse = AsyncMock(
            return_value=ParsedDocument(markdown_content="# Structured\n$$x^2$$")
        )

        text, images = await extract_documents_async("hello", [str(pdf), str(img)],
                                                     pdf_parser=parser)

        parser.parse.assert_awaited_once()
        assert "# Structured" in text
        assert "$$x^2$$" in text
        assert "[File: doc.pdf]" in text
        assert images == [str(img)]

    async def test_parser_failure_falls_back_to_pypdf(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        parser = MagicMock()
        parser.parse = AsyncMock(side_effect=RuntimeError("api down"))

        text, images = await extract_documents_async("hello", [str(pdf)], pdf_parser=parser)

        # Falls back to plain extraction (pypdf on fake bytes yields empty ->
        # doc text dropped, but no exception raised)
        assert images == []
        assert "hello" in text

    async def test_non_pdf_not_routed(self, tmp_path):
        docx = tmp_path / "notes.docx"
        docx.write_bytes(b"fake docx")

        parser = MagicMock()
        parser.parse = AsyncMock()

        await extract_documents_async("hello", [str(docx)], pdf_parser=parser)
        parser.parse.assert_not_awaited()


class TestMinerUConfig:
    def test_enable_requires_token(self):
        with pytest.raises(ValueError, match="apiToken"):
            MinerUConfig(enable=True)

    def test_with_token_ok(self):
        cfg = MinerUConfig(enable=True, apiToken="sk-test")
        assert cfg.api_token == "sk-test"
        assert cfg.model_version == "vlm"

    def test_legacy_mode_key_ignored(self):
        """Configs written before the local-mode removal may still carry a
        'mode' key; it must be ignored, not rejected."""
        cfg = MinerUConfig(enable=True, apiToken="sk-test", mode="api", device="cpu")
        assert cfg.enable is True

    def test_camelcase_alias(self):
        cfg = MinerUConfig(**{"modelVersion": "pipeline"})
        assert cfg.model_version == "pipeline"

    def test_defaults_disabled(self):
        cfg = MinerUConfig()
        assert cfg.enable is False
