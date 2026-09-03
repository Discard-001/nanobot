"""Tests for the bare-PDF auto flow: MinerU parse → archive markdown →
RAG ingest (header-aware chunks) → summarization prompt."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import ChannelsConfig
from nanobot.providers.base import LLMResponse


def _make_loop(tmp_path: Path) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="ok"))
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        channels_config=ChannelsConfig(),
    )
    # Minimal mineru + rag stubs (normally wired during tool registration)
    loop._mineru_parser = MagicMock()
    loop._mineru_parser.parse = AsyncMock(
        return_value=SimpleNamespace(
            markdown_content="# Paper Title\n\n## Method\n\nWe use X.\n\n$$E=mc^2$$"
        )
    )
    loop._rag_pipeline = MagicMock()
    loop._rag_pipeline.ingest = AsyncMock(
        return_value=SimpleNamespace(success=True, chunk_count=7, error=None)
    )
    return loop


async def test_bare_pdf_parses_archives_ingests_and_prompts_summary(tmp_path: Path):
    loop = _make_loop(tmp_path)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    content, media = await loop._prepare_message_media("", [str(pdf)])

    # Parser invoked once for the PDF
    loop._mineru_parser.parse.assert_awaited_once()
    # Markdown archived under workspace/knowledge/
    archive = tmp_path / "knowledge" / "paper.md"
    assert archive.exists()
    assert "# Paper Title" in archive.read_text(encoding="utf-8")
    # Ingested from the archived markdown
    loop._rag_pipeline.ingest.assert_awaited_once_with(archive)
    # Content carries the summary instruction + markdown + ingest status
    assert "结构化总结" in content
    assert "已解析并存入知识库（7 个分块）" in content
    assert "$$E=mc^2$$" in content
    # PDF is consumed (not passed downstream as media)
    assert media == []


async def test_bare_pdf_ingest_failure_still_summarizes(tmp_path: Path):
    loop = _make_loop(tmp_path)
    loop._rag_pipeline.ingest = AsyncMock(
        return_value=SimpleNamespace(success=False, chunk_count=0, error="embedding api down")
    )
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    content, media = await loop._prepare_message_media("", [str(pdf)])

    assert "入库失败：embedding api down" in content
    assert "# Paper Title" in content  # markdown still available for summary
    assert media == []


async def test_bare_pdf_parse_failure_degrades_to_plain_extraction(tmp_path: Path):
    loop = _make_loop(tmp_path)
    loop._mineru_parser.parse = AsyncMock(side_effect=RuntimeError("mineru quota exceeded"))
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    content, media = await loop._prepare_message_media("", [str(pdf)])

    # No ingest happened, no summary instruction, turn still completes
    loop._rag_pipeline.ingest.assert_not_awaited()
    assert "结构化总结" not in content
    assert media == []


async def test_pdf_with_text_uses_normal_path(tmp_path: Path):
    """PDF accompanied by user text skips auto-ingest (normal extraction)."""
    loop = _make_loop(tmp_path)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    content, media = await loop._prepare_message_media("总结这篇", [str(pdf)])

    loop._rag_pipeline.ingest.assert_not_awaited()
    assert content.startswith("总结这篇")
    assert media == []


async def test_bare_pdf_without_rag_uses_normal_path(tmp_path: Path):
    """Bare PDF but RAG disabled: plain extraction, no archive/ingest."""
    loop = _make_loop(tmp_path)
    loop._rag_pipeline = None
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    content, _ = await loop._prepare_message_media("", [str(pdf)])

    loop._mineru_parser.parse.assert_awaited_once()  # still parsed for context
    assert "结构化总结" not in content
    assert not (tmp_path / "knowledge").exists()


async def test_state_restore_triggers_bare_pdf_flow(tmp_path: Path):
    """End-to-end through _state_restore: bare PDF message becomes a
    summarization turn with the PDF consumed."""
    loop = _make_loop(tmp_path)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    from nanobot.agent.loop import TurnContext, TurnState

    ctx = TurnContext(
        msg=InboundMessage(
            channel="cli",
            sender_id="u",
            chat_id="c",
            content="",
            media=[str(pdf)],
        ),
        session_key="cli:c",
        state=TurnState.RESTORE,
        turn_id="turn-1",
    )

    assert await loop._state_restore(ctx) == "ok"
    assert "结构化总结" in ctx.msg.content
    assert ctx.msg.media == []
    assert (tmp_path / "knowledge" / "doc.md").exists()
