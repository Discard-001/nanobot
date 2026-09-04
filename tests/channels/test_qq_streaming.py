"""Tests for QQ C2C streaming (stream_messages) and voice transcription."""
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

try:
    from nanobot.channels import qq
    QQ_AVAILABLE = getattr(qq, "QQ_AVAILABLE", False)
except ImportError:
    QQ_AVAILABLE = False

if not QQ_AVAILABLE:
    pytest.skip("QQ dependencies not installed (qq-botpy)", allow_module_level=True)

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.qq import QQChannel, QQConfig, _QQStreamBuf


def _make_channel(streaming: bool = True, **cfg) -> QQChannel:
    config = QQConfig(
        app_id="app", secret="secret", allow_from=["*"], streaming=streaming, **cfg
    )
    ch = QQChannel(config, MessageBus())
    ch._client = SimpleNamespace()
    ch._client.api = SimpleNamespace()
    ch._client.api._http = SimpleNamespace()
    ch._client.api._http.request = AsyncMock(return_value={"id": "stream_001"})
    ch._client.api.post_c2c_message = AsyncMock()
    ch._client.api.post_group_message = AsyncMock()
    ch._chat_type_cache["user1"] = "c2c"
    return ch


def _last_stream_payload(ch: QQChannel) -> dict:
    call = ch._client.api._http.request.call_args
    assert call.kwargs.get("json") is not None
    return call.kwargs["json"]


class TestStreamMessages:
    @pytest.mark.asyncio
    async def test_first_reasoning_delta_creates_stream(self):
        ch = _make_channel()
        await ch.send_reasoning_delta("user1", "分析请求", {"message_id": "om_1"})

        buf = ch._stream_bufs["om_1"]
        assert buf.stream_msg_id == "stream_001"
        assert buf.index == 0
        assert buf.reasoning_rounds == 1
        payload = _last_stream_payload(ch)
        assert payload["input_mode"] == "replace"
        assert payload["input_state"] == 1
        assert payload["index"] == 0
        assert payload["content_type"] == "markdown"
        assert payload["msg_id"] == "om_1"
        assert "stream_msg_id" not in payload  # first chunk has no id yet
        assert "💭 **执行过程**" in payload["content_raw"]
        assert "> 分析请求" in payload["content_raw"]

    @pytest.mark.asyncio
    async def test_subsequent_chunk_carries_stream_msg_id(self):
        ch = _make_channel()
        buf = _QQStreamBuf(
            msg_id="om_1", openid="user1", stream_msg_id="stream_001", index=0,
            last_edit=0.0,
        )
        buf.append_reasoning("已思考")
        ch._stream_bufs["om_1"] = buf
        await ch.send_reasoning_delta("user1", "更多", {"message_id": "om_1"})

        payload = _last_stream_payload(ch)
        assert payload["stream_msg_id"] == "stream_001"
        assert payload["index"] == 1

    @pytest.mark.asyncio
    async def test_reasoning_end_flushes_full_trace(self):
        """Burst-delivered reasoning swallowed by the throttle must be flushed."""
        ch = _make_channel()
        buf = _QQStreamBuf(
            msg_id="om_1", openid="user1", stream_msg_id="stream_001", index=1,
            last_edit=time.monotonic(),
        )
        buf.append_reasoning("完整思考内容")
        ch._stream_bufs["om_1"] = buf
        await ch.send_reasoning_end("user1", {"message_id": "om_1"})

        buf = ch._stream_bufs["om_1"]
        assert buf.reasoning_open is False
        payload = _last_stream_payload(ch)
        assert "> 完整思考内容" in payload["content_raw"]

    @pytest.mark.asyncio
    async def test_multi_round_reasoning_separator(self):
        ch = _make_channel()
        buf = _QQStreamBuf(
            msg_id="om_1", openid="user1", stream_msg_id="stream_001", index=1,
            last_edit=0.0,
        )
        buf.append_reasoning("第一轮")
        buf.close_reasoning()
        ch._stream_bufs["om_1"] = buf
        await ch.send_reasoning_delta("user1", "第二轮", {"message_id": "om_1"})

        buf = ch._stream_bufs["om_1"]
        assert buf.reasoning_rounds == 2
        payload = _last_stream_payload(ch)
        assert "第一轮" in payload["content_raw"]
        assert "第二轮" in payload["content_raw"]
        assert "···" in payload["content_raw"]

    @pytest.mark.asyncio
    async def test_interleaved_rounds_are_append_only(self):
        """Regression (QQ error 40007): rendering must never mutate the
        submitted prefix. A second reasoning round arriving after a tool
        hint must be appended after it, not inserted before."""
        ch = _make_channel()
        await ch.send_reasoning_delta("user1", "第一轮思考", {"message_id": "om_1"})
        await ch.send_reasoning_end("user1", {"message_id": "om_1"})
        await ch.send(
            OutboundMessage(
                channel="qq", chat_id="user1",
                content="exec(command='ls')",
                metadata={"_tool_hint": True, "message_id": "om_1"},
            )
        )
        await ch.send_reasoning_delta("user1", "第二轮思考", {"message_id": "om_1"})
        await ch.send_reasoning_end("user1", {"message_id": "om_1"})
        await ch.send_delta("user1", "最终答案", {"message_id": "om_1", "_stream_delta": True})
        await ch.send_delta("user1", "", {"message_id": "om_1", "_stream_end": True})

        payloads = [
            c.kwargs["json"]["content_raw"]
            for c in ch._client.api._http.request.await_args_list
        ]
        # Every push must extend the previous one (append-only prefix)
        for prev, cur in zip(payloads, payloads[1:]):
            assert cur.startswith(prev), (
                f"prefix mutated: {prev!r} -> {cur!r}"
            )
        final = payloads[-1]
        # Arrival order preserved: reasoning → tool → reasoning → answer
        assert final.index("第一轮思考") < final.index("exec")
        assert final.index("exec") < final.index("第二轮思考")
        assert final.index("第二轮思考") < final.index("最终答案")

    @pytest.mark.asyncio
    async def test_stream_end_finalizes_with_state_10(self):
        ch = _make_channel()
        buf = _QQStreamBuf(
            msg_id="om_1", openid="user1", stream_msg_id="stream_001", index=2,
            last_edit=0.0,
        )
        buf.append_reasoning("思考")
        buf.append_answer("最终回答")
        ch._stream_bufs["om_1"] = buf
        await ch.send_delta(
            "user1", "", {"message_id": "om_1", "_stream_end": True}
        )

        payload = _last_stream_payload(ch)
        assert payload["input_state"] == 10
        assert "最终回答" in payload["content_raw"]
        # Buffer cleaned up after final close
        assert "om_1" not in ch._stream_bufs

    @pytest.mark.asyncio
    async def test_failed_stream_end_falls_back_to_normal_send(self):
        """Stream broke mid-turn: the final answer must still be delivered
        via a normal message (the final send() is suppressed by _streamed)."""
        ch = _make_channel()
        buf = _QQStreamBuf(msg_id="om_1", openid="user1", stream_msg_id="s", index=2)
        buf.append_reasoning("思考")
        buf.append_answer("兜底答案")
        buf.failed = True
        ch._stream_bufs["om_1"] = buf
        await ch.send_delta(
            "user1", "", {"message_id": "om_1", "_stream_end": True}
        )

        ch._client.api.post_c2c_message.assert_awaited_once()
        call = ch._client.api.post_c2c_message.await_args.kwargs
        assert call["content"] == "兜底答案"
        assert "om_1" not in ch._stream_bufs

    @pytest.mark.asyncio
    async def test_resuming_stream_end_keeps_buffer(self):
        """Tool-round boundary: stream stays open for the next round."""
        ch = _make_channel()
        buf = _QQStreamBuf(
            msg_id="om_1", openid="user1", stream_msg_id="stream_001", index=1,
            last_edit=0.0,
        )
        buf.append_answer("第一段")
        ch._stream_bufs["om_1"] = buf
        n_calls = ch._client.api._http.request.await_count
        await ch.send_delta(
            "user1", "", {"message_id": "om_1", "_stream_end": True, "_resuming": True}
        )

        assert "om_1" in ch._stream_bufs
        # No finalize push happened at the tool-round boundary
        assert ch._client.api._http.request.await_count == n_calls

    @pytest.mark.asyncio
    async def test_tool_hint_routed_into_stream(self):
        ch = _make_channel()
        buf = _QQStreamBuf(
            msg_id="om_1", openid="user1", stream_msg_id="stream_001", index=1,
            last_edit=0.0,
        )
        buf.append_reasoning("思考")
        ch._stream_bufs["om_1"] = buf
        await ch.send(
            OutboundMessage(
                channel="qq", chat_id="user1",
                content="web_search(query='test')",
                metadata={"_tool_hint": True, "message_id": "om_1"},
            )
        )

        buf = ch._stream_bufs["om_1"]
        assert any(
            k == _QQStreamBuf.KIND_TOOL and "web_search" in c for k, c in buf.events
        )
        payload = _last_stream_payload(ch)
        assert "🔧 `web_search" in payload["content_raw"]
        # Did not fall back to a standalone message
        ch._client.api.post_c2c_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tool_hint_group_chat_sends_standalone(self):
        ch = _make_channel()
        ch._chat_type_cache["group1"] = "group"
        await ch.send(
            OutboundMessage(
                channel="qq", chat_id="group1",
                content="read_file(path='a.py')",
                metadata={"_tool_hint": True, "message_id": "om_g"},
            )
        )

        ch._client.api.post_group_message.assert_awaited_once()
        call = ch._client.api.post_group_message.await_args.kwargs
        assert "🔧" in call["content"]
        assert "read_file" in call["content"]

    @pytest.mark.asyncio
    async def test_stream_disabled_uses_regular_path(self):
        ch = _make_channel(streaming=False)
        await ch.send_reasoning_delta("user1", "思考", {"message_id": "om_1"})
        ch._client.api._http.request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stream_failure_stops_pushes(self):
        """API error marks the buffer failed; further deltas are ignored."""
        ch = _make_channel()
        ch._client.api._http.request = AsyncMock(side_effect=RuntimeError("api error"))
        await ch.send_reasoning_delta("user1", "思考", {"message_id": "om_1"})

        buf = ch._stream_bufs["om_1"]
        assert buf.failed is True
        n_calls = ch._client.api._http.request.await_count
        await ch.send_reasoning_delta("user1", "更多", {"message_id": "om_1"})
        assert ch._client.api._http.request.await_count == n_calls


class TestVoiceTranscription:
    @pytest.mark.asyncio
    async def test_voice_attachment_transcribed(self, monkeypatch):
        ch = _make_channel()
        monkeypatch.setattr(
            ch, "_download_to_media_dir_chunked", AsyncMock(return_value="/tmp/voice.wav")
        )
        monkeypatch.setattr(
            ch, "transcribe_audio", AsyncMock(return_value="你好世界")
        )

        data = SimpleNamespace(
            id="msg_v1",
            content="",
            author=SimpleNamespace(id="user1", user_openid="user1"),
            attachments=[
                SimpleNamespace(
                    url="//multimedia.nt.qq.com/voice.silk",
                    filename="voice",
                    content_type="voice",
                    voice_wav_url="https://wav.example.com/v.wav",
                    asr_refer_text="",
                )
            ],
        )
        await ch._on_message(data, is_group=False)

        msg = await ch.bus.consume_inbound()
        assert "[transcription: 你好世界]" in msg.content
        # SILK raw url must not be downloaded into media
        ch._download_to_media_dir_chunked.assert_awaited_once()
        assert ch._download_to_media_dir_chunked.await_args.args[0] == (
            "https://wav.example.com/v.wav"
        )

    @pytest.mark.asyncio
    async def test_voice_falls_back_to_asr_refer_text(self, monkeypatch):
        ch = _make_channel()
        monkeypatch.setattr(
            ch, "_download_to_media_dir_chunked", AsyncMock(return_value=None)
        )

        data = SimpleNamespace(
            id="msg_v2",
            content="",
            author=SimpleNamespace(id="user1", user_openid="user1"),
            attachments=[
                SimpleNamespace(
                    url="",
                    filename="",
                    content_type="voice",
                    voice_wav_url="https://wav.example.com/v.wav",
                    asr_refer_text="平台识别结果",
                )
            ],
        )
        await ch._on_message(data, is_group=False)

        msg = await ch.bus.consume_inbound()
        assert "[transcription: 平台识别结果]" in msg.content

    @pytest.mark.asyncio
    async def test_voice_fields_read_from_raw_payload(self, monkeypatch):
        """botpy's _Attachments drops voice_wav_url/asr_refer_text; the raw
        payload patch keeps them in att.raw. Simulate a real botpy attachment:
        only fixed fields + raw dict, no voice attributes."""
        ch = _make_channel()
        monkeypatch.setattr(
            ch, "_download_to_media_dir_chunked", AsyncMock(return_value="/tmp/v.wav")
        )
        monkeypatch.setattr(
            ch, "transcribe_audio", AsyncMock(return_value="转写文本")
        )

        data = SimpleNamespace(
            id="msg_v3",
            content="",
            author=SimpleNamespace(id="user1", user_openid="user1"),
            attachments=[
                SimpleNamespace(
                    url="",
                    filename="voice",
                    content_type="voice",
                    raw={
                        "content_type": "voice",
                        "filename": "voice",
                        "voice_wav_url": "https://wav.example.com/v.wav",
                        "asr_refer_text": "平台文本",
                    },
                )
            ],
        )
        await ch._on_message(data, is_group=False)

        msg = await ch.bus.consume_inbound()
        assert "[transcription: 转写文本]" in msg.content

    @pytest.mark.asyncio
    async def test_voice_without_any_text_yields_notice(self, monkeypatch):
        """Voice that produces no transcription must not be silently
        dropped — the agent receives a failure notice."""
        ch = _make_channel()
        monkeypatch.setattr(
            ch, "_download_to_media_dir_chunked", AsyncMock(return_value=None)
        )

        data = SimpleNamespace(
            id="msg_v4",
            content="",
            author=SimpleNamespace(id="user1", user_openid="user1"),
            attachments=[
                SimpleNamespace(
                    url="", filename="", content_type="voice",
                    raw={"content_type": "voice"},
                )
            ],
        )
        await ch._on_message(data, is_group=False)

        msg = await ch.bus.consume_inbound()
        assert "[语音消息转写失败]" in msg.content

    @pytest.mark.asyncio
    async def test_ack_skipped_in_c2c_streaming_mode(self):
        ch = _make_channel()
        data = SimpleNamespace(
            id="msg_a1",
            content="hello",
            author=SimpleNamespace(id="user1", user_openid="user1"),
            attachments=[],
        )
        await ch._on_message(data, is_group=False)

        ch._client.api.post_c2c_message.assert_not_awaited()
        msg = await ch.bus.consume_inbound()
        assert msg.content == "hello"


def test_manager_preserves_class_tool_hint_default():
    """ChannelManager must not clobber QQ's class-level send_tool_hints=True.

    Regression: _init_channels used to override channel flags with the global
    ChannelsConfig defaults (send_tool_hints=False), which silently dropped
    every _tool_hint frame at dispatch even though QQ opted in at class level.
    """
    from nanobot.channels.manager import ChannelManager
    from nanobot.config.schema import Config

    cfg = Config()
    assert cfg.channels.send_tool_hints is False  # global default stays off
    cfg.channels.qq = {"enabled": True, "appId": "x", "secret": "y", "allowFrom": ["*"]}

    mgr = ChannelManager(cfg, MessageBus())
    assert "qq" in mgr.channels
    assert mgr.channels["qq"].send_tool_hints is True
    # Explicit per-channel override still wins
    cfg2 = Config()
    cfg2.channels.qq = {
        "enabled": True, "appId": "x", "secret": "y", "allowFrom": ["*"],
        "sendToolHints": False,
    }
    mgr2 = ChannelManager(cfg2, MessageBus())
    assert mgr2.channels["qq"].send_tool_hints is False
