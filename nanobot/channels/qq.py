"""QQ channel implementation using botpy SDK.

Inbound:
- Parse QQ botpy messages (C2C / Group)
- Download attachments to media dir using chunked streaming write (memory-safe)
- Voice messages (content_type=voice): download voice_wav_url, transcribe via
  the shared ASR provider (SiliconFlow/OpenAI/Groq), inline as [transcription: ...]
  with the platform asr_refer_text as fallback
- Publish to Nanobot bus via BaseChannel._handle_message()

Outbound:
- Send attachments (msg.media) first via QQ rich media API (base64 upload + msg_type=7)
- Then send text (plain or markdown)
- msg.media supports local paths, file:// paths, and http(s) URLs
- C2C streaming: /v2/users/{openid}/stream_messages with input_mode=replace
  (full text per chunk, prefix immutable). Reasoning trace and tool calls are
  surfaced above the answer via markdown quote blocks. Group chats have no
  streaming API — they fall back to the one-shot send path.

Notes:
- QQ restricts many audio/video formats. We conservatively classify as image vs file.
- Attachment structures differ across botpy versions; we try multiple field candidates.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
import re
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import unquote, urlparse

import aiohttp
from loguru import logger
from pydantic import Field

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import Base
from nanobot.security.network import validate_url_target
from nanobot.utils.logging_bridge import redirect_lib_logging

try:
    from nanobot.config.paths import get_media_dir
except Exception:  # pragma: no cover
    get_media_dir = None  # type: ignore

try:
    import botpy
    from botpy.http import Route

    QQ_AVAILABLE = True
except ImportError:  # pragma: no cover
    QQ_AVAILABLE = False
    botpy = None
    Route = None

if TYPE_CHECKING:
    from botpy.message import BaseMessage, C2CMessage, GroupMessage
    from botpy.types.message import Media


# QQ rich media file_type: 1=image, 4=file
# (2=voice, 3=video are restricted; we only use image vs file)
QQ_FILE_TYPE_IMAGE = 1
QQ_FILE_TYPE_FILE = 4

_IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".ico",
    ".svg",
}

# ---------------------------------------------------------------------------
# C2C streaming (stream_messages) support
# ---------------------------------------------------------------------------

_STREAM_HEADER = "💭 **执行过程**"
_STREAM_EDIT_INTERVAL = 0.8  # seconds between stream chunk pushes


@dataclass
class _QQStreamBuf:
    """Per-turn state for one C2C streaming message.

    QQ's stream API uses input_mode=replace: every chunk carries the full
    text and the already-delivered prefix is immutable (error 40007). Blocks
    must therefore render strictly in arrival order — re-grouping them by
    category (all reasoning, then all tool calls) mutates the submitted
    prefix when a later reasoning round arrives after a tool call. The
    buffer keeps an append-only event log instead.
    """

    # Event kinds recorded in the log.
    KIND_REASONING = "reasoning"
    KIND_TOOL = "tool"
    KIND_ANSWER = "answer"

    msg_id: str = ""  # inbound message id (passive reply window, 60 min)
    openid: str = ""  # C2C peer openid (stream endpoint target)
    stream_msg_id: str = ""  # assigned by the first chunk's response
    index: int = -1
    # (kind, content) in arrival order; rendering is a straight join so the
    # submitted prefix never changes between chunks.
    events: list[tuple[str, str]] = field(default_factory=list)
    reasoning_open: bool = False  # trailing reasoning segment accepts deltas
    reasoning_rounds: int = 0
    last_edit: float = 0.0
    failed: bool = False  # API error → stop retrying, fall back to send()

    def _quote(self, s: str) -> str:
        """Render a trace block as markdown quote lines."""
        return "\n".join(f"> {line}" if line.strip() else ">" for line in s.splitlines())

    def _stream_content(self) -> str:
        if not self.events:
            return ""
        parts = [_STREAM_HEADER]
        for kind, content in self.events:
            parts.append(self._quote(content) if kind == self.KIND_REASONING else content)
        return "\n\n".join(parts)

    def append_reasoning(self, delta: str) -> None:
        """Append a reasoning delta; a new round starts a fresh segment."""
        if self.reasoning_open and self.events and self.events[-1][0] == self.KIND_REASONING:
            self.events[-1] = (self.KIND_REASONING, self.events[-1][1] + delta)
            return
        # Separate consecutive rounds with an ellipsis marker.
        prefix = "···\n" if self.reasoning_rounds else ""
        self.reasoning_rounds += 1
        self.events.append((self.KIND_REASONING, prefix + delta))
        self.reasoning_open = True

    def close_reasoning(self) -> None:
        self.reasoning_open = False

    def append_tool(self, formatted: str) -> None:
        self.reasoning_open = False
        self.events.append((self.KIND_TOOL, formatted))

    def append_answer(self, delta: str) -> None:
        self.reasoning_open = False
        if self.events and self.events[-1][0] == self.KIND_ANSWER:
            self.events[-1] = (self.KIND_ANSWER, self.events[-1][1] + delta)
        else:
            self.events.append((self.KIND_ANSWER, delta))

    @property
    def answer_text(self) -> str:
        """Full answer text accumulated so far (for failure fallback)."""
        return "".join(c for k, c in self.events if k == self.KIND_ANSWER)

    def has_content(self) -> bool:
        return bool(self.events)


# Replace unsafe characters with "_", keep Chinese and common safe punctuation.
_SAFE_NAME_RE = re.compile(r"[^\w.\-()\[\]（）【】\u4e00-\u9fff]+", re.UNICODE)


def _sanitize_filename(name: str) -> str:
    """Sanitize filename to avoid traversal and problematic chars."""
    name = (name or "").strip()
    name = Path(name).name
    name = _SAFE_NAME_RE.sub("_", name).strip("._ ")
    return name


def _is_image_name(name: str) -> bool:
    return Path(name).suffix.lower() in _IMAGE_EXTS


def _guess_send_file_type(filename: str) -> int:
    """Conservative send type: images -> 1, else -> 4."""
    ext = Path(filename).suffix.lower()
    mime, _ = mimetypes.guess_type(filename)
    if ext in _IMAGE_EXTS or (mime and mime.startswith("image/")):
        return QQ_FILE_TYPE_IMAGE
    return QQ_FILE_TYPE_FILE


def _make_bot_class(channel: QQChannel) -> type[botpy.Client]:
    """Create a botpy Client subclass bound to the given channel."""
    intents = botpy.Intents(public_messages=True, direct_message=True)

    class _Bot(botpy.Client):
        def __init__(self):
            # Disable botpy's file log — nanobot uses loguru; default "botpy.log" fails on read-only fs
            super().__init__(intents=intents, ext_handlers=False)

        async def on_ready(self):
            logger.info("QQ bot ready: {}", self.robot.name)

        async def on_c2c_message_create(self, message: C2CMessage):
            await channel._on_message(message, is_group=False)

        async def on_group_at_message_create(self, message: GroupMessage):
            await channel._on_message(message, is_group=True)

        async def on_direct_message_create(self, message):
            await channel._on_message(message, is_group=False)

    return _Bot


class QQConfig(Base):
    """QQ channel configuration using botpy SDK."""

    enabled: bool = False
    app_id: str = ""
    secret: str = ""
    allow_from: list[str] = Field(default_factory=list)
    msg_format: Literal["plain", "markdown"] = "plain"
    ack_message: str = "⏳ Processing..."

    # C2C streaming (stream_messages): typing-effect replies with the
    # reasoning trace + tool calls surfaced above the answer. Group chats
    # have no streaming API and always use the one-shot send path.
    streaming: bool = True

    # Optional: directory to save inbound attachments. If empty, use nanobot get_media_dir("qq").
    media_dir: str = ""

    # Download tuning
    download_chunk_size: int = 1024 * 256  # 256KB
    download_max_bytes: int = 1024 * 1024 * 200  # 200MB safety limit


class QQChannel(BaseChannel):
    """QQ channel using botpy SDK with WebSocket connection."""

    name = "qq"
    display_name = "QQ"
    send_tool_hints: bool = True  # Enable tool call notifications
    _ws_probe_installed: bool = False  # process-wide, guards the parser patch

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return QQConfig().model_dump(by_alias=True)

    @classmethod
    def _install_ws_event_probe(cls) -> None:
        """Log every inbound QQ WS dispatch event (one line per message).

        Wraps each botpy parser so incoming event types and attachment kinds
        are visible in the service log. Unknown event types are logged by
        botpy itself as errors; total silence for an inbound message means
        the platform never pushed it (permission / event-subscription gap).
        """
        if cls._ws_probe_installed or not QQ_AVAILABLE:
            return
        cls._ws_probe_installed = True
        import botpy.connection as bp_connection

        orig_init = bp_connection.ConnectionState.__init__

        def patched_init(state_self, *args: Any, **kwargs: Any) -> None:
            orig_init(state_self, *args, **kwargs)
            for event_name, fn in list(state_self.parsers.items()):
                def wrapped(payload: dict, _name: str = event_name, _fn=fn):
                    try:
                        d = payload.get("d") or {}
                        atts = d.get("attachments")
                        att_info = ""
                        if isinstance(atts, list) and atts:
                            kinds = [
                                str(a.get("content_type") or a.get("file_type", "?"))
                                for a in atts
                                if isinstance(a, dict)
                            ]
                            att_info = f" attachments={kinds}"
                            # Voice payloads: log the raw field names so
                            # platform-side changes are visible in production.
                            for a in atts:
                                if (
                                    isinstance(a, dict)
                                    and str(a.get("content_type", "")).lower() == "voice"
                                ):
                                    att_info += f" voice_fields={sorted(a.keys())}"
                        logger.info("QQ WS event: {}{}", _name, att_info)
                    except Exception:  # probe must never break dispatch
                        pass
                    return _fn(payload)

                state_self.parsers[event_name] = wrapped

        bp_connection.ConnectionState.__init__ = patched_init

    _atts_raw_patched: bool = False  # process-wide guard for the attachments patch

    @classmethod
    def _install_attachments_raw_patch(cls) -> None:
        """Preserve raw attachment payloads on botpy message objects.

        botpy's nested ``_Attachments`` classes only surface a fixed field
        set (content_type/filename/url/...) and drop voice-specific payload
        fields — ``voice_wav_url`` and ``asr_refer_text`` — that QQ's C2C
        voice message events carry. Stashing the raw dict keeps them
        reachable via ``att.raw``.
        """
        if cls._atts_raw_patched or not QQ_AVAILABLE:
            return
        cls._atts_raw_patched = True
        from botpy import message as bp_message

        for obj in vars(bp_message).values():
            att_cls = getattr(obj, "_Attachments", None)
            if not isinstance(att_cls, type):
                continue
            orig_init = att_cls.__init__

            def patched_init(att_self, data, _orig=orig_init):
                _orig(att_self, data)
                att_self.raw = dict(data) if isinstance(data, dict) else {}

            att_cls.__init__ = patched_init

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = QQConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: QQConfig = config

        self._client: botpy.Client | None = None
        self._http: aiohttp.ClientSession | None = None

        self._processed_ids: deque[str] = deque(maxlen=1000)
        self._msg_seq: int = 1  # used to avoid QQ API dedup
        self._chat_type_cache: dict[str, str] = {}
        self._stream_bufs: dict[str, _QQStreamBuf] = {}

        self._media_root: Path = self._init_media_root()

    # ---------------------------
    # Lifecycle
    # ---------------------------

    def _init_media_root(self) -> Path:
        """Choose a directory for saving inbound attachments."""
        if self.config.media_dir:
            root = Path(self.config.media_dir).expanduser()
        elif get_media_dir:
            try:
                root = Path(get_media_dir("qq"))
            except Exception:
                root = Path.home() / ".nanobot" / "media" / "qq"
        else:
            root = Path.home() / ".nanobot" / "media" / "qq"

        root.mkdir(parents=True, exist_ok=True)
        self.logger.info("media directory: {}", str(root))
        return root

    async def start(self) -> None:
        """Start the QQ bot with auto-reconnect loop."""
        redirect_lib_logging("botpy", level="WARNING")
        if not QQ_AVAILABLE:
            self.logger.error("SDK not installed. Run: pip install qq-botpy")
            return

        if not self.config.app_id or not self.config.secret:
            self.logger.error("app_id and secret not configured")
            return

        self._running = True
        self._http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))

        self._install_ws_event_probe()
        self._install_attachments_raw_patch()
        self._client = _make_bot_class(self)()
        self.logger.info("bot started (C2C & Group supported)")
        await self._run_bot()

    async def _run_bot(self) -> None:
        """Run the bot connection with auto-reconnect."""
        while self._running:
            try:
                await self._client.start(appid=self.config.app_id, secret=self.config.secret)
            except Exception as e:
                self.logger.warning("bot error: {}", e)
            if self._running:
                self.logger.info("Reconnecting bot in 5 seconds...")
                await asyncio.sleep(5)

    async def stop(self) -> None:
        """Stop bot and cleanup resources."""
        self._running = False
        if self._client:
            with suppress(Exception):
                await self._client.close()
        self._client = None

        if self._http:
            with suppress(Exception):
                await self._http.close()
        self._http = None

        self.logger.info("bot stopped")

    # ---------------------------
    # Outbound (send)
    # ---------------------------

    async def send(self, msg: OutboundMessage) -> None:
        """Send attachments first, then text."""
        try:
            if not self._client:
                self.logger.warning("client not initialized")
                return

            msg_id = msg.metadata.get("message_id")
            chat_type = self._chat_type_cache.get(msg.chat_id, "c2c")
            is_group = chat_type == "group"

            # Tool hints: route into the active C2C stream when one exists;
            # otherwise deliver as a standalone (group chats / stream off).
            if msg.metadata.get("_tool_hint"):
                hint = (msg.content or "").strip()
                if not hint:
                    return
                if await self._append_tool_hint_stream(msg.chat_id, hint, msg.metadata):
                    return
                await self._send_text_only(
                    chat_id=msg.chat_id,
                    is_group=is_group,
                    msg_id=msg_id,
                    content=self._format_tool_hint(hint),
                )
                return

            # 1) Send media
            for media_ref in msg.media or []:
                ok = await self._send_media(
                    chat_id=msg.chat_id,
                    media_ref=media_ref,
                    msg_id=msg_id,
                    is_group=is_group,
                )
                if not ok:
                    filename = (
                        os.path.basename(urlparse(media_ref).path)
                        or os.path.basename(media_ref)
                        or "file"
                    )
                    await self._send_text_only(
                        chat_id=msg.chat_id,
                        is_group=is_group,
                        msg_id=msg_id,
                        content=f"[Attachment send failed: {filename}]",
                    )

            # 2) Send text
            if msg.content and msg.content.strip():
                await self._send_text_only(
                    chat_id=msg.chat_id,
                    is_group=is_group,
                    msg_id=msg_id,
                    content=msg.content.strip(),
                )
        except (aiohttp.ClientError, OSError):
            # Network / transport errors — propagate so ChannelManager can retry
            raise
        except Exception:
            self.logger.exception("Error sending message to chat_id={}", msg.chat_id)

    async def _send_text_only(
        self,
        chat_id: str,
        is_group: bool,
        msg_id: str | None,
        content: str,
    ) -> None:
        """Send a plain/markdown text message."""
        if not self._client:
            return

        self._msg_seq += 1
        use_markdown = self.config.msg_format == "markdown"
        payload: dict[str, Any] = {
            "msg_type": 2 if use_markdown else 0,
            "msg_id": msg_id,
            "msg_seq": self._msg_seq,
        }
        if use_markdown:
            payload["markdown"] = {"content": content}
        else:
            payload["content"] = content

        if is_group:
            await self._client.api.post_group_message(group_openid=chat_id, **payload)
        else:
            await self._client.api.post_c2c_message(openid=chat_id, **payload)

    async def _send_media(
        self,
        chat_id: str,
        media_ref: str,
        msg_id: str | None,
        is_group: bool,
    ) -> bool:
        """Read bytes -> base64 upload -> msg_type=7 send."""
        if not self._client:
            return False

        data, filename = await self._read_media_bytes(media_ref)
        if not data or not filename:
            return False

        try:
            file_type = _guess_send_file_type(filename)
            file_data_b64 = base64.b64encode(data).decode()

            media_obj = await self._post_base64file(
                chat_id=chat_id,
                is_group=is_group,
                file_type=file_type,
                file_data=file_data_b64,
                file_name=filename,
                srv_send_msg=False,
            )
            if not media_obj:
                self.logger.error("media upload failed: empty response")
                return False

            self._msg_seq += 1
            if is_group:
                await self._client.api.post_group_message(
                    group_openid=chat_id,
                    msg_type=7,
                    msg_id=msg_id,
                    msg_seq=self._msg_seq,
                    media=media_obj,
                )
            else:
                await self._client.api.post_c2c_message(
                    openid=chat_id,
                    msg_type=7,
                    msg_id=msg_id,
                    msg_seq=self._msg_seq,
                    media=media_obj,
                )

            self.logger.info("media sent: {}", filename)
            return True
        except (aiohttp.ClientError, OSError) as e:
            # Network / transport errors — propagate for retry by caller
            self.logger.warning("send media network error filename={} err={}", filename, e)
            raise
        except Exception:
            # API-level or other non-network errors — return False so send() can fallback
            self.logger.exception("send media failed filename={}", filename)
            return False

    async def _read_media_bytes(self, media_ref: str) -> tuple[bytes | None, str | None]:
        """Read bytes from http(s) or local file path; return (data, filename)."""
        media_ref = (media_ref or "").strip()
        if not media_ref:
            return None, None

        # Local file: plain path or file:// URI
        if not media_ref.startswith("http://") and not media_ref.startswith("https://"):
            try:
                if media_ref.startswith("file://"):
                    parsed = urlparse(media_ref)
                    # Windows: path in netloc; Unix: path in path
                    raw = parsed.path or parsed.netloc
                    local_path = Path(unquote(raw))
                else:
                    local_path = Path(os.path.expanduser(media_ref))

                if not local_path.is_file():
                    self.logger.warning("outbound media file not found: {}", str(local_path))
                    return None, None

                data = await asyncio.to_thread(local_path.read_bytes)
                return data, local_path.name
            except Exception as e:
                self.logger.warning("outbound media read error ref={} err={}", media_ref, e)
                return None, None

        # Remote URL
        ok, err = validate_url_target(media_ref)
        if not ok:
            self.logger.warning("outbound media URL validation failed url={} err={}", media_ref, err)
            return None, None

        if not self._http:
            self._http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))
        try:
            async with self._http.get(media_ref, allow_redirects=True) as resp:
                if resp.status >= 400:
                    self.logger.warning(
                        "outbound media download failed status={} url={}",
                        resp.status,
                        media_ref,
                    )
                    return None, None
                data = await resp.read()
                if not data:
                    return None, None
                filename = os.path.basename(urlparse(media_ref).path) or "file.bin"
                return data, filename
        except Exception as e:
            self.logger.warning("outbound media download error url={} err={}", media_ref, e)
            return None, None

    # https://github.com/tencent-connect/botpy/issues/198
    # https://bot.q.qq.com/wiki/develop/api-v2/server-inter/message/send-receive/rich-media.html
    async def _post_base64file(
        self,
        chat_id: str,
        is_group: bool,
        file_type: int,
        file_data: str,
        file_name: str | None = None,
        srv_send_msg: bool = False,
    ) -> Media:
        """Upload base64-encoded file and return Media object."""
        if not self._client:
            raise RuntimeError("QQ client not initialized")

        if is_group:
            endpoint = "/v2/groups/{group_openid}/files"
            id_key = "group_openid"
        else:
            endpoint = "/v2/users/{openid}/files"
            id_key = "openid"

        payload: dict[str, Any] = {
            id_key: chat_id,
            "file_type": file_type,
            "file_data": file_data,
            "srv_send_msg": srv_send_msg,
        }
        # Only pass file_name for non-image types (file_type=4).
        # Passing file_name for images causes QQ client to render them as
        # file attachments instead of inline images.
        if file_type != QQ_FILE_TYPE_IMAGE and file_name:
            payload["file_name"] = file_name

        route = Route("POST", endpoint, **{id_key: chat_id})
        result = await self._client.api._http.request(route, json=payload)

        # Extract only the file_info field to avoid extra fields (file_uuid, ttl, etc.)
        # that may confuse QQ client when sending the media object.
        if isinstance(result, dict) and "file_info" in result:
            return {"file_info": result["file_info"]}
        return result

    # ---------------------------
    # C2C streaming (stream_messages)
    # ---------------------------

    @staticmethod
    def _stream_key(chat_id: str, metadata: dict[str, Any] | None = None) -> str:
        """Scope streaming buffers to the inbound message when available."""
        meta = metadata or {}
        return meta.get("message_id") or chat_id

    def _streaming_enabled_for(self, chat_id: str) -> bool:
        """C2C only: group chats have no streaming API."""
        return (
            bool(self.config.streaming)
            and self._client is not None
            and self._chat_type_cache.get(chat_id, "c2c") == "c2c"
        )

    async def _post_stream_chunk(
        self, buf: _QQStreamBuf, *, final: bool
    ) -> str | None:
        """Push one chunk to /v2/users/{openid}/stream_messages.

        Returns the stream_msg_id (first chunk creates it), or None on
        failure. ``final=True`` sends input_state=10 to close the stream.
        """
        if not self._client:
            return None
        payload: dict[str, Any] = {
            "input_mode": "replace",
            "input_state": 10 if final else 1,
            "index": buf.index,
            "content_type": "markdown",
            "content_raw": buf._stream_content(),
            "msg_seq": 1,
        }
        if buf.msg_id:
            payload["msg_id"] = buf.msg_id
        if buf.stream_msg_id:
            payload["stream_msg_id"] = buf.stream_msg_id

        route = Route("POST", "/v2/users/{openid}/stream_messages", openid=buf.openid)
        try:
            result = await self._client.api._http.request(route, json=payload)
        except Exception:
            self.logger.exception("stream chunk failed index={}", buf.index)
            return None
        if isinstance(result, dict) and result.get("id"):
            return str(result["id"])
        return None

    async def _push_stream(
        self, chat_id: str, buf: _QQStreamBuf, *, final: bool = False, force: bool = False
    ) -> None:
        """Throttled push of the current full content as one stream chunk."""
        now = time.monotonic()
        if not force and (now - buf.last_edit) < _STREAM_EDIT_INTERVAL:
            return
        buf.index += 1
        stream_id = await self._post_stream_chunk(buf, final=final)
        buf.last_edit = time.monotonic()
        if stream_id:
            if not buf.stream_msg_id:
                buf.stream_msg_id = stream_id
        else:
            # First chunk failing means no stream exists; later failures keep
            # the buffer but stop further pushes (final send() still delivers
            # the complete answer because meta lacks _streamed only on error).
            buf.failed = True

    async def send_reasoning_delta(
        self, chat_id: str, delta: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Stream model reasoning into the C2C stream (markdown quote block)."""
        if not delta or not self._streaming_enabled_for(chat_id):
            return
        meta = metadata or {}
        key = self._stream_key(chat_id, meta)
        buf = self._stream_bufs.get(key)
        if buf is None:
            buf = _QQStreamBuf(msg_id=str(meta.get("message_id") or ""), openid=chat_id)
            self._stream_bufs[key] = buf
        if buf.failed:
            return
        buf.append_reasoning(delta)
        await self._push_stream(chat_id, buf)

    async def send_reasoning_end(
        self, chat_id: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Flush the complete reasoning pass (gateways may burst deltas)."""
        meta = metadata or {}
        buf = self._stream_bufs.get(self._stream_key(chat_id, meta))
        if buf is None or not buf.reasoning_open or buf.failed:
            return
        buf.close_reasoning()
        await self._push_stream(chat_id, buf, force=True)

    async def send_delta(
        self, chat_id: str, delta: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Stream answer text into the C2C stream; close on _stream_end."""
        meta = metadata or {}
        key = self._stream_key(chat_id, meta)
        buf = self._stream_bufs.get(key)
        if buf is None:
            # No stream was started (e.g. model answered without reasoning and
            # no tool hint arrived). Create one so the answer still streams.
            if not self._streaming_enabled_for(chat_id) or not meta.get("_stream_delta"):
                return
            buf = _QQStreamBuf(msg_id=str(meta.get("message_id") or ""), openid=chat_id)
            self._stream_bufs[key] = buf

        if delta:
            buf.append_answer(delta)

        if meta.get("_stream_end"):
            if meta.get("_resuming"):
                # Tool-round boundary: the agent keeps working; the stream
                # stays open so the next round continues on the same message.
                return
            if buf.failed:
                # Stream broke mid-turn and the final send() is suppressed by
                # _streamed — deliver the answer as a normal message so it
                # is not lost.
                text = buf.answer_text.strip()
                if text:
                    await self._send_text_only(
                        chat_id=chat_id,
                        is_group=False,
                        msg_id=buf.msg_id or None,
                        content=text,
                    )
                self._stream_bufs.pop(key, None)
                return
            if not buf.stream_msg_id and not buf.has_content():
                return  # nothing was ever pushed; fall back to send()
            await self._push_stream(chat_id, buf, final=True, force=True)
            self._stream_bufs.pop(key, None)
            return

        if buf.failed:
            return
        await self._push_stream(chat_id, buf)

    def _format_tool_hint(self, hint: str) -> str:
        return f"🔧 `{hint}`"

    async def _append_tool_hint_stream(
        self, chat_id: str, hint: str, metadata: dict[str, Any]
    ) -> bool:
        """Route a tool hint into the active stream. True if handled."""
        if not self._streaming_enabled_for(chat_id):
            return False
        buf = self._stream_bufs.get(self._stream_key(chat_id, metadata))
        if buf is None:
            buf = _QQStreamBuf(
                msg_id=str(metadata.get("message_id") or ""), openid=chat_id
            )
            self._stream_bufs[self._stream_key(chat_id, metadata)] = buf
        if buf.failed:
            return False
        buf.append_tool(self._format_tool_hint(hint))
        await self._push_stream(chat_id, buf, force=True)
        return True

    # ---------------------------
    # Inbound (receive)
    # ---------------------------

    async def _on_message(self, data: C2CMessage | GroupMessage, is_group: bool = False) -> None:
        """Parse inbound message, download attachments, and publish to the bus."""
        try:
            if is_group:
                chat_id = data.group_openid
                user_id = data.author.member_openid
                chat_type = "group"
            else:
                chat_id = str(
                    getattr(data.author, "id", None)
                    or getattr(data.author, "user_openid", "unknown")
                )
                user_id = chat_id
                chat_type = "c2c"

            content = (data.content or "").strip()

            if not self.is_allowed(user_id):
                return

            if data.id in self._processed_ids:
                return
            self._processed_ids.append(data.id)
            self._chat_type_cache[chat_id] = chat_type

            # the data used by tests don't contain attachments property
            # so we use getattr with a default of [] to avoid AttributeError in tests
            attachments = getattr(data, "attachments", None) or []
            media_paths, recv_lines, att_meta = await self._handle_attachments(attachments)

            # Voice messages: transcribe (ASR) and inline the text so the LLM
            # can consume it. Attachments with content_type=voice carry
            # voice_wav_url (SILK->WAV) and asr_refer_text (official ASR hint).
            for att in attachments:
                ctype = (getattr(att, "content_type", "") or "").lower()
                if ctype != "voice":
                    continue
                raw = getattr(att, "raw", None) or {}
                wav_url = str(
                    raw.get("voice_wav_url")
                    or getattr(att, "voice_wav_url", None)
                    or ""
                ).strip()
                asr_text = str(
                    raw.get("asr_refer_text")
                    or getattr(att, "asr_refer_text", None)
                    or ""
                ).strip()
                transcription = ""
                if wav_url:
                    wav_path = await self._download_to_media_dir_chunked(
                        wav_url, filename_hint="voice.wav"
                    )
                    if wav_path:
                        transcription = await self.transcribe_audio(wav_path) or ""
                if not transcription and asr_text:
                    # Fall back to the platform-provided ASR reference text.
                    transcription = asr_text
                if transcription:
                    self.logger.info("voice transcribed: {} chars", len(transcription))
                    voice_line = f"[transcription: {transcription}]"
                    content = f"{voice_line}\n{content}".strip() if content else voice_line
                else:
                    # Voice arrived but yielded no text — tell the LLM instead
                    # of silently dropping the message.
                    self.logger.warning(
                        "voice message without transcription (wav_url={} asr={})",
                        bool(wav_url), bool(asr_text),
                    )
                    content = content or "[语音消息转写失败]"

            # Compose content that always contains actionable saved paths
            if recv_lines:
                tag = (
                    "[Image]"
                    if any(_is_image_name(Path(p).name) for p in media_paths)
                    else "[File]"
                )
                file_block = "Received files:\n" + "\n".join(recv_lines)
                content = (
                    f"{content}\n\n{file_block}".strip() if content else f"{tag}\n{file_block}"
                )

            if not content and not media_paths:
                return

            # Skip the ack in C2C streaming mode: the stream itself provides
            # immediate feedback (first chunk lands within a second).
            send_ack = bool(self.config.ack_message) and not (
                chat_type == "c2c" and self.config.streaming
            )
            if send_ack:
                try:
                    await self._send_text_only(
                        chat_id=chat_id,
                        is_group=is_group,
                        msg_id=data.id,
                        content=self.config.ack_message,
                    )
                except Exception:
                    self.logger.debug("ack message failed for chat_id={}", chat_id)

            await self._handle_message(
                sender_id=user_id,
                chat_id=chat_id,
                content=content,
                media=media_paths if media_paths else None,
                metadata={
                    "message_id": data.id,
                    "attachments": att_meta,
                },
            )
        except Exception:
            self.logger.exception("Error handling inbound message id={}", getattr(data, "id", "?"))

    async def _handle_attachments(
        self,
        attachments: list[BaseMessage._Attachments],
    ) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        """Extract, download (chunked), and format attachments for agent consumption."""
        media_paths: list[str] = []
        recv_lines: list[str] = []
        att_meta: list[dict[str, Any]] = []

        if not attachments:
            return media_paths, recv_lines, att_meta

        for att in attachments:
            url = getattr(att, "url", None) or ""
            filename = getattr(att, "filename", None) or ""
            ctype = getattr(att, "content_type", None) or ""

            # Voice attachments are handled separately (voice_wav_url download
            # + ASR transcription in _on_message); the raw url is SILK format
            # and useless to the agent, so skip downloading it here.
            if ctype.lower() == "voice":
                continue

            self.logger.info("Downloading file: {}", filename or url)
            local_path = await self._download_to_media_dir_chunked(url, filename_hint=filename)

            att_meta.append(
                {
                    "url": url,
                    "filename": filename,
                    "content_type": ctype,
                    "saved_path": local_path,
                }
            )

            if local_path:
                media_paths.append(local_path)
                shown_name = filename or os.path.basename(local_path)
                recv_lines.append(f"- {shown_name}\n  saved: {local_path}")
            else:
                shown_name = filename or url
                recv_lines.append(f"- {shown_name}\n  saved: [download failed]")

        return media_paths, recv_lines, att_meta

    async def _download_to_media_dir_chunked(
        self,
        url: str,
        filename_hint: str = "",
    ) -> str | None:
        """Download an inbound attachment using streaming chunk write.

        Uses chunked streaming to avoid loading large files into memory.
        Enforces a max download size and writes to a .part temp file
        that is atomically renamed on success.
        """
        # Handle protocol-relative URLs (e.g. "//multimedia.nt.qq.com/...")
        if url.startswith("//"):
            url = f"https:{url}"

        if not self._http:
            self._http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))

        safe = _sanitize_filename(filename_hint)
        ts = int(time.time() * 1000)
        tmp_path: Path | None = None

        try:
            async with self._http.get(
                url,
                timeout=aiohttp.ClientTimeout(total=120),
                allow_redirects=True,
            ) as resp:
                if resp.status != 200:
                    self.logger.warning("download failed: status={} url={}", resp.status, url)
                    return None

                ctype = (resp.headers.get("Content-Type") or "").lower()

                # Infer extension: url -> filename_hint -> content-type -> fallback
                ext = Path(urlparse(url).path).suffix
                if not ext:
                    ext = Path(filename_hint).suffix
                if not ext:
                    if "png" in ctype:
                        ext = ".png"
                    elif "jpeg" in ctype or "jpg" in ctype:
                        ext = ".jpg"
                    elif "gif" in ctype:
                        ext = ".gif"
                    elif "webp" in ctype:
                        ext = ".webp"
                    elif "pdf" in ctype:
                        ext = ".pdf"
                    else:
                        ext = ".bin"

                if safe:
                    if not Path(safe).suffix:
                        safe = safe + ext
                    filename = safe
                else:
                    filename = f"qq_file_{ts}{ext}"

                target = self._media_root / filename
                if target.exists():
                    target = self._media_root / f"{target.stem}_{ts}{target.suffix}"

                tmp_path = target.with_suffix(target.suffix + ".part")

                # Stream write
                downloaded = 0
                chunk_size = max(1024, int(self.config.download_chunk_size or 262144))
                max_bytes = max(
                    1024 * 1024, int(self.config.download_max_bytes or (200 * 1024 * 1024))
                )

                def _open_tmp():
                    tmp_path.parent.mkdir(parents=True, exist_ok=True)
                    return open(tmp_path, "wb")  # noqa: SIM115

                f = await asyncio.to_thread(_open_tmp)
                try:
                    async for chunk in resp.content.iter_chunked(chunk_size):
                        if not chunk:
                            continue
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            self.logger.warning(
                                "download exceeded max_bytes={} url={} -> abort",
                                max_bytes,
                                url,
                            )
                            return None
                        await asyncio.to_thread(f.write, chunk)
                finally:
                    await asyncio.to_thread(f.close)

                # Atomic rename
                await asyncio.to_thread(os.replace, tmp_path, target)
                tmp_path = None  # mark as moved
                self.logger.info("file saved: {}", str(target))
                return str(target)

        except Exception:
            self.logger.exception("download error")
            return None
        finally:
            # Cleanup partial file
            if tmp_path is not None:
                with suppress(Exception):
                    tmp_path.unlink(missing_ok=True)
