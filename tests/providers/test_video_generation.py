from __future__ import annotations

import base64
from typing import Any

import httpx
import pytest

from nanobot.providers.video_generation import (
    OpenAIVideoGenerationClient,
    VideoGenerationError,
    get_video_gen_provider,
    register_video_gen_provider,
    video_gen_provider_names,
)

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x04\x00\x00\x00\xb5\x1c\x0c\x02"
    b"\x00\x00\x00\x0bIDATx\xdacd\xfc\xff\x1f\x00\x03\x03"
    b"\x02\x00\xef\xbf\xa7\xdb\x00\x00\x00\x00IEND\xaeB`\x82"
)
PNG_DATA_URL = (
    "data:image/png;base64,"
    + base64.b64encode(PNG_BYTES).decode("ascii")
)


def _client() -> OpenAIVideoGenerationClient:
    return OpenAIVideoGenerationClient(
        api_key="sk-test", api_base="https://gateway.example/v1"
    )


def _resp(status: int = 200, payload: dict[str, Any] | None = None) -> httpx.Response:
    return httpx.Response(
        status,
        json=payload or {},
        request=httpx.Request("POST", "https://gateway.example/v1/videos"),
    )


class TestBuildBody:
    def test_text_mode_minimal(self) -> None:
        body = _client()._build_body(
            prompt="a cat", model="agnes-video-2.5-flash", mode="text",
            seconds=None, size=None, aspect_ratio=None,
            first_frame=None, last_frame=None, images=None, audios=None,
        )
        assert body["mode"] == "text"
        assert body["prompt"] == "a cat"
        assert body["n"] == 1
        assert "images" not in body and "first_frame" not in body

    def test_keyframe_requires_frames(self) -> None:
        with pytest.raises(VideoGenerationError, match="first_frame"):
            _client()._build_body(
                prompt="p", model="m", mode="keyframe", seconds=None, size=None,
                aspect_ratio=None, first_frame=None, last_frame=None,
                images=None, audios=None,
            )

    def test_keyframe_builds_frames(self) -> None:
        body = _client()._build_body(
            prompt="p", model="m", mode="keyframe", seconds="6", size="720P",
            aspect_ratio="9:16", first_frame=PNG_DATA_URL, last_frame=None,
            images=None, audios=None,
        )
        assert body["first_frame"] == PNG_DATA_URL
        assert body["seconds"] == "6"
        assert body["aspect_ratio"] == "9:16"

    def test_reference_requires_media(self) -> None:
        with pytest.raises(VideoGenerationError, match="at least one"):
            _client()._build_body(
                prompt="p", model="m", mode="reference", seconds=None, size=None,
                aspect_ratio=None, first_frame=None, last_frame=None,
                images=[], audios=[],
            )

    def test_reference_builds_media_lists(self) -> None:
        body = _client()._build_body(
            prompt="p", model="m", mode="reference", seconds=None, size=None,
            aspect_ratio=None, first_frame=None, last_frame=None,
            images=[PNG_DATA_URL], audios=["https://a.example/x.mp3"],
        )
        assert body["images"] == [PNG_DATA_URL]
        assert body["audios"] == ["https://a.example/x.mp3"]

    def test_text_mode_rejects_media(self) -> None:
        with pytest.raises(VideoGenerationError, match="does not accept media"):
            _client()._build_body(
                prompt="p", model="m", mode="text", seconds=None, size=None,
                aspect_ratio=None, first_frame=None, last_frame=None,
                images=[PNG_DATA_URL], audios=None,
            )

    def test_invalid_mode(self) -> None:
        with pytest.raises(VideoGenerationError, match="invalid video mode"):
            _client()._build_body(
                prompt="p", model="m", mode="standard", seconds=None, size=None,
                aspect_ratio=None, first_frame=None, last_frame=None,
                images=None, audios=None,
            )


class TestGenerateFlow:
    async def test_create_and_poll_until_completed(self, monkeypatch) -> None:
        client = _client()
        create_payload = {
            "id": "t1", "task_id": "t1", "video_id": "v1",
            "status": "queued", "progress": 0,
        }
        poll_payloads = [
            {"video_id": "v1", "status": "in_progress", "progress": 40},
            {"video_id": "v1", "status": "completed", "progress": 100,
             "metadata": {"url": "https://cdn.example/v1.mp4"}},
        ]

        posts: list[str] = []
        gets: list[str] = []

        async def fake_post(url, *, headers, body):
            posts.append(url)
            return _resp(200, create_payload)

        async def fake_get(url, *, headers, params):
            gets.append((url, params))
            return _resp(200, poll_payloads[len(gets) - 1])

        monkeypatch.setattr(client, "_http_post", fake_post)
        monkeypatch.setattr(client, "_http_get", fake_get)
        monkeypatch.setattr(
            "nanobot.providers.video_generation.asyncio.sleep",
            _no_sleep,
        )

        response = await client.generate(
            prompt="a cat", model="agnes-video-2.5-flash", mode="text",
        )
        assert posts == ["https://gateway.example/v1/videos"]
        assert response.video_url == "https://cdn.example/v1.mp4"
        # polling passes model_name per the Agnes docs
        assert gets[0][0] == "https://gateway.example/v1/videos/v1"
        assert gets[0][1] == {"model_name": "agnes-video-2.5-flash"}

    async def test_failed_task_raises(self, monkeypatch) -> None:
        client = _client()
        monkeypatch.setattr(
            client, "_http_post",
            lambda url, **kw: _async(_resp(200, {"video_id": "v1", "status": "queued"})),
        )
        monkeypatch.setattr(
            client, "_http_get",
            lambda url, **kw: _async(_resp(200, {
                "video_id": "v1", "status": "failed",
                "error": {"message": "content policy"},
            })),
        )

        with pytest.raises(VideoGenerationError, match="content policy"):
            await client.generate(prompt="p", model="m", mode="text")

    async def test_queue_full_retries_then_succeeds(self, monkeypatch) -> None:
        client = _client()
        calls = {"n": 0}

        async def fake_post(url, *, headers, body):
            calls["n"] += 1
            if calls["n"] < 3:
                return _resp(429, {"code": "video_queue_full", "message": "queue full"})
            return _resp(200, {
                "video_id": "v2", "status": "completed",
                "metadata": {"url": "https://cdn.example/v2.mp4"},
            })

        monkeypatch.setattr(client, "_http_post", fake_post)
        monkeypatch.setattr(client, "_http_get", lambda url, **kw: _async(_resp(500)))
        monkeypatch.setattr(
            "nanobot.providers.video_generation.asyncio.sleep", _no_sleep
        )

        response = await client.generate(
            prompt="p", model="m", mode="text", create_retries=3,
        )
        assert calls["n"] == 3
        assert response.video_url == "https://cdn.example/v2.mp4"

    async def test_non_retryable_error_raises_immediately(self, monkeypatch) -> None:
        client = _client()
        calls = {"n": 0}

        async def fake_post(url, *, headers, body):
            calls["n"] += 1
            return _resp(400, {"code": "invalid_request", "message": "size must be 720P"})

        monkeypatch.setattr(client, "_http_post", fake_post)
        monkeypatch.setattr(
            "nanobot.providers.video_generation.asyncio.sleep", _no_sleep
        )

        with pytest.raises(VideoGenerationError, match="size must be 720P"):
            await client.generate(prompt="p", model="m", mode="text")
        assert calls["n"] == 1

    async def test_missing_api_key(self) -> None:
        client = OpenAIVideoGenerationClient(api_key=None, api_base="https://x/v1")
        with pytest.raises(VideoGenerationError, match="apiKey"):
            await client.generate(prompt="p", model="m")


class TestRegistry:
    def test_openai_registered(self) -> None:
        assert get_video_gen_provider("openai") is OpenAIVideoGenerationClient
        assert "openai" in video_gen_provider_names()

    def test_register_requires_name(self) -> None:
        class _Nameless(OpenAIVideoGenerationClient):
            provider_name = ""

        with pytest.raises(ValueError, match="provider_name"):
            register_video_gen_provider(_Nameless)


class _NoSleepAwaitable:
    def __await__(self):
        if False:
            yield
        return None


def _no_sleep(*_args, **_kwargs):
    return _NoSleepAwaitable()


def _async(value):
    class _Awaitable:
        def __await__(self):
            if False:
                yield
            return value

    return _Awaitable()
