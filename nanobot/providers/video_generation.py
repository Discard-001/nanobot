"""Video generation provider helpers.

Implements the OpenAI Videos-compatible async API used by Agnes Video
(``POST {base}/videos`` + polling ``GET {base}/videos/{video_id}``):

- Create a task with ``mode``: ``text`` (prompt only), ``keyframe``
  (first/last frame control) or ``reference`` (image/audio references).
- Poll until ``status`` is ``completed`` or ``failed``.
- Completed tasks expose the video URL via ``metadata.url``.

Media references accept public http(s) URLs or base64 data URLs (local
files can be inlined as data URLs by the tool layer).
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger

from nanobot.providers.registry import find_by_name

_DEFAULT_TIMEOUT_S = 60.0
_DEFAULT_POLL_INTERVAL_S = 2.0
_DEFAULT_POLL_TIMEOUT_S = 600.0
_DEFAULT_CREATE_RETRIES = 3
_CREATE_RETRY_BACKOFF_S = 5.0

# Transient create-side conditions worth retrying (queue full / rate limit).
_RETRYABLE_CODES = {"video_queue_full", "rate_limit_exceeded"}


class VideoGenerationError(RuntimeError):
    """Raised when the video generation provider cannot return a video."""


@dataclass(frozen=True)
class GeneratedVideoResponse:
    """Video URL and optional text returned by the provider."""

    video_url: str
    remote_url: str
    content: str
    raw: dict[str, Any]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_VIDEO_GEN_PROVIDERS: dict[str, type[VideoGenerationProvider]] = {}


def register_video_gen_provider(cls: type[VideoGenerationProvider]) -> None:
    """Register a video provider at import time only."""
    name = cls.provider_name
    if not name:
        raise ValueError(f"{cls.__name__} must set provider_name")
    _VIDEO_GEN_PROVIDERS[name] = cls


def get_video_gen_provider(name: str) -> type[VideoGenerationProvider] | None:
    return _VIDEO_GEN_PROVIDERS.get(name)


def video_gen_provider_names() -> tuple[str, ...]:
    """Return registered video generation provider names in registry order."""
    return tuple(_VIDEO_GEN_PROVIDERS)


def video_gen_provider_configs(config: Any) -> dict[str, Any]:
    providers_cfg = config.providers
    return {
        name: pc
        for name in _VIDEO_GEN_PROVIDERS
        if (pc := getattr(providers_cfg, name, None)) is not None
    }


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class VideoGenerationProvider(ABC):
    """Base class for video generation provider clients."""

    provider_name: str = ""
    missing_key_message: str = ""
    default_timeout: float = _DEFAULT_TIMEOUT_S

    def __init__(
        self,
        *,
        api_key: str | None,
        api_base: str | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_base = self._resolve_base_url(api_base)
        self.extra_headers = extra_headers or {}
        self.extra_body = extra_body or {}
        self.timeout = timeout if timeout is not None else self.default_timeout
        self._client = client

    def _resolve_base_url(self, api_base: str | None) -> str:
        if api_base:
            return api_base.rstrip("/")
        spec = find_by_name(self.provider_name)
        if spec and spec.default_api_base:
            return spec.default_api_base.rstrip("/")
        return self._default_base_url()

    def _default_base_url(self) -> str:
        return ""

    @abstractmethod
    async def generate(
        self,
        *,
        prompt: str,
        model: str,
        mode: str = "text",
        seconds: str | None = None,
        size: str | None = None,
        aspect_ratio: str | None = None,
        first_frame: str | None = None,
        last_frame: str | None = None,
        images: list[str] | None = None,
        audios: list[str] | None = None,
        poll_interval: float = _DEFAULT_POLL_INTERVAL_S,
        poll_timeout: float = _DEFAULT_POLL_TIMEOUT_S,
        create_retries: int = _DEFAULT_CREATE_RETRIES,
    ) -> GeneratedVideoResponse: ...

    def _require_video_url(self, data: dict[str, Any]) -> str:
        metadata = data.get("metadata")
        url = metadata.get("url") if isinstance(metadata, dict) else None
        if isinstance(url, str) and url:
            return url
        raise VideoGenerationError(
            f"{self.provider_name} completed task without a video URL: {data}"
        )


# ---------------------------------------------------------------------------
# OpenAI Videos-compatible client (Agnes Video etc.)
# ---------------------------------------------------------------------------


class OpenAIVideoGenerationClient(VideoGenerationProvider):
    """Async client for the OpenAI Videos-compatible video generation API."""

    provider_name = "openai"
    missing_key_message = (
        "OpenAI API key is not configured. Set providers.openai.apiKey."
    )

    def _default_base_url(self) -> str:
        return "https://api.openai.com/v1"

    async def generate(
        self,
        *,
        prompt: str,
        model: str,
        mode: str = "text",
        seconds: str | None = None,
        size: str | None = None,
        aspect_ratio: str | None = None,
        first_frame: str | None = None,
        last_frame: str | None = None,
        images: list[str] | None = None,
        audios: list[str] | None = None,
        poll_interval: float = _DEFAULT_POLL_INTERVAL_S,
        poll_timeout: float = _DEFAULT_POLL_TIMEOUT_S,
        create_retries: int = _DEFAULT_CREATE_RETRIES,
    ) -> GeneratedVideoResponse:
        if not self.api_key:
            raise VideoGenerationError(self.missing_key_message)

        body = self._build_body(
            prompt=prompt,
            model=model,
            mode=mode,
            seconds=seconds,
            size=size,
            aspect_ratio=aspect_ratio,
            first_frame=first_frame,
            last_frame=last_frame,
            images=images,
            audios=audios,
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

        data = await self._create_task(body, headers, create_retries)
        video_id = data.get("video_id") or data.get("id") or ""
        if not isinstance(video_id, str) or not video_id:
            raise VideoGenerationError(
                f"{self.provider_name} video create response missing video_id: {data}"
            )

        # A completed create response may already carry the final state.
        status = str(data.get("status") or "")
        if status in {"completed", "failed"}:
            final = data
        else:
            final = await self._poll_task(
                video_id, model=model, headers=headers,
                poll_interval=poll_interval, poll_timeout=poll_timeout,
            )

        if str(final.get("status")) == "failed":
            error = final.get("error")
            detail = error if isinstance(error, dict) else final
            raise VideoGenerationError(
                f"{self.provider_name} video generation failed: {detail}"
            )

        video_url = self._require_video_url(final)
        content = str(final.get("content") or "")
        return GeneratedVideoResponse(
            video_url=video_url, remote_url=video_url, content=content, raw=final
        )

    def _build_body(
        self,
        *,
        prompt: str,
        model: str,
        mode: str,
        seconds: str | None,
        size: str | None,
        aspect_ratio: str | None,
        first_frame: str | None,
        last_frame: str | None,
        images: list[str] | None,
        audios: list[str] | None,
    ) -> dict[str, Any]:
        if mode not in {"text", "keyframe", "reference"}:
            raise VideoGenerationError(
                f"invalid video mode {mode!r}; expected text/keyframe/reference"
            )

        body: dict[str, Any] = {"model": model, "prompt": prompt, "mode": mode}
        if seconds:
            body["seconds"] = str(seconds)
        if size:
            body["size"] = str(size)
        if aspect_ratio:
            body["aspect_ratio"] = str(aspect_ratio)
        body["n"] = 1

        if mode == "keyframe":
            if not first_frame and not last_frame:
                raise VideoGenerationError(
                    "keyframe mode requires first_frame and/or last_frame"
                )
            if first_frame:
                body["first_frame"] = first_frame
            if last_frame:
                body["last_frame"] = last_frame
        elif mode == "reference":
            image_list = [u for u in (images or []) if u]
            audio_list = [u for u in (audios or []) if u]
            if not image_list and not audio_list:
                raise VideoGenerationError(
                    "reference mode requires at least one image or audio reference"
                )
            if image_list:
                body["images"] = image_list
            if audio_list:
                body["audios"] = audio_list
        else:  # text mode must not carry media fields
            media_leak = [
                k for k, v in {
                    "first_frame": first_frame, "last_frame": last_frame,
                    "images": images, "audios": audios,
                }.items() if v
            ]
            if media_leak:
                raise VideoGenerationError(
                    f"text mode does not accept media fields: {media_leak}"
                )

        body.update(self.extra_body)
        return body

    async def _create_task(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
        create_retries: int,
    ) -> dict[str, Any]:
        """POST the create request, retrying transient queue/rate-limit errors."""
        url = f"{self.api_base}/videos"
        attempts = max(1, create_retries)
        last_error: VideoGenerationError | None = None
        for attempt in range(1, attempts + 1):
            response = await self._http_post(url, headers=headers, body=body)
            if response.status_code == 200:
                return response.json()

            detail = response.text[:500]
            code = ""
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    code = str(payload.get("code") or "")
            except Exception:
                payload = None

            retryable = (
                response.status_code in {429, 503}
                or code in _RETRYABLE_CODES
            )
            last_error = VideoGenerationError(
                f"OpenAI video creation failed (HTTP {response.status_code}): {detail}"
            )
            if not retryable or attempt == attempts:
                raise last_error
            logger.warning(
                "video create attempt {}/{} failed ({}), retrying in {}s",
                attempt, attempts, code or response.status_code, _CREATE_RETRY_BACKOFF_S,
            )
            await asyncio.sleep(_CREATE_RETRY_BACKOFF_S)
        raise last_error or VideoGenerationError("video creation failed")

    async def _poll_task(
        self,
        video_id: str,
        *,
        model: str,
        headers: dict[str, str],
        poll_interval: float,
        poll_timeout: float,
    ) -> dict[str, Any]:
        """Poll the task until completed/failed or the overall timeout expires."""
        url = f"{self.api_base}/videos/{video_id}"
        query = {"model_name": model} if model else None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(poll_interval, poll_timeout)

        while True:
            response = await self._http_get(url, headers=headers, params=query)
            if response.status_code != 200:
                raise VideoGenerationError(
                    f"OpenAI video polling failed (HTTP {response.status_code}): "
                    f"{response.text[:500]}"
                )
            data = response.json()
            if not isinstance(data, dict):
                raise VideoGenerationError("video polling returned a non-object payload")
            status = str(data.get("status") or "")
            if status in {"completed", "failed"}:
                return data
            if loop.time() >= deadline:
                raise VideoGenerationError(
                    f"video generation timed out after {poll_timeout:.0f}s "
                    f"(last status: {status or 'unknown'})"
                )
            await asyncio.sleep(max(0.5, poll_interval))

    async def _http_post(
        self, url: str, *, headers: dict[str, str], body: dict[str, Any]
    ) -> httpx.Response:
        if self._client is not None:
            return await self._client.post(url, headers=headers, json=body)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await client.post(url, headers=headers, json=body)

    async def _http_get(
        self, url: str, *, headers: dict[str, str], params: dict[str, str] | None
    ) -> httpx.Response:
        if self._client is not None:
            return await self._client.get(url, headers=headers, params=params)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await client.get(url, headers=headers, params=params)


register_video_gen_provider(OpenAIVideoGenerationClient)
