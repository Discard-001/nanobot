"""Artifact persistence helpers for generated media."""

from __future__ import annotations

import base64
import binascii
import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

from nanobot.config.paths import get_media_dir
from nanobot.utils.helpers import detect_image_mime, detect_video_mime, ensure_dir

_DATA_IMAGE_RE = re.compile(r"^data:(image/[A-Za-z0-9.+-]+);base64,(.*)$", re.DOTALL)
_MIME_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

class ArtifactError(ValueError):
    """Raised when an artifact cannot be safely decoded or stored."""


def decode_image_data_url(data_url: str) -> tuple[bytes, str]:
    """Decode a base64 image data URL and return ``(bytes, mime)``."""
    match = _DATA_IMAGE_RE.match(data_url.strip())
    if match is None:
        raise ArtifactError("expected a base64 image data URL")

    declared_mime, encoded = match.groups()
    try:
        raw = base64.b64decode(encoded, validate=True)
    except binascii.Error as exc:
        raise ArtifactError("invalid base64 image payload") from exc

    detected_mime = detect_image_mime(raw)
    if detected_mime is None:
        raise ArtifactError("unsupported or unrecognized image data")
    if declared_mime != detected_mime:
        declared_mime = detected_mime
    return raw, declared_mime


def _safe_relative_dir(save_dir: str) -> Path:
    normalized = save_dir.replace("\\", "/").strip("/")
    if not normalized:
        raise ArtifactError("save_dir must not be empty")
    rel = PurePosixPath(normalized)
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        raise ArtifactError("save_dir must be a safe relative path")
    return Path(*rel.parts)


def _artifact_root(save_dir: str) -> Path:
    media_root = get_media_dir().resolve()
    root = (media_root / _safe_relative_dir(save_dir)).resolve()
    try:
        root.relative_to(media_root)
    except ValueError as exc:
        raise ArtifactError("artifact directory escapes media root") from exc
    return root


def store_generated_image_artifact(
    data_url: str,
    *,
    prompt: str,
    model: str,
    source_images: list[str] | None = None,
    save_dir: str = "generated",
    provider: str = "openrouter",
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Persist a generated image and sidecar metadata under the media root."""
    raw, mime = decode_image_data_url(data_url)
    ext = _MIME_EXTENSIONS.get(mime)
    if ext is None:
        raise ArtifactError(f"unsupported image MIME type: {mime}")

    now = created_at or datetime.now().astimezone()
    day_dir = ensure_dir(_artifact_root(save_dir) / now.strftime("%Y-%m-%d"))
    artifact_id = f"img_{uuid.uuid4().hex[:12]}"
    image_path = day_dir / f"{artifact_id}{ext}"
    metadata_path = day_dir / f"{artifact_id}.json"

    image_path.write_bytes(raw)
    metadata: dict[str, Any] = {
        "id": artifact_id,
        "path": str(image_path),
        "mime": mime,
        "prompt": prompt,
        "model": model,
        "provider": provider,
        "source_images": list(source_images or []),
        "created_at": now.isoformat(),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def generated_image_tool_result(artifacts: list[dict[str, Any]]) -> str:
    """Return the compact structured result exposed to the LLM."""
    return json.dumps(
        {
            "artifacts": artifacts,
            "next_step": (
                "Use these artifact paths as reference_images for follow-up edits. "
                "Call the message tool with the artifact paths in the media parameter "
                "to deliver the images to the user. Keep raw paths internal unless the "
                "user asks for debug details."
            ),
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Video artifacts
# ---------------------------------------------------------------------------

_VIDEO_MIME_EXTENSIONS = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "video/x-matroska": ".mkv",
    "video/x-flv": ".flv",
}
_DOWNLOAD_CHUNK_BYTES = 256 * 1024


def _video_extension_from_url(url: str) -> str | None:
    path = url.split("?", 1)[0].split("#", 1)[0]
    ext = Path(path).suffix.lower()
    return ext if ext in _VIDEO_MIME_EXTENSIONS.values() else None


async def store_generated_video_artifact(
    source: str | Path,
    *,
    prompt: str,
    model: str,
    mode: str = "text",
    seconds: str | None = None,
    size: str | None = None,
    aspect_ratio: str | None = None,
    save_dir: str = "generated_videos",
    provider: str = "openai",
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Persist a generated video (URL or local file) under the media root.

    Remote sources are streamed to disk in chunks to keep memory usage flat
    (the deployment target runs under a tight memory cgroup).
    """
    now = created_at or datetime.now().astimezone()
    day_dir = ensure_dir(_artifact_root(save_dir) / now.strftime("%Y-%m-%d"))
    artifact_id = f"vid_{uuid.uuid4().hex[:12]}"

    source_str = str(source)
    mime: str | None = None
    if source_str.startswith(("http://", "https://")):
        ext = _video_extension_from_url(source_str) or ".mp4"
        final_path = day_dir / f"{artifact_id}{ext}"
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, read=120.0), follow_redirects=True
        ) as client:
            async with client.stream("GET", source_str) as response:
                if response.status_code != 200:
                    detail = (await response.aread())[:200]
                    raise ArtifactError(
                        f"failed to download generated video "
                        f"(HTTP {response.status_code}): {detail!r}"
                    )
                with final_path.open("wb") as fh:
                    async for chunk in response.aiter_bytes(_DOWNLOAD_CHUNK_BYTES):
                        fh.write(chunk)
        head = final_path.open("rb").read(32)
        mime = detect_video_mime(head, final_path.name)
    else:
        local = Path(source_str).expanduser()
        if not local.is_file():
            raise ArtifactError(f"generated video source not found: {local}")
        head = local.open("rb").read(32)
        mime = detect_video_mime(head, local.name)
        ext = _VIDEO_MIME_EXTENSIONS.get(mime or "", local.suffix.lower() or ".mp4")
        final_path = day_dir / f"{artifact_id}{ext}"
        if local.resolve() != final_path.resolve():
            shutil.copy2(local, final_path)

    metadata: dict[str, Any] = {
        "id": artifact_id,
        "path": str(final_path),
        "mime": mime or "video/mp4",
        "prompt": prompt,
        "model": model,
        "provider": provider,
        "mode": mode,
        "seconds": seconds,
        "size": size,
        "aspect_ratio": aspect_ratio,
        "created_at": now.isoformat(),
    }
    metadata_path = day_dir / f"{artifact_id}.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def generated_video_tool_result(artifacts: list[dict[str, Any]]) -> str:
    """Return the compact structured result exposed to the LLM."""
    return json.dumps(
        {
            "artifacts": artifacts,
            "next_step": (
                "Call the message tool with the artifact paths in the media parameter "
                "to deliver the videos to the user. Video generation is slow; tell the "
                "user the video is ready when delivering it. Keep raw paths internal "
                "unless the user asks for debug details."
            ),
        },
        ensure_ascii=False,
    )
