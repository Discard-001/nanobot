from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from nanobot.config.loader import set_config_path
from nanobot.utils.artifacts import (
    ArtifactError,
    decode_image_data_url,
    store_generated_image_artifact,
    store_generated_video_artifact,
)

PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def test_decode_image_data_url_validates_image_payload() -> None:
    raw, mime = decode_image_data_url(PNG_DATA_URL)

    assert raw.startswith(b"\x89PNG")
    assert mime == "image/png"

    with pytest.raises(ArtifactError):
        decode_image_data_url("data:image/png;base64,not-base64")


def test_store_generated_image_artifact_writes_image_and_sidecar(tmp_path: Path) -> None:
    set_config_path(tmp_path / "config.json")
    created_at = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)

    artifact = store_generated_image_artifact(
        PNG_DATA_URL,
        prompt="draw a tiny pixel",
        model="openai/gpt-5.4-image-2",
        source_images=["/tmp/ref.png"],
        save_dir="generated",
        created_at=created_at,
    )

    image_path = Path(artifact["path"])
    assert image_path.is_file()
    assert image_path.parent == tmp_path / "media" / "generated" / "2026-05-08"
    assert artifact["id"].startswith("img_")
    assert artifact["mime"] == "image/png"

    sidecar = image_path.with_suffix(".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metadata["path"] == str(image_path)
    assert metadata["source_images"] == ["/tmp/ref.png"]


def test_store_generated_image_artifact_rejects_unsafe_save_dir(tmp_path: Path) -> None:
    set_config_path(tmp_path / "config.json")

    with pytest.raises(ArtifactError):
        store_generated_image_artifact(
            PNG_DATA_URL,
            prompt="x",
            model="m",
            save_dir="../outside",
        )


MP4_BYTES = b"\x00\x00\x00\x18ftypmp42isom\x00\x00\x00\x08free"


@pytest.mark.asyncio
async def test_store_generated_video_artifact_local_source(tmp_path: Path) -> None:
    set_config_path(tmp_path / "config.json")
    created_at = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
    src = tmp_path / "input.mp4"
    src.write_bytes(MP4_BYTES)

    artifact = await store_generated_video_artifact(
        src,
        prompt="a cat running",
        model="agnes-video-2.5-flash",
        mode="text",
        seconds="5",
        size="720P",
        aspect_ratio="16:9",
        save_dir="generated_videos",
        provider="openai",
        created_at=created_at,
    )

    video_path = Path(artifact["path"])
    assert video_path.is_file()
    assert video_path.read_bytes() == MP4_BYTES
    assert video_path.parent == tmp_path / "media" / "generated_videos" / "2026-09-04"
    assert artifact["id"].startswith("vid_")
    assert artifact["mime"] == "video/mp4"
    assert artifact["mode"] == "text"

    sidecar = video_path.with_suffix(".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metadata["model"] == "agnes-video-2.5-flash"
    assert metadata["seconds"] == "5"


@pytest.mark.asyncio
async def test_store_generated_video_artifact_downloads_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_config_path(tmp_path / "config.json")

    class _FakeStream:
        def __init__(self, response: "_FakeResponse") -> None:
            self._response = response

        async def __aenter__(self):
            return self._response

        async def __aexit__(self, *exc):
            return False

    class _FakeResponse:
        def __init__(self, status: int = 200) -> None:
            self.status_code = status

        async def aiter_bytes(self, chunk_size):
            for i in range(0, len(MP4_BYTES), 8):
                yield MP4_BYTES[i : i + 8]

        async def aread(self):
            return b"error body"

    class _FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def stream(self, method: str, url: str):
            return _FakeStream(_FakeResponse())

    monkeypatch.setattr(
        "nanobot.utils.artifacts.httpx.AsyncClient", _FakeClient
    )

    artifact = await store_generated_video_artifact(
        "https://cdn.example/video.mp4",
        prompt="p",
        model="m",
        save_dir="generated_videos",
    )

    video_path = Path(artifact["path"])
    assert video_path.is_file()
    assert video_path.read_bytes() == MP4_BYTES
    assert video_path.name.startswith("vid_")
    assert video_path.suffix == ".mp4"


@pytest.mark.asyncio
async def test_store_generated_video_artifact_missing_local_source(tmp_path: Path) -> None:
    set_config_path(tmp_path / "config.json")

    with pytest.raises(ArtifactError, match="not found"):
        await store_generated_video_artifact(
            tmp_path / "missing.mp4", prompt="p", model="m",
        )
