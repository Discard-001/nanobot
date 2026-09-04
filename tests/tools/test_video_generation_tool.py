from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent.tools.video_generation import VideoGenerationTool
from nanobot.config.loader import set_config_path
from nanobot.config.schema import ProviderConfig, VideoGenerationToolConfig
from nanobot.providers.video_generation import GeneratedVideoResponse

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
WAV_BYTES = b"RIFF\x24\x00\x00\x00WAVEfmt "
VIDEO_URL = "https://cdn.example/generated.mp4"


class FakeVideoClient:
    instances: list["FakeVideoClient"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        self.instances.append(self)

    async def generate(self, **kwargs: Any) -> GeneratedVideoResponse:
        self.calls.append(kwargs)
        return GeneratedVideoResponse(
            video_url=VIDEO_URL, remote_url=VIDEO_URL, content="", raw={},
        )


def _make_tool(tmp_path: Path) -> VideoGenerationTool:
    return VideoGenerationTool(
        workspace=tmp_path,
        config=VideoGenerationToolConfig(enabled=True),
        provider_configs={"openai": ProviderConfig(api_key="sk-test")},
    )


@pytest.fixture(autouse=True)
def _stub_artifact_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Replace the tool's artifact store so no network download happens."""
    mp4 = tmp_path / "sample.mp4"
    mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42isom")
    monkeypatch.setattr(
        "nanobot.agent.tools.video_generation.store_generated_video_artifact",
        _fake_store(mp4),
    )


@pytest.mark.asyncio
async def test_text_mode_generates_and_stores_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_config_path(tmp_path / "config.json")
    FakeVideoClient.instances = []
    monkeypatch.setattr(
        "nanobot.agent.tools.video_generation.get_video_gen_provider",
        lambda name: FakeVideoClient if name == "openai" else None,
    )
    tool = _make_tool(tmp_path)

    result = await tool.execute(prompt="a cat running", seconds="6")

    payload = json.loads(result)
    artifact = payload["artifacts"][0]
    assert artifact["mode"] == "text"
    assert artifact["seconds"] == "6"
    assert artifact["model"] == "agnes-video-2.5-flash"

    fake = FakeVideoClient.instances[0]
    assert fake.kwargs["api_key"] == "sk-test"
    call = fake.calls[0]
    assert call["mode"] == "text"
    assert call["seconds"] == "6"
    assert call["model"] == "agnes-video-2.5-flash"
    assert call["first_frame"] is None
    assert call["images"] is None


@pytest.mark.asyncio
async def test_keyframe_mode_inlines_local_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_config_path(tmp_path / "config.json")
    FakeVideoClient.instances = []
    monkeypatch.setattr(
        "nanobot.agent.tools.video_generation.get_video_gen_provider",
        lambda name: FakeVideoClient,
    )
    tool = _make_tool(tmp_path)
    frame = tmp_path / "frame.png"
    frame.write_bytes(PNG_BYTES)

    result = await tool.execute(prompt="morph", first_frame="frame.png")
    payload = json.loads(result)
    assert payload["artifacts"][0]["mode"] == "keyframe"

    call = FakeVideoClient.instances[0].calls[0]
    assert call["mode"] == "keyframe"
    assert call["first_frame"].startswith("data:image/png;base64,")
    assert call["last_frame"] is None


@pytest.mark.asyncio
async def test_reference_mode_inlines_image_and_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_config_path(tmp_path / "config.json")
    FakeVideoClient.instances = []
    monkeypatch.setattr(
        "nanobot.agent.tools.video_generation.get_video_gen_provider",
        lambda name: FakeVideoClient,
    )
    tool = _make_tool(tmp_path)
    img = tmp_path / "ref.png"
    img.write_bytes(PNG_BYTES)
    audio = tmp_path / "ref.wav"
    audio.write_bytes(WAV_BYTES)

    result = await tool.execute(
        prompt="<Picture 1> with <Audio 1>",
        reference_images=["ref.png"],
        reference_audios=["ref.wav"],
    )
    payload = json.loads(result)
    assert payload["artifacts"][0]["mode"] == "reference"

    call = FakeVideoClient.instances[0].calls[0]
    assert call["mode"] == "reference"
    assert call["images"] == [PNG_DATA_URL]
    assert call["audios"] == [
        "data:audio/wav;base64," + base64.b64encode(WAV_BYTES).decode("ascii")
    ]


@pytest.mark.asyncio
async def test_http_urls_pass_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_config_path(tmp_path / "config.json")
    FakeVideoClient.instances = []
    monkeypatch.setattr(
        "nanobot.agent.tools.video_generation.get_video_gen_provider",
        lambda name: FakeVideoClient,
    )
    tool = _make_tool(tmp_path)

    await tool.execute(
        prompt="p",
        reference_images=["https://img.example/a.png"],
    )
    call = FakeVideoClient.instances[0].calls[0]
    assert call["images"] == ["https://img.example/a.png"]


@pytest.mark.asyncio
async def test_keyframe_rejects_reference_media(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    result = await tool.execute(
        prompt="p",
        first_frame="https://img.example/a.png",
        reference_images=["https://img.example/b.png"],
    )
    assert result.startswith("Error:")
    assert "keyframe mode does not accept" in result


@pytest.mark.asyncio
async def test_invalid_aspect_ratio(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    result = await tool.execute(prompt="p", aspect_ratio="2:1")
    assert result.startswith("Error:")
    assert "aspect_ratio" in result


@pytest.mark.asyncio
async def test_unsupported_provider(tmp_path: Path) -> None:
    tool = VideoGenerationTool(
        workspace=tmp_path,
        config=VideoGenerationToolConfig(enabled=True, provider="nope"),
        provider_configs={},
    )
    result = await tool.execute(prompt="p")
    assert result.startswith("Error:")
    assert "unsupported video generation provider" in result


def _fake_store(source: Path):
    from nanobot.utils.artifacts import ArtifactError

    async def _store(src, **kwargs):
        if str(src) != VIDEO_URL:
            raise ArtifactError(f"unexpected source {src}")
        return {
            "id": "vid_test", "path": str(source), "mime": "video/mp4",
            "model": kwargs.get("model"), "mode": kwargs.get("mode"),
            "seconds": kwargs.get("seconds"),
        }

    return _store
