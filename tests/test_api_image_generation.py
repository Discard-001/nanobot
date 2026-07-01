"""Tests for image generation HTTP API routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.api.server import create_app
from nanobot.config.loader import set_config_path
from nanobot.config.schema import ImageGenerationToolConfig, ProviderConfig, ToolsConfig
from nanobot.providers.image_generation import GeneratedImageResponse

try:
    from aiohttp.test_utils import TestClient, TestServer

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class FakeImageClient:
    instances: list["FakeImageClient"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        self.instances.append(self)

    async def generate(self, **kwargs: Any) -> GeneratedImageResponse:
        self.calls.append(kwargs)
        return GeneratedImageResponse(images=[PNG_DATA_URL], content="", raw={})


@pytest.fixture
def fake_agent_loop(tmp_path: Path) -> MagicMock:
    set_config_path(tmp_path / "config.json")
    agent = MagicMock()
    agent.process_direct = AsyncMock(return_value="mock response")
    agent._connect_mcp = AsyncMock()
    agent.close_mcp = AsyncMock()
    agent.workspace = tmp_path
    agent.tools_config = ToolsConfig(
        image_generation=ImageGenerationToolConfig(
            enabled=True,
            provider="openrouter",
            model="openai/gpt-image-1",
            default_aspect_ratio="1:1",
            default_image_size="1024x1024",
            max_images_per_turn=2,
        )
    )
    agent._image_generation_provider_configs = {
        "openrouter": ProviderConfig(api_key="sk-or-test")
    }
    return agent


@pytest.fixture
def app(fake_agent_loop: MagicMock):
    return create_app(fake_agent_loop, model_name="chat-model", request_timeout=10.0)


@pytest.fixture
async def aiohttp_client():
    clients: list[TestClient] = []

    async def _make_client(app):
        client = TestClient(TestServer(app))
        await client.start_server()
        clients.append(client)
        return client

    try:
        yield _make_client
    finally:
        for client in clients:
            await client.close()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_openai_images_generations_returns_b64_json(
    aiohttp_client,
    app,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeImageClient.instances = []
    monkeypatch.setattr(
        "nanobot.agent.tools.image_generation.get_image_gen_provider",
        lambda name: FakeImageClient if name == "openrouter" else None,
    )
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/images/generations",
        json={
            "model": "openai/gpt-image-1",
            "prompt": "draw a small robot",
            "n": 2,
            "size": "1024x1024",
            "response_format": "b64_json",
        },
    )

    assert resp.status == 200
    body = await resp.json()
    assert isinstance(body["created"], int)
    assert len(body["data"]) == 2
    assert body["data"][0]["b64_json"].startswith("iVBOR")
    assert "url" not in body["data"][0]

    fake = FakeImageClient.instances[0]
    assert len(fake.calls) == 2
    assert fake.calls[0]["prompt"] == "draw a small robot"
    assert fake.calls[0]["model"] == "openai/gpt-image-1"
    assert fake.calls[0]["image_size"] == "1024x1024"


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_native_image_generate_returns_artifacts(
    aiohttp_client,
    app,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeImageClient.instances = []
    monkeypatch.setattr(
        "nanobot.agent.tools.image_generation.get_image_gen_provider",
        lambda name: FakeImageClient if name == "openrouter" else None,
    )
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/image/generate",
        json={
            "prompt": "draw a poster",
            "count": 1,
            "aspect_ratio": "16:9",
            "image_size": "2K",
        },
    )

    assert resp.status == 200
    body = await resp.json()
    assert len(body["artifacts"]) == 1
    artifact = body["artifacts"][0]
    assert Path(artifact["path"]).is_file()
    assert artifact["provider"] == "openrouter"
    assert artifact["model"] == "openai/gpt-image-1"

    fake = FakeImageClient.instances[0]
    assert fake.calls[0]["aspect_ratio"] == "16:9"
    assert fake.calls[0]["image_size"] == "2K"


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_image_generation_disabled_returns_400(aiohttp_client, fake_agent_loop) -> None:
    fake_agent_loop.tools_config.image_generation.enabled = False
    client = await aiohttp_client(create_app(fake_agent_loop, model_name="chat-model"))

    resp = await client.post(
        "/v1/images/generations",
        json={"prompt": "draw"},
    )

    assert resp.status == 400
    body = await resp.json()
    assert "image generation is not enabled" in body["error"]["message"].lower()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_image_generation_count_limit_returns_400(aiohttp_client, app) -> None:
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/image/generate",
        json={"prompt": "draw", "count": 3},
    )

    assert resp.status == 400
    body = await resp.json()
    assert "maximagesperturn" in body["error"]["message"].replace(".", "").lower()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_image_generation_model_mismatch_returns_400(aiohttp_client, app) -> None:
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/images/generations",
        json={"model": "other-image-model", "prompt": "draw"},
    )

    assert resp.status == 400
    body = await resp.json()
    assert "openai/gpt-image-1" in body["error"]["message"]
