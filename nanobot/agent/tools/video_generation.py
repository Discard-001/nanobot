"""Video generation tool."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import Field

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import (
    ArraySchema,
    StringSchema,
    tool_parameters_schema,
)
from nanobot.config.paths import get_media_dir
from nanobot.config.schema import Base
from nanobot.providers.image_generation import image_path_to_data_url
from nanobot.providers.video_generation import (
    VideoGenerationError,
    VideoGenerationProvider,
    get_video_gen_provider,
)
from nanobot.security.workspace_access import current_tool_workspace
from nanobot.security.workspace_policy import WorkspaceBoundaryError, resolve_allowed_path
from nanobot.utils.artifacts import (
    ArtifactError,
    generated_video_tool_result,
    store_generated_video_artifact,
)
from nanobot.utils.helpers import detect_audio_mime

if TYPE_CHECKING:
    from nanobot.config.schema import ProviderConfig

_SUPPORTED_ASPECT_RATIOS = {"21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}


class VideoGenerationToolConfig(Base):
    """Video generation tool configuration."""

    enabled: bool = False
    provider: str = "openai"
    model: str = "agnes-video-2.5-flash"
    default_seconds: str = "5"
    # None = omit (let the API apply its default; agnes-video-2.5-flash only
    # accepts "720P", agnes-video-2.5 also accepts "2K")
    default_size: str | None = None
    default_aspect_ratio: str = "16:9"
    poll_interval_s: float = Field(default=2.0, ge=0.5)
    poll_timeout_s: float = Field(default=600.0, ge=30.0)
    create_retries: int = Field(default=3, ge=1, le=6)
    save_dir: str = "generated_videos"
    max_videos_per_turn: int = Field(default=1, ge=1, le=2)


@tool_parameters(
    tool_parameters_schema(
        prompt=StringSchema(
            "Detailed video generation prompt. Describe subject and scene, "
            "action and changes, camera movement, visual style, and pacing. "
            "In reference mode use <Picture N>/<Audio N> to refer to media.",
            min_length=1,
        ),
        seconds=StringSchema(
            'Video duration as a string, e.g. "4" to "12".',
        ),
        aspect_ratio=StringSchema(
            "Output aspect ratio: 21:9, 16:9, 4:3, 1:1, 3:4, 9:16.",
        ),
        first_frame=StringSchema(
            "First-frame image (local path or public http(s) URL) for keyframe "
            "mode. Provide first_frame and/or last_frame.",
        ),
        last_frame=StringSchema(
            "Last-frame image (local path or public http(s) URL) for keyframe mode.",
        ),
        reference_images=ArraySchema(
            StringSchema(
                "Reference image: local path or public http(s) URL. Use <Picture N> "
                "in the prompt to refer to them.",
            ),
            description="Reference images for reference mode (up to 5 for flash models).",
        ),
        reference_audios=ArraySchema(
            StringSchema(
                "Reference audio: local path or public http(s) URL. Use <Audio N> "
                "in the prompt to refer to them.",
            ),
            description="Reference audios for reference mode (up to 3 for flash models).",
        ),
        required=["prompt"],
    )
)
class VideoGenerationTool(Tool):
    """Generate videos through the configured video provider."""

    config_key = "video_generation"

    @classmethod
    def config_cls(cls):
        return VideoGenerationToolConfig

    @classmethod
    def enabled(cls, ctx: Any) -> bool:
        return ctx.config.video_generation.enabled

    @classmethod
    def create(cls, ctx: Any) -> Tool:
        return cls(
            workspace=ctx.workspace,
            config=ctx.config.video_generation,
            provider_configs=ctx.video_generation_provider_configs,
        )

    def __init__(
        self,
        *,
        workspace: str | Path,
        config: VideoGenerationToolConfig,
        provider_configs: dict[str, ProviderConfig] | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser()
        self.config = config
        self.provider_configs = dict(provider_configs or {})

    @property
    def name(self) -> str:
        return "generate_video"

    @property
    def description(self) -> str:
        return (
            "Generate a video from a text prompt, optional first/last keyframes, "
            "or reference images/audios. Modes: text (prompt only), keyframe "
            "(first_frame/last_frame), reference (reference_images/reference_audios). "
            "Returns artifact ids and local paths. Generation is slow (usually "
            "1-5 minutes); the tool polls until completion."
        )

    def _provider_client(self) -> VideoGenerationProvider | None:
        provider = self.provider_configs.get(self.config.provider)
        cls = get_video_gen_provider(self.config.provider)
        if cls is None:
            return None
        return cls(
            api_key=provider.api_key if provider else None,
            api_base=provider.api_base if provider else None,
            extra_headers=provider.extra_headers if provider else None,
            extra_body=provider.extra_body if provider else None,
        )

    def _resolve_media_path(self, value: str) -> Path:
        access = current_tool_workspace(self.workspace, restrict_to_workspace=True)
        workspace = access.project_path or self.workspace
        try:
            resolved = resolve_allowed_path(
                value,
                workspace=workspace,
                allowed_root=access.allowed_root,
                extra_allowed_roots=[get_media_dir()] if access.allowed_root is not None else None,
                strict=True,
            )
        except WorkspaceBoundaryError as exc:
            raise VideoGenerationError(
                "media references must be inside the workspace or nanobot media directory"
            ) from exc
        except OSError as exc:
            raise VideoGenerationError(f"media reference not found: {value}") from exc
        if not resolved.is_file():
            raise VideoGenerationError(f"media reference is not a file: {value}")
        return resolved

    def _to_media_reference(self, value: str, *, audio: bool = False) -> str:
        """Convert a local path or URL into an API-acceptable reference.

        http(s) URLs pass through; local files are inlined as base64 data URLs.
        """
        value = value.strip()
        if value.startswith(("http://", "https://", "data:")):
            return value
        resolved = self._resolve_media_path(value)
        if audio:
            raw = resolved.read_bytes()
            mime = detect_audio_mime(raw, resolved.name)
            if mime is None:
                raise VideoGenerationError(f"unsupported audio reference: {value}")
            encoded = base64.b64encode(raw).decode("ascii")
            return f"data:{mime};base64,{encoded}"
        return image_path_to_data_url(resolved)

    async def execute(
        self,
        prompt: str,
        seconds: str | None = None,
        aspect_ratio: str | None = None,
        first_frame: str | None = None,
        last_frame: str | None = None,
        reference_images: list[str] | None = None,
        reference_audios: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        client = self._provider_client()
        if client is None:
            return f"Error: unsupported video generation provider '{self.config.provider}'"

        ratio = (aspect_ratio or self.config.default_aspect_ratio or "").strip()
        if ratio and ratio not in _SUPPORTED_ASPECT_RATIOS:
            return (
                f"Error: unsupported aspect_ratio {ratio!r}; "
                f"supported: {sorted(_SUPPORTED_ASPECT_RATIOS)}"
            )

        # Resolve mode from the provided media.
        ff = first_frame.strip() if isinstance(first_frame, str) else ""
        lf = last_frame.strip() if isinstance(last_frame, str) else ""
        images = [v for v in (reference_images or []) if v and v.strip()]
        audios = [v for v in (reference_audios or []) if v and v.strip()]

        if ff or lf:
            mode = "keyframe"
            if images or audios:
                return "Error: keyframe mode does not accept reference_images/reference_audios"
        elif images or audios:
            mode = "reference"
        else:
            mode = "text"

        try:
            if mode == "keyframe":
                first_ref = self._to_media_reference(ff) if ff else None
                last_ref = self._to_media_reference(lf) if lf else None
                image_refs = audio_refs = None
            elif mode == "reference":
                first_ref = last_ref = None
                image_refs = [self._to_media_reference(v) for v in images]
                audio_refs = [self._to_media_reference(v, audio=True) for v in audios]
            else:
                first_ref = last_ref = image_refs = audio_refs = None

            response = await client.generate(
                prompt=prompt,
                model=self.config.model,
                mode=mode,
                seconds=seconds or self.config.default_seconds,
                size=self.config.default_size,
                aspect_ratio=ratio or None,
                first_frame=first_ref,
                last_frame=last_ref,
                images=image_refs,
                audios=audio_refs,
                poll_interval=self.config.poll_interval_s,
                poll_timeout=self.config.poll_timeout_s,
                create_retries=self.config.create_retries,
            )

            artifact = await store_generated_video_artifact(
                response.video_url,
                prompt=prompt,
                model=self.config.model,
                mode=mode,
                seconds=seconds or self.config.default_seconds,
                size=self.config.default_size,
                aspect_ratio=ratio or None,
                save_dir=self.config.save_dir,
                provider=self.config.provider,
            )
            return generated_video_tool_result([artifact])
        except (ArtifactError, VideoGenerationError, OSError) as exc:
            return f"Error: {exc}"
