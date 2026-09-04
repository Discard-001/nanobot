***

name: video-generation
description: Generate videos from text prompts, keyframes, or reference images/audios.
--------------------------------------------------------------------------------------

# Video Generation

Use the `generate_video` tool when the user asks you to create, generate, or animate a video or short clip.

If the `generate_video` tool is not available in the current tool list, tell the user that video generation is not enabled for this nanobot instance.

## When To Use

- Text-to-video: call `generate_video` with a concrete `prompt` (mode is inferred automatically).

- First/last frame control: pass `first_frame` and/or `last_frame` (local image path or public URL) for keyframe mode.

- Reference-guided: pass `reference_images` (up to 5) and/or `reference_audios` (up to 3); refer to them in the prompt as `<Picture 1>`, `<Audio 1>`, etc.

- After generating, call the `message` tool with the artifact path in the `media` parameter to deliver the video to the user.

## Prompt Rules

Describe in this order for stable results:

1. Subject and scene (who/what, environment, time of day).
2. Action and changes (how the subject moves, how the scene evolves).
3. Camera language (push, pull, pan, tilt, tracking, fixed, shot size).
4. Visual style (lighting, color, material, realism, mood).
5. Sound and pacing when relevant.

## Parameters

- `seconds`: duration "4" to "12" (string). Platform minimum is 4; default "4" (fastest generation).
- `aspect_ratio`: 21:9, 16:9, 4:3, 1:1, 3:4, 9:16.
- Generation is slow (usually 1–5 minutes). While waiting is handled by the tool, tell the user upfront that video generation takes a while.

## Artifact Rules

- The result is a video artifact (mp4) with a local path under the media directory.

- Deliver via the `message` tool's `media` parameter.

- Reuse generated image artifacts as `first_frame`/`reference_images` inputs for video continuation workflows.

