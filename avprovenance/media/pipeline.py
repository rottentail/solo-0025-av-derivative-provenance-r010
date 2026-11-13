"""Route a normalized plan to the matching media implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from ..errors import UnsupportedMedia
from ..models import MediaDescriptor
from .audio_ops import render_audio
from .errors import MediaOperationFailure
from .inspector import inspect_media, inspect_media_file
from .video_ops import render_video, render_video_to_stream


@dataclass(frozen=True)
class RenderOutcome:
    data: bytes
    descriptor: MediaDescriptor
    operation_types: list[str]


@dataclass(frozen=True)
class StreamRenderOutcome:
    descriptor: MediaDescriptor
    operation_types: list[str]


def apply_media_operations(
    primary_bytes: bytes,
    operations: list[dict[str, Any]],
    secondary_bytes: bytes | None,
) -> RenderOutcome:
    primary = inspect_media(primary_bytes)
    if secondary_bytes is not None:
        secondary = inspect_media(secondary_bytes)
        if secondary.kind != primary.kind:
            raise MediaOperationFailure(
                "MEDIA_KIND_MISMATCH",
                "Primary and secondary media must share a media kind.",
            )

    if primary.kind == "audio":
        output = render_audio(primary_bytes, operations, secondary_bytes)
    elif primary.kind == "video":
        output = render_video(primary_bytes, operations, secondary_bytes)
    else:
        raise UnsupportedMedia("The primary media kind has no renderer.")

    descriptor = inspect_media(output)
    if descriptor.kind != primary.kind:
        raise UnsupportedMedia("The renderer changed the media kind unexpectedly.")
    return RenderOutcome(
        data=output,
        descriptor=descriptor,
        operation_types=[str(operation["type"]) for operation in operations],
    )


def apply_media_operations_to_target(
    primary_path: Path,
    operations: list[dict[str, Any]],
    target: BinaryIO,
    secondary_path: Path | None = None,
) -> StreamRenderOutcome:
    """Render a persisted source into *target* with bounded working memory.

    Video sources are streamed frame by frame.  Audio sources are decoded
    fully because the PCM canvas is bounded by the same upload budget.
    """

    primary = inspect_media_file(primary_path)
    if secondary_path is not None:
        secondary = inspect_media_file(secondary_path)
        if secondary.kind != primary.kind:
            raise MediaOperationFailure(
                "MEDIA_KIND_MISMATCH",
                "Primary and secondary media must share a media kind.",
            )

    if primary.kind == "audio":
        primary_bytes = primary_path.read_bytes()
        secondary_bytes = secondary_path.read_bytes() if secondary_path else None
        output = render_audio(primary_bytes, operations, secondary_bytes)
        target.write(output)
        descriptor = inspect_media(output)
    elif primary.kind == "video":
        if secondary_path is not None:
            raise MediaOperationFailure(
                "VIDEO_SECONDARY_UNSUPPORTED",
                "The current video operation set accepts one source.",
            )
        with open(primary_path, "rb") as source:
            descriptor, _frame_count = render_video_to_stream(
                source,
                operations,
                target,
            )
    else:
        raise UnsupportedMedia("The primary media kind has no renderer.")

    if descriptor.kind != primary.kind:
        raise UnsupportedMedia("The renderer changed the media kind unexpectedly.")
    return StreamRenderOutcome(
        descriptor=descriptor,
        operation_types=[str(operation["type"]) for operation in operations],
    )
