"""Planar Y4M operations that preserve deterministic frame layout.

Large sources are rendered as a stream.  Geometry and operation bounds are
resolved against a validated header first, one frame is read at a time, and
the transformed bytes are written straight to the output sink.  Working
memory therefore depends on one frame, never on the clip length.  The clip
based helpers remain for callers that already hold decoded frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, BinaryIO

from ..models import MediaDescriptor
from .errors import MediaOperationFailure
from .y4m import (
    Y4mClip,
    Y4mFrame,
    Y4mStreamInfo,
    descriptor_for_y4m,
    encode_y4m_header,
    iter_y4m_frames,
    scan_y4m,
)


def trim_video(clip: Y4mClip, operation: dict[str, Any]) -> Y4mClip:
    start = int(operation["start_frame"])
    count = int(operation["frame_count"])
    if start < 0 or start >= clip.frame_count:
        raise MediaOperationFailure(
            "VIDEO_TRIM_OUT_OF_RANGE",
            "The requested video trim starts outside the source.",
        )
    end = min(clip.frame_count, start + max(1, count))
    frames = clip.frames[start:end]
    if not frames:
        raise MediaOperationFailure(
            "VIDEO_TRIM_EMPTY",
            "The requested video trim produces no frames.",
        )
    return Y4mClip(
        width=clip.width,
        height=clip.height,
        rate_numerator=clip.rate_numerator,
        rate_denominator=clip.rate_denominator,
        interlace=clip.interlace,
        pixel_format=clip.pixel_format,
        frames=frames,
    )


def _crop_plane(
    plane: bytes,
    plane_width: int,
    x: int,
    y: int,
    width: int,
    height: int,
) -> bytes:
    rows = []
    for row_index in range(y, y + height):
        start = row_index * plane_width + x
        rows.append(plane[start : start + width])
    return b"".join(rows)


def crop_video(clip: Y4mClip, operation: dict[str, Any]) -> Y4mClip:
    x = int(operation["x"])
    y = int(operation["y"])
    width = int(operation["width"])
    height = int(operation["height"])
    if any(value % 2 for value in (x, y, width, height)):
        raise MediaOperationFailure(
            "VIDEO_CROP_ALIGNMENT",
            "Video crop geometry must align to even pixels.",
        )
    if width < 2 or height < 2:
        raise MediaOperationFailure(
            "VIDEO_CROP_TOO_SMALL",
            "Crop width and height must be at least two pixels.",
        )
    if x + width > clip.width or y + height > clip.height:
        raise MediaOperationFailure(
            "VIDEO_CROP_OUT_OF_RANGE",
            "Crop geometry exceeds the source frame.",
        )

    chroma_x = x // 2
    chroma_y = y // 2
    chroma_width = width // 2
    chroma_height = height // 2
    frames = []
    for frame in clip.frames:
        frames.append(
            Y4mFrame(
                y_plane=_crop_plane(
                    frame.y_plane, clip.width, x, y, width, height
                ),
                u_plane=_crop_plane(
                    frame.u_plane,
                    clip.chroma_width,
                    chroma_x,
                    chroma_y,
                    chroma_width,
                    chroma_height,
                ),
                v_plane=_crop_plane(
                    frame.v_plane,
                    clip.chroma_width,
                    chroma_x,
                    chroma_y,
                    chroma_width,
                    chroma_height,
                ),
            )
        )
    return Y4mClip(
        width=width,
        height=height,
        rate_numerator=clip.rate_numerator,
        rate_denominator=clip.rate_denominator,
        interlace=clip.interlace,
        pixel_format=clip.pixel_format,
        frames=tuple(frames),
    )


def offset_luma(clip: Y4mClip, operation: dict[str, Any]) -> Y4mClip:
    amount = int(operation["amount"])
    frames = []
    for frame in clip.frames:
        adjusted = bytes(
            max(0, min(255, value + amount))
            for value in frame.y_plane
        )
        frames.append(
            Y4mFrame(
                y_plane=adjusted,
                u_plane=frame.u_plane,
                v_plane=frame.v_plane,
            )
        )
    return Y4mClip(
        width=clip.width,
        height=clip.height,
        rate_numerator=clip.rate_numerator,
        rate_denominator=clip.rate_denominator,
        interlace=clip.interlace,
        pixel_format=clip.pixel_format,
        frames=tuple(frames),
    )


def _mirror_plane(plane: bytes, width: int, height: int) -> bytes:
    rows = []
    for row_index in range(height):
        start = row_index * width
        row = plane[start : start + width]
        rows.append(row[::-1])
    return b"".join(rows)


def mirror_video(clip: Y4mClip, operation: dict[str, Any]) -> Y4mClip:
    del operation
    frames = tuple(
        Y4mFrame(
            y_plane=_mirror_plane(frame.y_plane, clip.width, clip.height),
            u_plane=_mirror_plane(
                frame.u_plane, clip.chroma_width, clip.chroma_height
            ),
            v_plane=_mirror_plane(
                frame.v_plane, clip.chroma_width, clip.chroma_height
            ),
        )
        for frame in clip.frames
    )
    return Y4mClip(
        width=clip.width,
        height=clip.height,
        rate_numerator=clip.rate_numerator,
        rate_denominator=clip.rate_denominator,
        interlace=clip.interlace,
        pixel_format=clip.pixel_format,
        frames=frames,
    )


@dataclass
class _PlaneView:
    """Affine column selection over an untouched source plane.

    ``column`` is the source column read first, ``step`` is +1 or -1, and the
    remaining fields describe the surviving rectangle within the stride.
    Mirroring and cropping compose here instead of on pixel data.
    """

    column: int
    step: int
    width: int
    height: int
    row_offset: int
    source_stride: int

    def mirrored(self) -> None:
        self.column += self.step * (self.width - 1)
        self.step = -self.step

    def cropped(self, x: int, y: int, width: int, height: int) -> None:
        # Geometry is validated on the luma plane only; a legal luma crop of
        # height two leaves a chroma plane one pixel tall.
        if x < 0 or y < 0 or width < 1 or height < 1:
            raise MediaOperationFailure(
                "VIDEO_CROP_TOO_SMALL",
                "Crop width and height must be at least two pixels.",
            )
        if x + width > self.width or y + height > self.height:
            raise MediaOperationFailure(
                "VIDEO_CROP_OUT_OF_RANGE",
                "Crop geometry exceeds the source frame.",
            )
        self.column += self.step * x
        self.row_offset += y
        self.width = width
        self.height = height


@lru_cache(maxsize=129)
def _luma_table(amount: int) -> bytes:
    return bytes(max(0, min(255, value + amount)) for value in range(256))


def _validate_crop(operation: dict[str, Any]) -> tuple[int, int, int, int]:
    x = int(operation["x"])
    y = int(operation["y"])
    width = int(operation["width"])
    height = int(operation["height"])
    if any(value % 2 for value in (x, y, width, height)):
        raise MediaOperationFailure(
            "VIDEO_CROP_ALIGNMENT",
            "Video crop geometry must align to even pixels.",
        )
    if width < 2 or height < 2:
        raise MediaOperationFailure(
            "VIDEO_CROP_TOO_SMALL",
            "Crop width and height must be at least two pixels.",
        )
    return x, y, width, height


def _next_trim_span(
    start: int,
    end: int,
    operation: dict[str, Any],
) -> tuple[int, int]:
    current_length = end - start
    relative_start = int(operation["start_frame"])
    count = int(operation["frame_count"])
    if relative_start < 0 or relative_start >= current_length:
        raise MediaOperationFailure(
            "VIDEO_TRIM_OUT_OF_RANGE",
            "The requested video trim starts outside the source.",
        )
    relative_end = min(current_length, relative_start + max(1, count))
    if relative_end <= relative_start:
        raise MediaOperationFailure(
            "VIDEO_TRIM_EMPTY",
            "The requested video trim produces no frames.",
        )
    return start + relative_start, start + relative_end


def _extract_plane(
    plane: bytes,
    view: _PlaneView,
    luma_amounts: tuple[int, ...],
) -> bytes:
    rows = []
    if view.step == 1:
        for row_index in range(view.row_offset, view.row_offset + view.height):
            base = row_index * view.source_stride + view.column
            rows.append(plane[base : base + view.width])
    else:
        for row_index in range(view.row_offset, view.row_offset + view.height):
            base = row_index * view.source_stride + view.column
            rows.append(plane[base - view.width + 1 : base + 1][::-1])
    result = b"".join(rows)
    # Luma offsets clamp at every step, so they are applied in plan order.
    for amount in luma_amounts:
        result = result.translate(_luma_table(amount))
    return result


def _source_for_streaming(
    source: bytes | bytearray | memoryview | BinaryIO,
) -> tuple[Y4mStreamInfo, Any]:
    if isinstance(source, (bytes, bytearray, memoryview)):
        payload = bytes(source)
        return scan_y4m(payload), payload
    stream = source
    info = scan_y4m(stream)
    seek = getattr(stream, "seek", None)
    if seek is None:
        raise MediaOperationFailure(
            "VIDEO_SOURCE_UNSEEKABLE",
            "Streaming Y4M rendering requires a seekable source.",
        )
    seek(0)
    return info, stream


def render_video_to_stream(
    source: bytes | bytearray | memoryview | BinaryIO,
    operations: list[dict[str, Any]],
    target: BinaryIO,
    secondary_bytes: bytes | None = None,
) -> tuple[MediaDescriptor, int]:
    """Render *source* into *target* one frame at a time.

    The full source is validated before the first output byte is written, so
    corrupted or truncated media never yields a published result.  Frames past
    a trim boundary are still read and validated, matching the eager decoder.
    """

    if secondary_bytes is not None:
        raise MediaOperationFailure(
            "VIDEO_SECONDARY_UNSUPPORTED",
            "The current video operation set accepts one source.",
        )

    info, frame_source = _source_for_streaming(source)

    luma_view = _PlaneView(
        column=0,
        step=1,
        width=info.width,
        height=info.height,
        row_offset=0,
        source_stride=info.width,
    )
    chroma_view = _PlaneView(
        column=0,
        step=1,
        width=info.chroma_width,
        height=info.chroma_height,
        row_offset=0,
        source_stride=info.chroma_width,
    )
    trim_start = 0
    trim_end = info.frame_count
    luma_amounts: list[int] = []

    try:
        for operation in operations:
            operation_type = operation["type"]
            if operation_type == "media_trim":
                trim_start, trim_end = _next_trim_span(
                    trim_start, trim_end, operation
                )
            elif operation_type == "video_crop":
                x, y, width, height = _validate_crop(operation)
                luma_view.cropped(x, y, width, height)
                chroma_view.cropped(
                    x // 2,
                    y // 2,
                    width // 2,
                    height // 2,
                )
            elif operation_type == "video_luma_offset":
                luma_amounts.append(int(operation["amount"]))
            elif operation_type == "video_mirror":
                luma_view.mirrored()
                chroma_view.mirrored()
            else:
                raise MediaOperationFailure(
                    "UNSUPPORTED_VIDEO_OPERATION",
                    f"Unsupported video operation: {operation_type}",
                )
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaOperationFailure(
            "VIDEO_OPERATION_INVALID",
            "The video operation payload is malformed.",
        ) from exc

    target.write(
        encode_y4m_header(
            width=luma_view.width,
            height=luma_view.height,
            rate_numerator=info.rate_numerator,
            rate_denominator=info.rate_denominator,
            interlace=info.interlace,
            pixel_format=info.pixel_format,
        )
    )

    emitted = 0
    for index, frame in enumerate(iter_y4m_frames(frame_source, info)):
        if index < trim_start:
            continue
        if index >= trim_end:
            # Keep reading: trailing corruption must still fail the render.
            continue
        target.write(b"FRAME\n")
        target.write(_extract_plane(frame.y_plane, luma_view, tuple(luma_amounts)))
        target.write(_extract_plane(frame.u_plane, chroma_view, ()))
        target.write(_extract_plane(frame.v_plane, chroma_view, ()))
        emitted += 1

    if emitted == 0:
        raise MediaOperationFailure(
            "VIDEO_TRIM_EMPTY",
            "The requested video trim produces no frames.",
        )
    if emitted != trim_end - trim_start:
        raise MediaOperationFailure(
            "VIDEO_TRIM_EMPTY",
            "The source ended before the requested frames were read.",
        )

    output_info = Y4mStreamInfo(
        width=luma_view.width,
        height=luma_view.height,
        rate_numerator=info.rate_numerator,
        rate_denominator=info.rate_denominator,
        interlace=info.interlace,
        pixel_format=info.pixel_format,
        frame_count=emitted,
    )
    return descriptor_for_y4m(output_info), emitted


def render_video(
    primary_bytes: bytes,
    operations: list[dict[str, Any]],
    secondary_bytes: bytes | None,
) -> bytes:
    """Render a small in-memory source to bytes using the streaming engine."""

    payload = bytearray()
    render_video_to_stream(primary_bytes, operations, _MemorySink(payload), secondary_bytes)
    return bytes(payload)


class _MemorySink:
    """Minimal binary sink that appends to a bytearray."""

    def __init__(self, payload: bytearray) -> None:
        self._payload = payload

    def write(self, data: bytes) -> int:
        self._payload.extend(data)
        return len(data)


def render_video_to_bytes(
    source: bytes | bytearray | memoryview | BinaryIO,
    operations: list[dict[str, Any]],
) -> bytes:
    payload = bytearray()
    render_video_to_stream(source, operations, _MemorySink(payload))
    return bytes(payload)


__all__ = [
    "crop_video",
    "mirror_video",
    "offset_luma",
    "render_video",
    "render_video_to_bytes",
    "render_video_to_stream",
    "trim_video",
]
