"""Planar YUV4MPEG2 parsing and serialization for bounded C420 media.

The streaming helpers in this module never retain decoded frames.  A caller
can scan a source to learn its descriptor with a constant amount of working
memory, then re-read it one frame at a time while performing byte operations.
The clip based helpers remain available for small in-memory callers and are
used by the compatibility render path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Iterator, Union

from ..errors import UnsupportedMedia
from ..models import MediaDescriptor


SUPPORTED_PIXEL_FORMATS = {"C420", "C420jpeg", "C420mpeg2", "C420p"}
MAX_FRAMES = 240
_SKIP_CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True)
class Y4mFrame:
    y_plane: bytes
    u_plane: bytes
    v_plane: bytes


@dataclass(frozen=True)
class Y4mClip:
    width: int
    height: int
    rate_numerator: int
    rate_denominator: int
    interlace: str
    pixel_format: str
    frames: tuple[Y4mFrame, ...]

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def chroma_width(self) -> int:
        return self.width // 2

    @property
    def chroma_height(self) -> int:
        return self.height // 2


@dataclass(frozen=True)
class Y4mStreamInfo:
    """Header metadata plus the validated frame count of a Y4M source."""

    width: int
    height: int
    rate_numerator: int
    rate_denominator: int
    interlace: str
    pixel_format: str
    frame_count: int

    @property
    def chroma_width(self) -> int:
        return self.width // 2

    @property
    def chroma_height(self) -> int:
        return self.height // 2

    @property
    def luma_plane_size(self) -> int:
        return self.width * self.height

    @property
    def chroma_plane_size(self) -> int:
        return self.chroma_width * self.chroma_height

    @property
    def frame_size(self) -> int:
        return self.luma_plane_size + self.chroma_plane_size * 2


def _read_line(data: bytes, position: int) -> tuple[str, int]:
    newline = data.find(b"\n", position)
    if newline < 0:
        raise UnsupportedMedia("Y4M stream ended inside a header line.")
    raw = data[position:newline]
    if raw.endswith(b"\r"):
        raw = raw[:-1]
    try:
        line = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise UnsupportedMedia("Y4M header lines must be ASCII.") from exc
    return line, newline + 1


def _parse_header(line: str) -> dict[str, str | int]:
    if not line.startswith("YUV4MPEG2"):
        raise UnsupportedMedia("The payload does not begin with a Y4M header.")
    fields = line.split()
    if not fields or fields[0] != "YUV4MPEG2":
        raise UnsupportedMedia("The Y4M signature is malformed.")

    values: dict[str, str | int] = {
        "interlace": "?",
        "pixel_format": "C420jpeg",
    }
    for field in fields[1:]:
        if len(field) < 2:
            raise UnsupportedMedia("A Y4M header field is too short.")
        key, raw_value = field[0], field[1:]
        if key == "W":
            values["width"] = int(raw_value)
        elif key == "H":
            values["height"] = int(raw_value)
        elif key == "F":
            parts = raw_value.split(":")
            if len(parts) != 2:
                raise UnsupportedMedia("The Y4M frame-rate field is malformed.")
            values["rate_numerator"] = int(parts[0])
            values["rate_denominator"] = int(parts[1])
        elif key == "I":
            values["interlace"] = raw_value
        elif key == "C":
            values["pixel_format"] = f"C{raw_value}"

    required = ("width", "height", "rate_numerator", "rate_denominator")
    if any(key not in values for key in required):
        raise UnsupportedMedia("Y4M header is missing geometry or frame rate.")
    return values


def _validate_header(values: dict[str, str | int]) -> None:
    width = int(values["width"])
    height = int(values["height"])
    numerator = int(values["rate_numerator"])
    denominator = int(values["rate_denominator"])
    pixel_format = str(values["pixel_format"])
    if width < 2 or height < 2 or width > 1920 or height > 1080:
        raise UnsupportedMedia(
            "Y4M geometry is outside the supported bounds.",
            details={"width": width, "height": height},
        )
    if width % 2 or height % 2:
        raise UnsupportedMedia("Y4M width and height must both be even.")
    if numerator <= 0 or denominator <= 0 or numerator > 240 or denominator > 1001:
        raise UnsupportedMedia("Y4M frame rate is outside the supported bounds.")
    if pixel_format not in SUPPORTED_PIXEL_FORMATS:
        raise UnsupportedMedia(
            "Only planar 4:2:0 Y4M payloads are supported.",
            details={"pixel_format": pixel_format},
        )


def _validated_header(line: str) -> dict[str, str | int]:
    try:
        header = _parse_header(line)
        _validate_header(header)
    except ValueError as exc:
        raise UnsupportedMedia("A Y4M numeric header field is malformed.") from exc
    return header


class _ByteCursor:
    """Cursor over an immutable bytes payload that never copies frame data."""

    __slots__ = ("_data", "_position")

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._position = 0

    def readline(self) -> bytes | None:
        if self._position >= len(self._data):
            return None
        newline = self._data.find(b"\n", self._position)
        if newline < 0:
            raise UnsupportedMedia("Y4M stream ended inside a header line.")
        raw = self._data[self._position : newline + 1]
        self._position = newline + 1
        return raw

    def take(self, size: int) -> bytes:
        end = self._position + size
        if end > len(self._data):
            raise UnsupportedMedia("A Y4M frame payload is truncated.")
        chunk = self._data[self._position : end]
        self._position = end
        return chunk

    def skip(self, size: int) -> None:
        end = self._position + size
        if end > len(self._data):
            raise UnsupportedMedia("A Y4M frame payload is truncated.")
        self._position = end


class _StreamCursor:
    """Cursor over a binary file object using a fixed-size read buffer."""

    __slots__ = ("_stream",)

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream

    def readline(self) -> bytes | None:
        raw = self._stream.readline()
        if not raw:
            return None
        if not raw.endswith(b"\n"):
            raise UnsupportedMedia("Y4M stream ended inside a header line.")
        return raw

    def take(self, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = self._stream.read(min(remaining, _SKIP_CHUNK_BYTES))
            if not chunk:
                raise UnsupportedMedia("A Y4M frame payload is truncated.")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def skip(self, size: int) -> None:
        remaining = size
        while remaining:
            chunk = self._stream.read(min(remaining, _SKIP_CHUNK_BYTES))
            if not chunk:
                raise UnsupportedMedia("A Y4M frame payload is truncated.")
            remaining -= len(chunk)


Cursor = Union[_ByteCursor, _StreamCursor]
Source = Union[bytes, bytearray, memoryview, BinaryIO]


def _make_cursor(source: Source) -> Cursor:
    if isinstance(source, bytes):
        return _ByteCursor(source)
    if isinstance(source, (bytearray, memoryview)):
        return _ByteCursor(bytes(source))
    readline = getattr(source, "readline", None)
    if readline is None:
        raise TypeError("A Y4M source must be bytes or a binary stream.")
    return _StreamCursor(source)


def _read_header_line(cursor: Cursor) -> str | None:
    raw = cursor.readline()
    if raw is None:
        return None
    line = raw[:-1]
    if line.endswith(b"\r"):
        line = line[:-1]
    try:
        return line.decode("ascii")
    except UnicodeDecodeError as exc:
        raise UnsupportedMedia("Y4M header lines must be ASCII.") from exc


def _info_from_header(
    header: dict[str, str | int],
    frame_count: int,
) -> Y4mStreamInfo:
    return Y4mStreamInfo(
        width=int(header["width"]),
        height=int(header["height"]),
        rate_numerator=int(header["rate_numerator"]),
        rate_denominator=int(header["rate_denominator"]),
        interlace=str(header["interlace"]),
        pixel_format=str(header["pixel_format"]),
        frame_count=frame_count,
    )


def scan_y4m(source: Source) -> Y4mStreamInfo:
    """Validate a Y4M source and return its geometry without retaining frames."""

    cursor = _make_cursor(source)
    first = _read_header_line(cursor)
    if first is None or not first.startswith("YUV4MPEG2"):
        raise UnsupportedMedia("The payload is not a Y4M stream.")
    header = _validated_header(first)
    info = _info_from_header(header, 0)
    frame_size = info.frame_size

    frame_count = 0
    while True:
        line = _read_header_line(cursor)
        if line is None:
            break
        if not line.startswith("FRAME"):
            raise UnsupportedMedia("A Y4M frame header is missing.")
        cursor.skip(frame_size)
        frame_count += 1
        if frame_count > MAX_FRAMES:
            raise UnsupportedMedia("Y4M frame count exceeds the supported bound.")

    if not frame_count:
        raise UnsupportedMedia("Y4M stream contains no frames.")
    return _info_from_header(header, frame_count)


def iter_y4m_frames(
    source: Source,
    info: Y4mStreamInfo | None = None,
) -> Iterator[Y4mFrame]:
    """Yield one decoded frame at a time, retaining only the current frame."""

    cursor = _make_cursor(source)
    first = _read_header_line(cursor)
    if first is None or not first.startswith("YUV4MPEG2"):
        raise UnsupportedMedia("The payload is not a Y4M stream.")
    header = _validated_header(first)
    current = _info_from_header(header, 0)
    if info is not None and (
        info.width != current.width
        or info.height != current.height
    ):
        raise UnsupportedMedia("The Y4M geometry changed during processing.")

    frame_size = current.frame_size
    luma_size = current.luma_plane_size
    chroma_size = current.chroma_plane_size
    frame_count = 0
    while True:
        line = _read_header_line(cursor)
        if line is None:
            break
        if not line.startswith("FRAME"):
            raise UnsupportedMedia("A Y4M frame header is missing.")
        payload = cursor.take(frame_size)
        frame_count += 1
        if frame_count > MAX_FRAMES:
            raise UnsupportedMedia("Y4M frame count exceeds the supported bound.")
        if info is not None and frame_count > info.frame_count:
            raise UnsupportedMedia("The Y4M frame count changed during processing.")
        yield Y4mFrame(
            y_plane=payload[:luma_size],
            u_plane=payload[luma_size : luma_size + chroma_size],
            v_plane=payload[luma_size + chroma_size :],
        )

    if not frame_count:
        raise UnsupportedMedia("Y4M stream contains no frames.")
    if info is not None and frame_count != info.frame_count:
        raise UnsupportedMedia("The Y4M frame count changed during processing.")


def decode_y4m(data: bytes) -> Y4mClip:
    if not data.startswith(b"YUV4MPEG2"):
        raise UnsupportedMedia("The payload is not a Y4M stream.")
    header_line, position = _read_line(data, 0)
    header = _validated_header(header_line)

    width = int(header["width"])
    height = int(header["height"])
    chroma_width = width // 2
    chroma_height = height // 2
    y_size = width * height
    chroma_size = chroma_width * chroma_height
    frames: list[Y4mFrame] = []

    while position < len(data):
        line, position = _read_line(data, position)
        if not line.startswith("FRAME"):
            raise UnsupportedMedia("A Y4M frame header is missing.")
        frame_size = y_size + chroma_size * 2
        end = position + frame_size
        if end > len(data):
            raise UnsupportedMedia("A Y4M frame payload is truncated.")
        y_end = position + y_size
        u_end = y_end + chroma_size
        frames.append(
            Y4mFrame(
                y_plane=data[position:y_end],
                u_plane=data[y_end:u_end],
                v_plane=data[u_end:end],
            )
        )
        position = end
        if len(frames) > MAX_FRAMES:
            raise UnsupportedMedia("Y4M frame count exceeds the supported bound.")

    if not frames:
        raise UnsupportedMedia("Y4M stream contains no frames.")
    return Y4mClip(
        width=width,
        height=height,
        rate_numerator=int(header["rate_numerator"]),
        rate_denominator=int(header["rate_denominator"]),
        interlace=str(header["interlace"]),
        pixel_format=str(header["pixel_format"]),
        frames=tuple(frames),
    )


def descriptor_for_y4m(info: Y4mStreamInfo) -> MediaDescriptor:
    return MediaDescriptor(
        kind="video",
        mime_type="video/x-yuv4mpeg2",
        container="y4m",
        properties={
            "width": info.width,
            "height": info.height,
            "frame_count": info.frame_count,
            "rate_numerator": info.rate_numerator,
            "rate_denominator": info.rate_denominator,
            "interlace": info.interlace,
            "pixel_format": info.pixel_format,
            "duration_seconds": round(
                info.frame_count
                * info.rate_denominator
                / info.rate_numerator,
                9,
            ),
        },
    )


def inspect_y4m(data: bytes) -> MediaDescriptor:
    return descriptor_for_y4m(scan_y4m(data))


def encode_y4m(clip: Y4mClip) -> bytes:
    if not clip.frames:
        raise UnsupportedMedia("Cannot encode a Y4M stream without frames.")
    header = (
        "YUV4MPEG2 "
        f"W{clip.width} H{clip.height} "
        f"F{clip.rate_numerator}:{clip.rate_denominator} "
        f"I{clip.interlace} {clip.pixel_format}\n"
    ).encode("ascii")
    parts = [header]
    for frame in clip.frames:
        parts.append(b"FRAME\n")
        parts.append(frame.y_plane)
        parts.append(frame.u_plane)
        parts.append(frame.v_plane)
    return b"".join(parts)


def encode_y4m_header(
    *,
    width: int,
    height: int,
    rate_numerator: int,
    rate_denominator: int,
    interlace: str,
    pixel_format: str,
) -> bytes:
    """Return the canonical header used by :func:`encode_y4m`."""

    return (
        "YUV4MPEG2 "
        f"W{width} H{height} "
        f"F{rate_numerator}:{rate_denominator} "
        f"I{interlace} {pixel_format}\n"
    ).encode("ascii")
