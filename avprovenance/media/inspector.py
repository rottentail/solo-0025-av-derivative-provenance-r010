"""Resolve media bytes to a supported inspector and descriptor."""

from __future__ import annotations

from pathlib import Path

from ..errors import UnsupportedMedia
from ..models import MediaDescriptor
from .wav import inspect_wav, inspect_wav_file
from .y4m import descriptor_for_y4m, inspect_y4m, scan_y4m


def inspect_media(data: bytes) -> MediaDescriptor:
    if not data:
        raise UnsupportedMedia("The media payload is empty.")
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return inspect_wav(data)
    if data.startswith(b"YUV4MPEG2"):
        return inspect_y4m(data)
    raise UnsupportedMedia(
        "The payload is not a supported PCM WAV or planar Y4M stream.",
        details={"prefix_hex": data[:12].hex()},
    )


def inspect_media_file(path: Path) -> MediaDescriptor:
    """Probe a persisted media file with a bounded read.

    WAV geometry comes from the RIFF header only.  Y4M is validated frame by
    frame from a stream, so neither kind is read into memory whole.
    """

    prefix = b""
    try:
        with open(path, "rb") as handle:
            prefix = handle.read(12)
    except OSError as exc:
        raise UnsupportedMedia("The media payload could not be read.") from exc
    if not prefix:
        raise UnsupportedMedia("The media payload is empty.")
    if prefix.startswith(b"RIFF") and prefix[8:12] == b"WAVE":
        return inspect_wav_file(path)
    if prefix.startswith(b"YUV4MPEG2"):
        try:
            with open(path, "rb") as handle:
                return descriptor_for_y4m(scan_y4m(handle))
        except OSError as exc:
            raise UnsupportedMedia("The media payload could not be read.") from exc
    raise UnsupportedMedia(
        "The payload is not a supported PCM WAV or planar Y4M stream.",
        details={"prefix_hex": prefix.hex()},
    )
