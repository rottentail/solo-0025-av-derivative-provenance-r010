"""Incremental JSON reader for base64 media uploads.

The upload body is parsed as it arrives.  The encoded payload is decoded in
chunks straight into a spooling file, so peak memory tracks the read buffer
and the JSON scanner, never the encoded or decoded media length.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import BinaryIO

from ..errors import ValidationFailure
from ..validation import MAX_HTTP_BODY_BYTES, MAX_MEDIA_BYTES

READ_CHUNK_BYTES = 64 * 1024
MAX_FILENAME_RAW_BYTES = 4096
MAX_JSON_STRING_BYTES = 16 * 1024
MAX_JSON_DEPTH = 200
_DECODE_GROUP_CHARACTERS = 4 * (READ_CHUNK_BYTES // 4)
_BASE64_ALPHABET_PATTERN = re.compile(rb"^[A-Za-z0-9+/=]*$")
_JSON_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}
_HEX_DIGITS = re.compile(r"^[0-9a-fA-F]{4}$")


class _BodyReader:
    """Chunked reader with one-slot lookahead over a known body length."""

    def __init__(
        self,
        stream: BinaryIO,
        length: int,
        chunk_bytes: int = READ_CHUNK_BYTES,
    ) -> None:
        self._stream = stream
        self._remaining = length
        self._chunk_bytes = chunk_bytes
        self._buffer = b""
        self._position = 0

    def _refill(self) -> bool:
        if self._position < len(self._buffer):
            return True
        if self._remaining <= 0:
            return False
        data = self._stream.read(min(self._chunk_bytes, self._remaining))
        if not data:
            raise ValidationFailure("Request body ended before Content-Length.")
        self._remaining -= len(data)
        self._buffer = data
        self._position = 0
        return True

    def peek(self) -> int | None:
        if not self._refill():
            return None
        return self._buffer[self._position]

    def take(self) -> int | None:
        value = self.peek()
        if value is not None:
            self._position += 1
        return value

    def stream_until(self, stop_bytes: bytes, consumer) -> None:
        """Consume bytes up to one of *stop_bytes*, handing each chunk over.

        Each chunk is passed to *consumer* and released immediately, so a
        multi-megabyte string body never lands in a list.
        """

        while True:
            if not self._refill():
                raise ValidationFailure(
                    "Request body must be valid UTF-8 JSON."
                )
            buffer = self._buffer
            position = self._position
            nearest = len(buffer)
            for stop in stop_bytes:
                index = buffer.find(bytes((stop,)), position)
                if index >= 0 and index < nearest:
                    nearest = index
            if nearest > position:
                consumer(buffer[position:nearest])
                self._position = nearest
            if nearest < len(buffer):
                return
            self._position = len(buffer)

    def skip_whitespace(self) -> None:
        while True:
            value = self._buffer[self._position] if self._position < len(self._buffer) else None
            if value is None:
                if not self._refill():
                    return
                continue
            if not chr(value).isspace():
                return
            self._position += 1


class _Base64Spool:
    """Streams decoded base64 groups into a file with fixed memory use."""

    def __init__(self, handle: BinaryIO, path: Path) -> None:
        self._handle = handle
        self.path = path
        self._pending = b""
        self._padded = False
        self._closed = False
        self.size = 0

    def reset(self) -> None:
        self._handle.seek(0)
        self._handle.truncate(0)
        self._pending = b""
        self._padded = False
        self._closed = False
        self.size = 0

    def _emit(self, group: bytes) -> None:
        try:
            decoded = base64.b64decode(group, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValidationFailure(
                "data_base64 is not strict base64."
            ) from exc
        self.size += len(decoded)
        if self.size > MAX_MEDIA_BYTES:
            raise ValidationFailure(
                "decoded media exceeds the payload budget.",
                details={
                    "size_bytes": self.size,
                    "max_bytes": MAX_MEDIA_BYTES,
                },
            )
        self._handle.write(decoded)

    def feed(self, span: bytes) -> None:
        if not span:
            return
        if self._closed:
            raise ValidationFailure("data_base64 is not strict base64.")
        if not _BASE64_ALPHABET_PATTERN.fullmatch(span):
            raise ValidationFailure("data_base64 is not strict base64.")
        if self._padded:
            # Only padding may follow padding, and a final group is 4 chars.
            if span.strip(b"="):
                raise ValidationFailure("data_base64 is not strict base64.")
            self._pending += span
            if len(self._pending) > 4:
                raise ValidationFailure("data_base64 is not strict base64.")
            return
        equals = span.find(b"=")
        if equals >= 0:
            self._feed_alphabet(span[:equals])
            self._pending += span[equals:]
            self._padded = True
            if len(self._pending) > 4:
                raise ValidationFailure("data_base64 is not strict base64.")
            return
        self._feed_alphabet(span)

    def _feed_alphabet(self, span: bytes) -> None:
        combined = self._pending + span
        complete = len(combined) - (len(combined) % 4)
        while complete:
            take = min(complete, _DECODE_GROUP_CHARACTERS)
            self._emit(combined[:take])
            combined = combined[take:]
            complete -= take
        self._pending = combined

    def feed_single(self, character: str) -> None:
        if ord(character) > 0x7F:
            raise ValidationFailure("data_base64 is not strict base64.")
        self.feed(character.encode("ascii"))

    def finish(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pending:
            if self._padded and len(self._pending) == 4:
                self._emit(self._pending)
            else:
                raise ValidationFailure("data_base64 is not strict base64.")
        self._handle.flush()
        if self.size == 0:
            raise ValidationFailure("data_base64 decoded to an empty payload.")


@dataclass(frozen=True)
class AssetUpload:
    filename: object
    payload_path: Path
    payload_size: int


_MISSING = object()


def _strict_base64_span(span: bytes) -> None:
    if not _BASE64_ALPHABET_PATTERN.fullmatch(span):
        raise ValidationFailure("data_base64 is not strict base64.")


def _filename_raw_escape(escape: bytes) -> bytes:
    """Convert a decoded escape back into JSON source bytes.

    ``raw.extend`` copies bytes verbatim, so ``\\u003c`` would otherwise turn
    into ``<`` and could inject a quote or backslash into the raw buffer.
    """

    if escape == b"\\u":
        raise ValidationFailure("Request body must be valid UTF-8 JSON.")
    marker = escape[1:2]
    if marker == b'"':
        return b"\\" + b'"'
    if marker == b"\\":
        return b"\\\\"
    if marker == b"/":
        return b"\\/"
    return b"\\u" + escape[2:6] if len(escape) == 6 else b"\\" + marker


class _JsonAssetScanner:
    def __init__(self, reader: _BodyReader, spool: _Base64Spool) -> None:
        self._reader = reader
        self._spool = spool
        self.filename: object = _MISSING
        self.payload: object = _MISSING

    def _invalid(self) -> ValidationFailure:
        return ValidationFailure("Request body must be valid UTF-8 JSON.")

    def _expect(self, character: str) -> None:
        value = self._reader.take()
        if value is None or chr(value) != character:
            raise self._invalid()

    def scan(self) -> None:
        self._reader.skip_whitespace()
        value = self._reader.peek()
        if value is None:
            raise ValidationFailure("A JSON request body is required.")
        if chr(value) != "{":
            raise self._invalid()
        self._reader.take()
        self._reader.skip_whitespace()
        value = self._reader.peek()
        if value is not None and chr(value) == "}":
            self._reader.take()
        else:
            self._members()
        self._reader.skip_whitespace()
        if self._reader.peek() is not None:
            raise self._invalid()

    def _members(self) -> None:
        while True:
            self._reader.skip_whitespace()
            self._expect('"')
            key = self._string_value(capture=False)
            if not isinstance(key, str):
                raise self._invalid()
            self._reader.skip_whitespace()
            self._expect(":")
            self._reader.skip_whitespace()
            if key == "filename":
                self.filename = self._value(capture="filename")
            elif key == "data_base64":
                self.payload = self._value(capture="payload")
            else:
                self._value(capture=None)
            self._reader.skip_whitespace()
            value = self._reader.peek()
            if value is None:
                raise self._invalid()
            character = chr(value)
            self._reader.take()
            if character == ",":
                continue
            if character == "}":
                return
            raise self._invalid()

    def _value(self, capture: str | None) -> object:
        value = self._reader.peek()
        if value is None:
            raise self._invalid()
        character = chr(value)
        if character == '"':
            self._reader.take()
            if capture == "filename":
                return self._filename_value()
            if capture == "payload":
                return self._payload_value()
            self._string_value(capture=False)
            return _MISSING
        if capture == "payload":
            # A non-string data_base64 invalidates any earlier payload value.
            self._spool.reset()
        self._skip_complex_value()
        return _MISSING

    def _filename_value(self) -> object:
        raw = bytearray()
        overflow = False

        def take_span(span: bytes) -> None:
            nonlocal overflow
            if overflow:
                return
            if len(raw) + len(span) > MAX_FILENAME_RAW_BYTES:
                overflow = True
                raw.clear()
                return
            raw.extend(span)

        while True:
            self._reader.stream_until(b'"\\', take_span)
            stop = self._reader.peek()
            if stop is None:
                raise self._invalid()
            if stop == ord('"'):
                self._reader.take()
                break
            self._reader.take()
            take_span(_filename_raw_escape(self._consume_escape()))
        if overflow:
            # Force the downstream length check to reject it.
            return "x" * (MAX_FILENAME_RAW_BYTES + 1)
        try:
            return json.loads(b'"' + bytes(raw) + b'"')
        except (ValueError, UnicodeDecodeError) as exc:
            raise self._invalid() from exc

    def _payload_value(self) -> object:
        self._spool.reset()

        def take_span(span: bytes) -> None:
            _strict_base64_span(span)
            self._spool.feed(span)

        while True:
            self._reader.stream_until(b'"\\', take_span)
            stop = self._reader.peek()
            if stop is None:
                raise self._invalid()
            if stop == ord('"'):
                self._reader.take()
                break
            self._reader.take()
            # Escaped characters are validated inside feed_single, so they
            # cannot bypass the alphabet check that spans receive.
            escape = self._consume_escape()
            self._spool.feed_single(self._escape_to_character(escape))
        self._spool.finish()
        return True

    def _string_value(self, capture: bool) -> object:
        del capture
        raw = bytearray()

        def take_span(span: bytes) -> None:
            if len(raw) + len(span) > MAX_JSON_STRING_BYTES:
                raise self._invalid()
            raw.extend(span)

        while True:
            self._reader.stream_until(b'"\\', take_span)
            stop = self._reader.peek()
            if stop is None:
                raise self._invalid()
            if stop == ord('"'):
                self._reader.take()
                break
            self._reader.take()
            escape = _filename_raw_escape(self._consume_escape())
            if len(raw) + len(escape) > MAX_JSON_STRING_BYTES:
                raise self._invalid()
            raw.extend(escape)
        try:
            return json.loads(b'"' + bytes(raw) + b'"')
        except (ValueError, UnicodeDecodeError) as exc:
            raise self._invalid() from exc

    def _consume_escape(self) -> bytes:
        """Consume an escape sequence after its backslash and return its bytes."""

        marker = self._reader.take()
        if marker is None:
            raise self._invalid()
        character = chr(marker)
        if character in _JSON_ESCAPES:
            return b"\\" + bytes((marker,))
        if character != "u":
            raise self._invalid()
        digits = bytearray()
        for _ in range(4):
            value = self._reader.take()
            if value is None:
                raise self._invalid()
            digits.append(value)
        if not _HEX_DIGITS.match(digits.decode("ascii", "replace")):
            raise self._invalid()
        return b"\\u" + bytes(digits)

    @staticmethod
    def _escape_to_character(escape: bytes) -> str:
        try:
            return json.loads(b'"' + escape + b'"')
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValidationFailure(
                "data_base64 is not strict base64."
            ) from exc

    def _skip_complex_value(self) -> None:
        """Skip one JSON value without buffering it.

        Validation follows RFC 8259 closely enough that any body this scanner
        accepts also parses with ``json.loads``.  Recursion is bounded by
        ``MAX_JSON_DEPTH``; each level costs one Python frame, not the size of
        the value.
        """

        value = self._reader.peek()
        if value is None:
            raise self._invalid()
        character = chr(value)
        if character == '"':
            self._reader.take()
            self._string_value(capture=False)
        elif character == "{":
            self._reader.take()
            self._skip_object()
        elif character == "[":
            self._reader.take()
            self._skip_array()
        else:
            self._consume_scalar()

    def _skip_object(self, depth: int = 1) -> None:
        if depth > MAX_JSON_DEPTH:
            raise self._invalid()
        self._reader.skip_whitespace()
        value = self._reader.peek()
        if value is None:
            raise self._invalid()
        if chr(value) == "}":
            self._reader.take()
            return
        while True:
            self._reader.skip_whitespace()
            value = self._reader.peek()
            if value is None or chr(value) != '"':
                raise self._invalid()
            self._reader.take()
            self._string_value(capture=False)
            self._reader.skip_whitespace()
            value = self._reader.peek()
            if value is None or chr(value) != ":":
                raise self._invalid()
            self._reader.take()
            self._reader.skip_whitespace()
            self._skip_nested(depth)
            self._reader.skip_whitespace()
            value = self._reader.peek()
            if value is None:
                raise self._invalid()
            character = chr(value)
            self._reader.take()
            if character == ",":
                continue
            if character == "}":
                return
            raise self._invalid()

    def _skip_array(self, depth: int = 1) -> None:
        if depth > MAX_JSON_DEPTH:
            raise self._invalid()
        self._reader.skip_whitespace()
        value = self._reader.peek()
        if value is None:
            raise self._invalid()
        if chr(value) == "]":
            self._reader.take()
            return
        while True:
            self._reader.skip_whitespace()
            self._skip_nested(depth)
            self._reader.skip_whitespace()
            value = self._reader.peek()
            if value is None:
                raise self._invalid()
            character = chr(value)
            self._reader.take()
            if character == ",":
                continue
            if character == "]":
                return
            raise self._invalid()

    def _skip_nested(self, depth: int) -> None:
        value = self._reader.peek()
        if value is None:
            raise self._invalid()
        character = chr(value)
        if character == '"':
            self._reader.take()
            self._string_value(capture=False)
        elif character == "{":
            self._reader.take()
            self._skip_object(depth + 1)
        elif character == "[":
            self._reader.take()
            self._skip_array(depth + 1)
        else:
            self._consume_scalar()

    def _consume_scalar(self) -> None:
        value = self._reader.peek()
        if value is None:
            raise self._invalid()
        character = chr(value)
        if character in "-0123456789":
            self._consume_number()
            return
        for literal in (b"true", b"false", b"null"):
            if self._consume_literal(literal):
                return
        raise self._invalid()

    def _consume_literal(self, literal: bytes) -> bool:
        for index in range(len(literal)):
            value = self._reader.peek()
            if value is None or bytes((value,)) != literal[index : index + 1]:
                return False
            self._reader.take()
        return True

    def _consume_number(self) -> None:
        # Follows the RFC 8259 number grammar, including the rule that a
        # leading zero may not be followed by more digits.
        seen_digit = False
        value = self._reader.peek()
        if value is not None and chr(value) == "-":
            self._reader.take()
        value = self._reader.peek()
        if value is not None and chr(value) == "0":
            self._reader.take()
            seen_digit = True
            value = self._reader.peek()
            if value is not None and chr(value).isdigit():
                raise self._invalid()
        else:
            while True:
                value = self._reader.peek()
                if value is None or not chr(value).isdigit():
                    break
                self._reader.take()
                seen_digit = True
        if not seen_digit:
            raise self._invalid()
        value = self._reader.peek()
        if value is not None and chr(value) == ".":
            self._reader.take()
            if not self._digits():
                raise self._invalid()
        value = self._reader.peek()
        if value is not None and chr(value) in "eE":
            self._reader.take()
            value = self._reader.peek()
            if value is not None and chr(value) in "+-":
                self._reader.take()
            if not self._digits():
                raise self._invalid()

    def _digits(self) -> bool:
        seen = False
        while True:
            value = self._reader.peek()
            if value is None or not chr(value).isdigit():
                return seen
            self._reader.take()
            seen = True


def read_asset_upload(
    handler: BaseHTTPRequestHandler,
    spool_handle: BinaryIO,
    spool_path: Path,
) -> AssetUpload:
    """Parse a POST /v1/assets body into a filename and spooled payload file."""

    raw_length = handler.headers.get("Content-Length")
    if raw_length is None:
        raise ValidationFailure("Content-Length is required for this request.")
    try:
        length = int(raw_length)
    except ValueError as exc:
        raise ValidationFailure("Content-Length must be an integer.") from exc
    if length < 0 or length > MAX_HTTP_BODY_BYTES:
        raise ValidationFailure(
            "Request body exceeds the configured limit.",
            details={"max_bytes": MAX_HTTP_BODY_BYTES},
        )
    if length == 0:
        raise ValidationFailure("A JSON request body is required.")

    spool = _Base64Spool(spool_handle, spool_path)
    scanner = _JsonAssetScanner(_BodyReader(handler.rfile, length), spool)
    scanner.scan()

    if scanner.filename is _MISSING:
        filename: object = None
    else:
        filename = scanner.filename
    payload_present = scanner.payload is not _MISSING
    if not payload_present:
        raise ValidationFailure("data_base64 must be a string.")
    spool_handle.flush()
    return AssetUpload(
        filename=filename,
        payload_path=spool_path,
        payload_size=spool.size,
    )
