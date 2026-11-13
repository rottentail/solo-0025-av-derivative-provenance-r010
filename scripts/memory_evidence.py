#!/usr/bin/env python3
"""Measure working memory for legal near-limit Y4M derivations.

Each case uploads, renders, and verifies one Y4M through the public HTTP
surface while a subprocess records its own peak resident set size.  The
upload body is pre-encoded to a file and streamed from disk, so the client
cannot mask or inflate the server's media-pipeline footprint.

Assertions:

* every case completes and its integrity check reports ``verified``;
* the rendered bytes equal the legacy clip-at-a-time algorithm exactly;
* peak RSS stays flat as the frame count grows within one geometry, proving
  that memory does not scale with clip length;
* the largest case is a legal payload within 10 percent of the configured
  media budget.

Usage::

    python3 scripts/memory_evidence.py
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import resource
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from avprovenance.http_api import build_http_server
from avprovenance.media.video_ops import (
    crop_video,
    mirror_video,
    offset_luma,
    render_video_to_bytes,
    trim_video,
)
from avprovenance.media.y4m import decode_y4m, encode_y4m
from avprovenance.validation import MAX_MEDIA_BYTES

WIDTH = 1920
HEIGHT = 1080
FRAME_COUNTS = (2, 5, 8)
OPERATIONS = [
    {"type": "media_trim", "start_frame": 0, "frame_count": 6},
    {"type": "video_crop", "x": 200, "y": 100, "width": 1280, "height": 720},
    {"type": "video_luma_offset", "amount": 9},
    {"type": "video_mirror"},
]
# Trim reaches past the smaller clips, so frame_count clamps per case.
CLAMPED_OPERATIONS = [dict(OPERATIONS[0], frame_count=240)] + OPERATIONS[1:]
RSS_TOLERANCE_MB = 8.0


def _source_header() -> bytes:
    return (
        f"YUV4MPEG2 W{WIDTH} H{HEIGHT} F24:1 Ip C420jpeg\n"
    ).encode("ascii")


def write_source(path: Path, frames: int) -> int:
    chroma_width = WIDTH // 2
    chroma_height = HEIGHT // 2
    with open(path, "wb") as handle:
        handle.write(_source_header())
        for frame_index in range(frames):
            handle.write(b"FRAME\n")
            handle.write(
                bytes(
                    (x * 3 + y * 5 + frame_index * 11) % 256
                    for y in range(HEIGHT)
                    for x in range(WIDTH)
                )
            )
            handle.write(
                bytes(
                    (96 + x * 2 + frame_index * 3) % 256
                    for _y in range(chroma_height)
                    for x in range(chroma_width)
                )
            )
            handle.write(
                bytes(
                    (144 + y * 2 + frame_index * 5) % 256
                    for y in range(chroma_height)
                    for _x in range(chroma_width)
                )
            )
    return path.stat().st_size


def write_upload_body(body_path: Path, source_path: Path, filename: str) -> int:
    """Base64-encode in 3-byte groups so no group straddles a chunk."""

    with open(body_path, "wb") as out:
        out.write(b'{"filename":')
        out.write(json.dumps(filename).encode("utf-8"))
        out.write(b',"data_base64":"')
        with open(source_path, "rb") as source:
            while True:
                chunk = source.read(255 * 1024)
                if not chunk:
                    break
                out.write(base64.b64encode(chunk))
        out.write(b'"}')
    return body_path.stat().st_size


class _StreamBody:
    def __init__(self, path: Path) -> None:
        self._handle = open(path, "rb")

    def read(self, size: int = -1) -> bytes:
        return self._handle.read(256 * 1024 if size < 0 else size)

    def close(self) -> None:
        self._handle.close()


def _request(host: str, port: int, method: str, path: str, body=None, headers=None):
    connection = http.client.HTTPConnection(host, port, timeout=600)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        raw = response.read()
        return response.status, json.loads(raw.decode("utf-8"))
    finally:
        connection.close()


def _legacy_digest(source_bytes: bytes) -> str:
    """Reproduce the released clip-at-a-time algorithm for the same plan."""

    import hashlib

    clip = decode_y4m(source_bytes)
    for operation in CLAMPED_OPERATIONS:
        kind = operation["type"]
        if kind == "media_trim":
            clip = trim_video(clip, dict(operation))
        elif kind == "video_crop":
            clip = crop_video(clip, operation)
        elif kind == "video_luma_offset":
            clip = offset_luma(clip, operation)
        elif kind == "video_mirror":
            clip = mirror_video(clip, operation)
        else:  # pragma: no cover - fixed plan
            raise AssertionError(kind)
    return hashlib.sha256(encode_y4m(clip)).hexdigest()


def run_legacy_check(frames: int) -> dict[str, Any]:
    """Compare the streaming renderer with the eager algorithm, in isolation.

    This runs in its own process because the eager algorithm necessarily
    holds every decoded frame, which would otherwise dominate the pipeline
    measurement.
    """

    import hashlib

    work = Path(tempfile.mkdtemp(prefix="av-legacy-"))
    source_path = work / "source.y4m"
    write_source(source_path, frames)
    source_bytes = source_path.read_bytes()
    streamed = render_video_to_bytes(source_bytes, CLAMPED_OPERATIONS)
    return {
        "frames": frames,
        "streamed_sha256": hashlib.sha256(streamed).hexdigest(),
        "legacy_sha256": _legacy_digest(source_bytes),
        "streamed_bytes": len(streamed),
    }


def run_worker(frames: int) -> dict[str, Any]:
    work = Path(tempfile.mkdtemp(prefix="av-memory-"))
    source_path = work / "source.y4m"
    source_size = write_source(source_path, frames)
    body_path = work / "body.json"
    body_size = write_upload_body(body_path, source_path, "source.y4m")

    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    result: dict[str, Any] = {
        "frames": frames,
        "source_bytes": source_size,
        "upload_body_bytes": body_size,
        "media_budget_bytes": MAX_MEDIA_BYTES,
        "within_budget": source_size <= MAX_MEDIA_BYTES,
        "peak_rss_mb_before_pipeline": round(peak_before, 1),
    }

    with tempfile.TemporaryDirectory(prefix="av-memory-data-") as data_dir:
        server = build_http_server("127.0.0.1", 0, Path(data_dir))
        thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.02},
            daemon=True,
        )
        thread.start()
        host, port = server.server_address[:2]
        try:
            body = _StreamBody(body_path)
            headers = {
                "Content-Type": "application/json",
                "Content-Length": str(body_size),
                "Accept": "application/json",
            }
            started = time.time()
            status, asset = _request(
                str(host), int(port), "POST", "/v1/assets", body, headers
            )
            body.close()
            result["upload_seconds"] = round(time.time() - started, 2)
            if status != 201:
                result["error"] = {"stage": "upload", **asset}
                return result

            started = time.time()
            status, rendition = _request(
                str(host),
                int(port),
                "POST",
                "/v1/renditions",
                json.dumps(
                    {
                        "primary_asset_id": asset["id"],
                        "output_name": "derived.y4m",
                        "operations": CLAMPED_OPERATIONS,
                    }
                ).encode("utf-8"),
                {"Content-Type": "application/json"},
            )
            result["render_seconds"] = round(time.time() - started, 2)
            if status != 201:
                result["error"] = {"stage": "render", **rendition}
                return result

            started = time.time()
            status, check = _request(
                str(host),
                int(port),
                "POST",
                f"/v1/renditions/{rendition['id']}/integrity-checks",
                b"{}",
                {"Content-Type": "application/json"},
            )
            result["check_seconds"] = round(time.time() - started, 2)
            result["rendition_status"] = rendition["status"]
            result["check_status"] = check.get("status")
            result["check_reason"] = check.get("reason")
            result["input_descriptor_frames"] = asset["descriptor"]["properties"][
                "frame_count"
            ]
            result["output_descriptor_frames"] = rendition["output_asset"][
                "descriptor"
            ]["properties"]["frame_count"]
            result["output_sha256"] = rendition["output_asset"]["sha256"]
            result["reproduced_sha256"] = check.get("reproduced_digest")
        finally:
            server.shutdown()
            thread.join(timeout=60)
            server.server_close()

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    result["peak_rss_mb"] = round(peak, 1)
    result["peak_rss_growth_mb"] = round(peak - peak_before, 1)
    return result


def _run_child(frames: int) -> dict[str, Any]:
    import subprocess

    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--frames",
            str(frames),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if process.returncode != 0:
        raise SystemExit(
            f"worker failed for frames={frames}\n{process.stderr[-4000:]}"
        )
    return json.loads(process.stdout.strip().splitlines()[-1])


def _run_legacy_child(frames: int) -> dict[str, Any]:
    import subprocess

    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--legacy-check",
            "--frames",
            str(frames),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if process.returncode != 0:
        raise SystemExit(
            f"legacy check failed for frames={frames}\n{process.stderr[-4000:]}"
        )
    return json.loads(process.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--legacy-check", action="store_true")
    parser.add_argument("--frames", type=int, default=8)
    arguments = parser.parse_args()

    if arguments.worker:
        result = run_worker(arguments.frames)
        print(json.dumps(result, sort_keys=True))
        return 0 if "error" not in result else 1
    if arguments.legacy_check:
        result = run_legacy_check(arguments.frames)
        print(json.dumps(result, sort_keys=True))
        return 0

    failures: list[str] = []
    measurements: list[dict[str, Any]] = []
    for frames in FRAME_COUNTS:
        result = _run_child(frames)
        measurements.append(result)
        print(
            json.dumps(result, sort_keys=True),
            file=sys.stderr,
        )
        for key in ("rendition_status", "check_status"):
            if result.get(key) != ("completed" if key == "rendition_status" else "verified"):
                failures.append(f"frames={frames}: {key}={result.get(key)!r}")
        if not result.get("within_budget", False):
            failures.append(
                f"frames={frames}: source exceeds the media budget"
            )

    largest = measurements[-1]
    legacy = _run_legacy_child(FRAME_COUNTS[-1])
    bytes_match = (
        legacy["streamed_sha256"] == legacy["legacy_sha256"]
        and legacy["streamed_sha256"] == largest.get("output_sha256")
    )
    if not bytes_match:
        failures.append("rendered bytes differ from the legacy algorithm")
    budget_fraction = largest["source_bytes"] / largest["media_budget_bytes"]
    if budget_fraction < 0.90:
        failures.append(
            "largest case is not near the media budget: "
            f"{budget_fraction:.3f}"
        )

    baselines = [item["peak_rss_mb"] for item in measurements]
    for item in measurements:
        excess = item["peak_rss_mb"] - min(baselines)
        if excess > RSS_TOLERANCE_MB:
            failures.append(
                f"frames={item['frames']}: peak RSS exceeds the smallest "
                f"case by {excess:.1f} MB"
            )

    summary = {
        "geometry": [WIDTH, HEIGHT],
        "media_budget_bytes": MAX_MEDIA_BYTES,
        "near_budget_fraction": round(budget_fraction, 4),
        "peak_rss_mb_by_frames": {
            str(item["frames"]): item["peak_rss_mb"] for item in measurements
        },
        "render_seconds_by_frames": {
            str(item["frames"]): item["render_seconds"] for item in measurements
        },
        "bytes_match_legacy": bytes_match,
        "failures": failures,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
