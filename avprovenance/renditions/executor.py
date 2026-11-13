"""Execute a normalized plan and convert media failures to domain errors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

from ..errors import RenderFailed, UnsupportedMedia
from ..media import (
    MediaOperationFailure,
    RenderOutcome,
    StreamRenderOutcome,
    apply_media_operations,
    apply_media_operations_to_target,
)


def execute_plan(
    primary_bytes: bytes,
    operations: list[dict[str, Any]],
    secondary_bytes: bytes | None,
) -> RenderOutcome:
    try:
        return apply_media_operations(primary_bytes, operations, secondary_bytes)
    except MediaOperationFailure as exc:
        raise RenderFailed(
            "The rendition could not be produced.",
            details={"reason": exc.code, "detail": exc.message},
        ) from exc
    except UnsupportedMedia as exc:
        raise RenderFailed(
            "The rendition produced unsupported media bytes.",
            details={"reason": exc.code},
        ) from exc


def execute_plan_to_target(
    primary_path: Path,
    operations: list[dict[str, Any]],
    target: BinaryIO,
    secondary_path: Path | None = None,
) -> StreamRenderOutcome:
    """Execute a plan from persisted parents into a streaming sink."""

    try:
        return apply_media_operations_to_target(
            primary_path,
            operations,
            target,
            secondary_path,
        )
    except MediaOperationFailure as exc:
        raise RenderFailed(
            "The rendition could not be produced.",
            details={"reason": exc.code, "detail": exc.message},
        ) from exc
    except UnsupportedMedia as exc:
        raise RenderFailed(
            "The rendition produced unsupported media bytes.",
            details={"reason": exc.code},
        ) from exc
