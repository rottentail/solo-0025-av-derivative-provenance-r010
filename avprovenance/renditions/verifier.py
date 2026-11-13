"""Recompute a completed rendition and compare it with recorded output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import RenderFailed, UnsupportedMedia
from ..media import MediaOperationFailure, apply_media_operations
from ..media.pipeline import apply_media_operations_to_target
from ..models import MediaAsset
from ..serialization import sha256_file, sha256_hex


@dataclass(frozen=True)
class VerificationOutcome:
    status: str
    expected_digest: str
    reproduced_digest: str | None
    expected_descriptor: dict[str, Any]
    reproduced_descriptor: dict[str, Any] | None
    reason: str | None


def _mismatch(
    expected_digest: str,
    expected_descriptor: dict[str, Any],
    reason: str,
    *,
    reproduced_digest: str | None = None,
    reproduced_descriptor: dict[str, Any] | None = None,
) -> VerificationOutcome:
    return VerificationOutcome(
        status="mismatch",
        expected_digest=expected_digest,
        reproduced_digest=reproduced_digest,
        expected_descriptor=expected_descriptor,
        reproduced_descriptor=reproduced_descriptor,
        reason=reason,
    )


def reproduce_output(
    *,
    output_asset: MediaAsset,
    recorded_output_bytes: bytes,
    primary_bytes: bytes,
    operations: list[dict[str, Any]],
    secondary_bytes: bytes | None,
) -> VerificationOutcome:
    expected_descriptor = output_asset.descriptor.to_dict()
    expected_digest = output_asset.sha256
    recorded_digest = sha256_hex(recorded_output_bytes)
    if recorded_digest != expected_digest:
        return _mismatch(
            expected_digest,
            expected_descriptor,
            "recorded_output_digest_drift",
        )

    try:
        outcome = apply_media_operations(
            primary_bytes,
            operations,
            secondary_bytes,
        )
    except (MediaOperationFailure, UnsupportedMedia, RenderFailed) as exc:
        reason = getattr(exc, "code", "reproduction_failed")
        return _mismatch(expected_digest, expected_descriptor, str(reason))

    reproduced_digest = sha256_hex(outcome.data)
    reproduced_descriptor = outcome.descriptor.to_dict()
    if reproduced_digest != expected_digest:
        return _mismatch(
            expected_digest,
            expected_descriptor,
            "reproduced_bytes_differ",
            reproduced_digest=reproduced_digest,
            reproduced_descriptor=reproduced_descriptor,
        )
    if reproduced_descriptor != expected_descriptor:
        return _mismatch(
            expected_digest,
            expected_descriptor,
            "reproduced_descriptor_differs",
            reproduced_digest=reproduced_digest,
            reproduced_descriptor=reproduced_descriptor,
        )
    return VerificationOutcome(
        status="verified",
        expected_digest=expected_digest,
        reproduced_digest=reproduced_digest,
        expected_descriptor=expected_descriptor,
        reproduced_descriptor=reproduced_descriptor,
        reason=None,
    )


def reproduce_output_to_target(
    *,
    output_asset: MediaAsset,
    recorded_output_path: Path,
    primary_path: Path,
    operations: list[dict[str, Any]],
    target_path: Path,
    secondary_path: Path | None,
) -> VerificationOutcome:
    """Reproduce into a file and compare by streaming digest.

    Neither the recorded output nor the reproduction is held in memory; both
    sides are hashed from disk with a fixed read buffer.
    """

    expected_descriptor = output_asset.descriptor.to_dict()
    expected_digest = output_asset.sha256
    try:
        recorded_digest, _size = sha256_file(recorded_output_path)
    except OSError:
        return _mismatch(
            expected_digest,
            expected_descriptor,
            "source_or_output_unavailable",
        )
    if recorded_digest != expected_digest:
        return _mismatch(
            expected_digest,
            expected_descriptor,
            "recorded_output_digest_drift",
        )

    try:
        with open(target_path, "wb") as sink:
            outcome = apply_media_operations_to_target(
                primary_path,
                operations,
                sink,
                secondary_path,
            )
    except (MediaOperationFailure, UnsupportedMedia, RenderFailed, OSError) as exc:
        reason = getattr(exc, "code", "reproduction_failed")
        return _mismatch(expected_digest, expected_descriptor, str(reason))

    try:
        reproduced_digest, _size = sha256_file(target_path)
    except OSError:
        return _mismatch(
            expected_digest,
            expected_descriptor,
            "reproduced_output_unavailable",
        )
    reproduced_descriptor = outcome.descriptor.to_dict()
    if reproduced_digest != expected_digest:
        return _mismatch(
            expected_digest,
            expected_descriptor,
            "reproduced_bytes_differ",
            reproduced_digest=reproduced_digest,
            reproduced_descriptor=reproduced_descriptor,
        )
    if reproduced_descriptor != expected_descriptor:
        return _mismatch(
            expected_digest,
            expected_descriptor,
            "reproduced_descriptor_differs",
            reproduced_digest=reproduced_digest,
            reproduced_descriptor=reproduced_descriptor,
        )
    return VerificationOutcome(
        status="verified",
        expected_digest=expected_digest,
        reproduced_digest=reproduced_digest,
        expected_descriptor=expected_descriptor,
        reproduced_descriptor=reproduced_descriptor,
        reason=None,
    )
