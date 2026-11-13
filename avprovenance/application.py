"""Application orchestration for ingestion, rendering, and audits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import (
    ApplicationError,
    IntegrityMismatch,
    PersistenceFailure,
    PlanInvalid,
    RenditionNotCompleted,
)
from .models import (
    IntegrityAudit,
    MediaAsset,
    ProvenanceEdge,
    Rendition,
)
from .media import inspect_media, inspect_media_file
from .renditions import (
    compile_plan,
    execute_plan_to_target,
    reproduce_output_to_target,
)
from .serialization import (
    deep_copy,
    new_identifier,
    sha256_file,
    sha256_hex,
    utc_now,
)
from .storage import (
    ContentFileRepository,
    JsonStateFile,
    ProvenanceRepository,
)
from .validation import (
    require_object,
    require_string,
    validate_filename,
)


class MediaProvenanceApplication:
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.content_files = ContentFileRepository(self.data_dir / "content")
        self.state_file = JsonStateFile(self.data_dir / "state.json")
        self.repository = ProvenanceRepository(self.state_file)

    def _rendition_view(self, rendition: Rendition) -> dict[str, Any]:
        value = rendition.to_dict()
        if rendition.output_asset_id:
            try:
                value["output_asset"] = self.repository.get_asset(
                    rendition.output_asset_id
                ).to_public_dict()
            except ApplicationError:
                value["output_asset"] = None
        else:
            value["output_asset"] = None
        return value

    def ingest_asset(self, filename: Any, data: bytes) -> dict[str, Any]:
        safe_name = validate_filename(filename)
        descriptor = inspect_media(data)
        digest = sha256_hex(data)
        content_key = self.content_files.key_for_digest(digest)
        existed = self.content_files.exists(content_key)
        self.content_files.write(data, digest)
        return self._commit_asset(
            safe_name,
            descriptor,
            digest,
            len(data),
            content_key,
            existed,
        )

    def _commit_asset(
        self,
        safe_name: str,
        descriptor,
        digest: str,
        size_bytes: int,
        content_key: str,
        existed: bool,
    ) -> dict[str, Any]:
        asset = MediaAsset(
            id=new_identifier("asset"),
            filename=safe_name,
            kind=descriptor.kind,
            sha256=digest,
            size_bytes=size_bytes,
            content_key=content_key,
            descriptor=descriptor,
            created_at=utc_now(),
        )
        try:
            persisted = self.repository.create_asset(asset)
        except Exception:
            if not existed:
                self.content_files.remove(content_key)
            raise
        return persisted.to_public_dict()

    def ingest_asset_from_file(
        self,
        filename: Any,
        payload_path: Path,
    ) -> dict[str, Any]:
        """Persist an already-spooled upload without loading it into memory.

        The caller owns *payload_path* and removes it if it is still present
        after this call returns or raises.
        """

        safe_name = validate_filename(filename)
        descriptor = inspect_media_file(payload_path)
        digest, size_bytes = sha256_file(payload_path)
        content_key = self.content_files.key_for_digest(digest)
        existed = self.content_files.exists(content_key)
        self.content_files.commit_temporary(payload_path, digest)
        return self._commit_asset(
            safe_name,
            descriptor,
            digest,
            size_bytes,
            content_key,
            existed,
        )

    def get_asset(self, asset_id: str) -> dict[str, Any]:
        return self.repository.get_asset(asset_id).to_public_dict()

    def create_rendition(self, payload: Any) -> dict[str, Any]:
        request = require_object(payload, "request")
        primary_asset_id = require_string(
            request.get("primary_asset_id"),
            "primary_asset_id",
            maximum=96,
        )
        secondary_value = request.get("secondary_asset_id")
        secondary_asset_id = None
        if secondary_value is not None:
            secondary_asset_id = require_string(
                secondary_value,
                "secondary_asset_id",
                maximum=96,
            )
        output_name = validate_filename(request.get("output_name"))

        primary = self.repository.get_asset(primary_asset_id)
        secondary = (
            self.repository.get_asset(secondary_asset_id)
            if secondary_asset_id
            else None
        )
        if secondary is not None and secondary.kind != primary.kind:
            raise PlanInvalid("Primary and secondary assets must share one media kind.")

        plan = compile_plan(
            primary.kind,
            request.get("operations"),
            has_secondary=secondary is not None,
        )
        if secondary is not None and "audio_mix" not in plan.operation_types:
            raise PlanInvalid(
                "A secondary audio asset requires an audio_mix operation."
            )
        if primary.kind == "audio" and not output_name.lower().endswith(".wav"):
            raise PlanInvalid("Audio output_name must end with .wav.")
        if primary.kind == "video" and not output_name.lower().endswith(".y4m"):
            raise PlanInvalid("Video output_name must end with .y4m.")

        now = utc_now()
        rendition = Rendition(
            id=new_identifier("rendition"),
            primary_asset_id=primary.id,
            secondary_asset_id=secondary.id if secondary else None,
            operations=plan.operations,
            plan_digest=plan.digest,
            output_name=output_name,
            status="queued",
            transitions=[{"from": None, "to": "queued", "at": now}],
            output_asset_id=None,
            error_code=None,
            error_message=None,
            created_at=now,
            updated_at=now,
        )
        self.repository.create_rendition(rendition)
        self.repository.transition_rendition(rendition.id, "running")

        primary_path: Path | None = None
        secondary_path: Path | None = None
        output_handle = None
        output_temp_path: Path | None = None
        try:
            primary_path = self.content_files.path_for_key(primary.content_key)
            secondary_path = (
                self.content_files.path_for_key(secondary.content_key)
                if secondary
                else None
            )
            output_handle, output_temp_path = self.content_files.open_temporary()
            outcome = execute_plan_to_target(
                primary_path,
                plan.operations,
                output_handle,
                secondary_path,
            )
            output_handle.flush()
            output_handle.close()
            output_handle = None
            output_digest, output_size = sha256_file(output_temp_path)
            output_key = self.content_files.key_for_digest(output_digest)
            output_existed = self.content_files.exists(output_key)
            self.content_files.commit_temporary(output_temp_path, output_digest)
            output_temp_path = None
            output_asset = MediaAsset(
                id=new_identifier("asset"),
                filename=output_name,
                kind=outcome.descriptor.kind,
                sha256=output_digest,
                size_bytes=output_size,
                content_key=output_key,
                descriptor=outcome.descriptor,
                created_at=utc_now(),
            )
            edge = ProvenanceEdge(
                id=new_identifier("edge"),
                rendition_id=rendition.id,
                parent_asset_ids=[
                    item
                    for item in (primary.id, secondary.id if secondary else None)
                    if item is not None
                ],
                operation_types=list(outcome.operation_types),
                plan_digest=plan.digest,
                output_asset_id=output_asset.id,
                created_at=utc_now(),
            )
            try:
                completed = self.repository.complete_rendition(
                    rendition.id,
                    output_asset,
                    edge,
                )
            except Exception:
                if not output_existed:
                    self.content_files.remove(output_key)
                raise
            return self._rendition_view(completed)
        except ApplicationError as exc:
            current = self.repository.get_rendition(rendition.id)
            if current.status == "running":
                self.repository.fail_rendition(
                    rendition.id,
                    exc.code,
                    exc.message,
                )
            raise
        except Exception as exc:
            current = self.repository.get_rendition(rendition.id)
            if current.status == "running":
                self.repository.fail_rendition(
                    rendition.id,
                    "RENDER_FAILED",
                    "The rendition failed before completion.",
                )
            raise ApplicationError(
                code="RENDER_FAILED",
                message="The rendition failed before completion.",
                status=409,
            ) from exc
        finally:
            if output_handle is not None:
                try:
                    output_handle.close()
                except OSError:
                    pass
            if output_temp_path is not None:
                self.content_files.discard_temporary(output_temp_path)

    def get_rendition(self, rendition_id: str) -> dict[str, Any]:
        return self._rendition_view(self.repository.get_rendition(rendition_id))

    def get_lineage(self, rendition_id: str) -> dict[str, Any]:
        return self.repository.lineage_for(rendition_id)

    def _begin_integrity_audit(self, rendition_id: str) -> IntegrityAudit:
        now = utc_now()
        audit = IntegrityAudit(
            id=new_identifier("audit"),
            rendition_id=rendition_id,
            status="pending",
            expected_digest=None,
            reproduced_digest=None,
            expected_descriptor=None,
            reproduced_descriptor=None,
            reason=None,
            created_at=now,
            updated_at=now,
        )
        return self.repository.insert_integrity_audit(audit)

    def _finish_mismatch(
        self,
        audit: IntegrityAudit,
        reason: str,
        *,
        expected_digest: str | None = None,
        expected_descriptor: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        finished = self.repository.settle_integrity_audit(
            audit.id,
            status="mismatch",
            expected_digest=expected_digest,
            reproduced_digest=None,
            expected_descriptor=expected_descriptor,
            reproduced_descriptor=None,
            reason=reason,
        )
        body = finished.to_dict()
        raise IntegrityMismatch(
            "The recorded rendition did not reproduce exactly.",
            details={
                "audit": body,
                "reason": reason,
            },
        )

    def verify_rendition(self, rendition_id: str) -> dict[str, Any]:
        rendition = self.repository.get_rendition(rendition_id)
        if rendition.status != "completed" or not rendition.output_asset_id:
            raise RenditionNotCompleted(rendition_id, rendition.status)

        audit = self._begin_integrity_audit(rendition_id)
        try:
            edge = self.repository.get_edge(rendition_id)
            output_asset = self.repository.get_asset(rendition.output_asset_id)
            primary_asset = self.repository.get_asset(rendition.primary_asset_id)
            secondary_asset = (
                self.repository.get_asset(rendition.secondary_asset_id)
                if rendition.secondary_asset_id
                else None
            )
        except ApplicationError:
            self._finish_mismatch(audit, "provenance_unavailable")

        expected_descriptor = output_asset.descriptor.to_dict()
        recorded_output_path: Path | None = None
        primary_path: Path | None = None
        secondary_path: Path | None = None
        reproduction_temp_path: Path | None = None
        try:
            recorded_output_path = self.content_files.path_for_key(
                output_asset.content_key
            )
            primary_path = self.content_files.path_for_key(
                primary_asset.content_key
            )
            secondary_path = (
                self.content_files.path_for_key(secondary_asset.content_key)
                if secondary_asset
                else None
            )
            for path in (
                recorded_output_path,
                primary_path,
                secondary_path,
            ):
                if path is not None and not path.is_file():
                    raise PersistenceFailure("A referenced media file is missing.")
        except ApplicationError:
            self._finish_mismatch(
                audit,
                "source_or_output_unavailable",
                expected_digest=output_asset.sha256,
                expected_descriptor=expected_descriptor,
            )

        if edge.plan_digest != rendition.plan_digest:
            self._finish_mismatch(
                audit,
                "plan_digest_drift",
                expected_digest=output_asset.sha256,
                expected_descriptor=expected_descriptor,
            )
        if edge.parent_asset_ids != [
            item
            for item in (
                rendition.primary_asset_id,
                rendition.secondary_asset_id,
            )
            if item is not None
        ]:
            self._finish_mismatch(
                audit,
                "parent_set_drift",
                expected_digest=output_asset.sha256,
                expected_descriptor=expected_descriptor,
            )

        try:
            reproduction_handle, reproduction_temp_path = (
                self.content_files.open_temporary()
            )
            reproduction_handle.close()
            outcome = reproduce_output_to_target(
                output_asset=output_asset,
                recorded_output_path=recorded_output_path,
                primary_path=primary_path,
                operations=deep_copy(rendition.operations),
                target_path=reproduction_temp_path,
                secondary_path=secondary_path,
            )
        except ApplicationError:
            self._finish_mismatch(
                audit,
                "source_or_output_unavailable",
                expected_digest=output_asset.sha256,
                expected_descriptor=expected_descriptor,
            )
        finally:
            if reproduction_temp_path is not None:
                self.content_files.discard_temporary(reproduction_temp_path)
        finished = self.repository.settle_integrity_audit(
            audit.id,
            status=outcome.status,
            expected_digest=outcome.expected_digest,
            reproduced_digest=outcome.reproduced_digest,
            expected_descriptor=outcome.expected_descriptor,
            reproduced_descriptor=outcome.reproduced_descriptor,
            reason=outcome.reason,
        )
        if finished.status == "verified":
            return finished.to_dict()
        raise IntegrityMismatch(
            "The recorded rendition did not reproduce exactly.",
            details={
                "audit": finished.to_dict(),
                "reason": finished.reason,
            },
        )

    def close(self) -> None:
        return
