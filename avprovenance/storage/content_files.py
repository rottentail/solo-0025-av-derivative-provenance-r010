"""Content-addressed media files with path checks and atomic writes."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import BinaryIO

from ..errors import PersistenceFailure
from ..serialization import sha256_file, sha256_hex


class ContentFileRepository:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key_for_digest(digest: str) -> str:
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise PersistenceFailure("A content digest has an invalid shape.")
        return f"{digest[:2]}/{digest[2:4]}/{digest}"

    def _path_for_key(self, key: str) -> Path:
        parts = Path(key).parts
        if len(parts) != 3:
            raise PersistenceFailure("A content key has an invalid shape.")
        root = self.root.resolve()
        candidate = (self.root / key).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise PersistenceFailure("A content key escapes the data root.") from exc
        return candidate

    def path_for_key(self, key: str) -> Path:
        """Return the absolute blob path for a validated content key."""

        return self._path_for_key(key)

    def write(self, data: bytes, digest: str) -> str:
        if sha256_hex(data) != digest:
            raise PersistenceFailure("Content digest does not match the supplied bytes.")
        key = self.key_for_digest(digest)
        target = self._path_for_key(key)
        if target.exists():
            return key
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".content-",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        )
        temporary_path = Path(temporary.name)
        try:
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary.close()
            os.replace(temporary_path, target)
        except OSError as exc:
            temporary.close()
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise PersistenceFailure("Media bytes could not be saved.") from exc
        return key

    def open_temporary(self) -> tuple[BinaryIO, Path]:
        """Create a temporary sibling file for a streaming content write."""

        self.root.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".content-",
            suffix=".tmp",
            dir=self.root,
            delete=False,
        )
        return temporary, Path(temporary.name)

    def commit_temporary(self, temporary_path: Path, digest: str) -> str:
        """Verify and atomically place a streamed temporary file at its key."""

        if not digest:
            raise PersistenceFailure("A content digest has an invalid shape.")
        key = self.key_for_digest(digest)
        target = self._path_for_key(key)
        if target.exists():
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            return key
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(temporary_path, target)
        except OSError as exc:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise PersistenceFailure("Media bytes could not be saved.") from exc
        return key

    def discard_temporary(self, temporary_path: Path) -> None:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass

    def read(self, key: str) -> bytes:
        path = self._path_for_key(key)
        try:
            return path.read_bytes()
        except OSError as exc:
            raise PersistenceFailure("A referenced media file is missing.") from exc

    def consume(self, temporary_path: Path, digest: str) -> str:
        """Verify a finished temporary file, publishing it under its digest."""

        computed, _size = sha256_file(temporary_path)
        if computed != digest:
            self.discard_temporary(temporary_path)
            raise PersistenceFailure(
                "Content digest does not match the supplied bytes."
            )
        return self.commit_temporary(temporary_path, digest)

    def remove(self, key: str) -> None:
        path = self._path_for_key(key)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            return

    def exists(self, key: str) -> bool:
        return self._path_for_key(key).is_file()
