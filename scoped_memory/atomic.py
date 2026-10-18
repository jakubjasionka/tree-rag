"""Crash-safe atomic persistence for JSON documents and FAISS vector indices."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import faiss


def _fsync_directory(dir_path: Path) -> None:
    """Best-effort fsync on parent directory descriptor for POSIX durability."""
    try:
        dir_fd = os.open(str(dir_path), os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def _atomic_write_json(path: Path | str, data: Any, indent: int = 2) -> None:
    """Atomically persist JSON via temporary file, file fsync, directory fsync, and POSIX rename.

    Guarantees that concurrent readers or file-synchronization daemons never observe
    a truncated or half-written `docs.json` file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.tmp.")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
            f.write(chr(10))
            f.flush()
            os.fsync(f.fileno())
        _fsync_directory(target.parent)
        os.rename(tmp_path, str(target))
        _fsync_directory(target.parent)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def atomic_write_json(path: Path | str, data: Any, indent: int = 2) -> None:
    """Public alias for `_atomic_write_json`."""
    _atomic_write_json(path, data, indent=indent)


def _atomic_write_faiss(index: faiss.Index, path: Path | str) -> None:
    """Atomically persist a FAISS index via temporary file, fsync, and POSIX rename."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + f".tmp.{uuid.uuid4().hex[:8]}")
    try:
        faiss.write_index(index, str(tmp_path))
        try:
            with open(tmp_path, "rb") as f:
                os.fsync(f.fileno())
        except OSError:
            pass
        _fsync_directory(target.parent)
        os.rename(str(tmp_path), str(target))
        _fsync_directory(target.parent)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def atomic_write_faiss(index: faiss.Index, path: Path | str) -> None:
    """Public alias for `_atomic_write_faiss`."""
    _atomic_write_faiss(index, path)
