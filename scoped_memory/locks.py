"""Per-workspace and global POSIX file locking (`flock`) for conflict-free concurrency."""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

from scoped_memory.sanitize import sanitize


def _lock_path_for_workspace(base_dir: Path, workspace: str | None = None) -> Path:
    """Resolve the lock file path for a specific workspace branch or the global store."""
    if workspace is None:
        return base_dir / ".global.lock"
    ws = sanitize(workspace)
    return base_dir / ws / ".lock"


def lock_path_for_workspace(base_dir: Path, workspace: str | None = None) -> Path:
    """Public alias for `_lock_path_for_workspace`."""
    return _lock_path_for_workspace(base_dir, workspace)


def _try_flock_nonblock(
    base_dir: Path, workspace: str | None = None
) -> tuple[bool, IO[str] | None]:
    """Attempt to acquire an exclusive non-blocking POSIX flock for a workspace branch.

    Isolates write serialization to a single workspace (`<base_dir>/<ws>/.lock`)
    so concurrent writers on other workspaces proceed in parallel without contention.
    """
    lock_path = _lock_path_for_workspace(base_dir, workspace)
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True, fh
        except BlockingIOError:
            fh.close()
            return False, None
        except Exception:
            fh.close()
            return False, None
    except Exception:
        return False, None


def try_workspace_lock(base_dir: Path, workspace: str | None = None) -> tuple[bool, IO[str] | None]:
    """Public alias for `_try_flock_nonblock`."""
    return _try_flock_nonblock(base_dir, workspace)


def _try_global_flock_nonblock(base_dir: Path) -> tuple[bool, IO[str] | None]:
    """Attempt to acquire the global exclusive non-blocking lock (`<base_dir>/.global.lock`)."""
    return _try_flock_nonblock(base_dir, workspace=None)


def try_global_lock(base_dir: Path) -> tuple[bool, IO[str] | None]:
    """Public alias for `_try_global_flock_nonblock`."""
    return _try_global_flock_nonblock(base_dir)


def _release_flock(fh: IO[str] | None) -> None:
    """Release an acquired POSIX flock and close its file descriptor."""
    if fh is None:
        return
    try:
        fcntl.flock(fh, fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        fh.close()
    except Exception:
        pass


def release_lock(fh: IO[str] | None) -> None:
    """Public alias for `_release_flock`."""
    _release_flock(fh)


@contextmanager
def workspace_lock(base_dir: Path, workspace: str | None = None) -> Iterator[bool]:
    """Context manager yielding `True` if the non-blocking lock was acquired, `False` otherwise."""
    acquired, fh = _try_flock_nonblock(base_dir, workspace)
    try:
        yield acquired
    finally:
        _release_flock(fh)
