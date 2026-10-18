"""Concurrency guard, read-only pending queue (`_pending/<ws>/*.md`), and conflict-free merge."""

from __future__ import annotations

import glob
import json
import os
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from scoped_memory.atomic import _atomic_write_faiss, _atomic_write_json
from scoped_memory.embed import Embedder
from scoped_memory.index import ALLOWED_CATEGORIES, WorkspaceIndex
from scoped_memory.locks import (
    _release_flock,
    _try_flock_nonblock,
    _try_global_flock_nonblock,
)
from scoped_memory.sanitize import normalize_text, sanitize, text_hash

DEFAULT_PEER_URI = "http://peer-node.example:PORT/v1"


@dataclass(frozen=True)
class PeerNode:
    """Abstract peer descriptor for multi-node vector store synchronization."""

    node_id: str
    endpoint_url: str = DEFAULT_PEER_URI
    enabled: bool = True


@dataclass(frozen=True)
class SyncPolicy:
    """Policy governing conflict-free multi-node merging and pending queue compaction."""

    near_duplicate_threshold: float = 0.975
    mtime_stability_sec: float = 0.0
    auto_heal_conflicts: bool = True
    ttl_guard_cache_sec: float = 2.0


def write_pending_markdown(
    base_dir: Path,
    workspace: str,
    category: str,
    text: str,
    source: str = "manual",
) -> Path:
    """Persist a write to `_pending/<workspace>/<timestamp>_<id>.md` with YAML frontmatter.

    Used when the workspace is in read-only mode (active concurrent writer, lock contention,
    or unresolved sync-conflict file) so zero agent observations are lost.
    """
    ws = sanitize(workspace)
    pending_dir = base_dir / "_pending" / ws
    pending_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{time.strftime('%Y-%m-%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.md"
    path = pending_dir / fname
    cat = (category or "note").lower()
    if cat not in ALLOWED_CATEGORIES:
        cat = "note"
    nl = chr(10)
    frontmatter = nl.join(
        [
            "---",
            f"category: {cat}",
            f"workspace: {ws}",
            f"source: {source or 'manual'}",
            f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "---",
            f"{normalize_text(text)}",
            "",
        ]
    )
    path.write_text(frontmatter, encoding="utf-8")
    return path


def parse_pending_markdown(path: Path) -> tuple[str, str, str, str]:
    """Parse a `_pending/<ws>/*.md` file into `(category, body, source, workspace)`."""
    raw = path.read_text(encoding="utf-8")
    nl = chr(10)
    pattern = f"^---{nl}(.*?){nl}---{nl}(.*)$"
    m = re.match(pattern, raw, re.DOTALL)
    if not m:
        return ("note", raw.strip(), f"pending:{path.name}", path.parent.name)
    front, body = m.group(1), m.group(2)
    meta: dict[str, str] = {}
    for line in front.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    category = meta.get("category", "note").lower()
    if category not in ALLOWED_CATEGORIES:
        category = "note"
    source = meta.get("source", f"pending:{path.name}")
    workspace = sanitize(meta.get("workspace", path.parent.name))
    return (category, body.strip(), source, workspace)


def pending_count(base_dir: Path, workspace: str | None = None) -> int:
    """Count queued markdown entries in `_pending` (globally or for a single workspace)."""
    pending_root = base_dir / "_pending"
    if not pending_root.exists():
        return 0
    if workspace:
        ws = sanitize(workspace)
        ws_dir = pending_root / ws
        if not ws_dir.exists():
            return 0
        return sum(1 for p in ws_dir.rglob("*.md") if not p.is_symlink())
    return sum(1 for p in pending_root.rglob("*.md") if not p.is_symlink())


def pending_health(base_dir: Path) -> dict[str, int]:
    """Return pending queue counts grouped by workspace."""
    pending_root = base_dir / "_pending"
    if not pending_root.exists():
        return {}
    counts: dict[str, int] = {}
    for p in pending_root.rglob("*.md"):
        if p.is_symlink():
            continue
        ws = p.parent.name
        counts[ws] = counts.get(ws, 0) + 1
    return counts


def has_sync_conflict(base_dir: Path, workspace: str | None = None) -> str | None:
    """Detect any `*.sync-conflict-*` artifacts in `base_dir` or a target workspace."""
    patterns: list[str] = []
    if workspace:
        ws = sanitize(workspace)
        for candidate in {workspace, ws}:
            patterns.append(str(base_dir / candidate / "*sync-conflict*"))
    patterns.append(str(base_dir / "*sync-conflict*"))
    patterns.append(str(base_dir / "_pending" / "*sync-conflict*"))
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    return None


def load_docs_and_vectors(
    docs_path: Path, faiss_path: Path, expected_dim: int
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Load `docs.json` and reconstruct `(text_hash -> vector)` from `index.faiss`."""
    docs: list[dict[str, Any]] = []
    vec_map: dict[str, np.ndarray] = {}
    if docs_path.exists():
        try:
            raw = json.loads(docs_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                docs = [
                    d
                    for d in raw
                    if isinstance(d, dict) and normalize_text(str(d.get("text", "")))
                ]
        except Exception:
            docs = []

    if faiss_path.exists() and docs:
        try:
            idx = faiss.read_index(str(faiss_path))
            if int(idx.d) == expected_dim and int(idx.ntotal) == len(docs):
                for i, d in enumerate(docs):
                    h = text_hash(str(d.get("text", "")))
                    vec_map[h] = idx.reconstruct(i).astype("float32")
        except Exception:
            pass
    return docs, vec_map


def merge_doc_collections(
    collections: list[tuple[list[dict[str, Any]], dict[str, np.ndarray]]],
    embedder: Embedder,
    near_dup_threshold: float = 0.975,
) -> tuple[list[dict[str, Any]], faiss.IndexFlatIP]:
    """Losslessly merge multiple `(docs, vec_map)` collections into a unified `(docs, IndexFlatIP)`.

    Reuses reconstructed FAISS vectors whenever available and only embeds newly discovered
    documents whose content hashes are missing from `vec_pool`.
    """
    dim = embedder.dim
    by_hash: dict[str, dict[str, Any]] = {}
    vec_pool: dict[str, np.ndarray] = {}

    for docs, vmap in collections:
        for h, v in vmap.items():
            if h not in vec_pool and v is not None and v.shape == (dim,):
                vec_pool[h] = v
        for d in docs:
            txt = normalize_text(str(d.get("text", "")))
            if not txt:
                continue
            h = text_hash(txt)
            existing = by_hash.get(h)
            if existing is None:
                doc_copy = dict(d)
                doc_copy["text"] = txt
                by_hash[h] = doc_copy
            else:
                t_new = str(d.get("updated_at") or d.get("time") or "")
                t_old = str(existing.get("updated_at") or existing.get("time") or "")
                if t_new > t_old:
                    merged_doc = dict(existing)
                    merged_doc.update(d)
                    merged_doc["text"] = txt
                    by_hash[h] = merged_doc

    merged_docs = list(by_hash.values())
    merged_docs.sort(key=lambda x: (str(x.get("time") or ""), str(x.get("id") or "")))

    missing_indices: list[int] = []
    missing_texts: list[str] = []
    for i, d in enumerate(merged_docs):
        h = text_hash(str(d.get("text", "")))
        if h not in vec_pool:
            missing_indices.append(i)
            missing_texts.append(str(d.get("text", "")))

    if missing_texts:
        computed = embedder.embed_passages(missing_texts)
        for idx_pos, vec in zip(missing_indices, computed, strict=False):
            h = text_hash(str(merged_docs[idx_pos].get("text", "")))
            vec_pool[h] = vec

    final_docs: list[dict[str, Any]] = []
    index = faiss.IndexFlatIP(dim)
    for d in merged_docs:
        h = text_hash(str(d.get("text", "")))
        vec = vec_pool.get(h)
        if vec is None:
            continue
        vec_2d = np.asarray(vec, dtype="float32").reshape(1, -1)
        if int(index.ntotal) > 0:
            scores, positions = index.search(vec_2d, 1)
            top_score = float(scores[0][0])
            top_pos = int(positions[0][0])
            if top_score >= near_dup_threshold and 0 <= top_pos < len(final_docs):
                continue
        index.add(vec_2d)
        final_docs.append(d)

    return final_docs, index


def heal_workspace_conflicts(
    ws_dir: Path,
    embedder: Embedder,
    policy: SyncPolicy | None = None,
) -> dict[str, Any]:
    """Resolve `*.sync-conflict-*` files in `ws_dir` via lossless union merge."""
    active_policy = policy or SyncPolicy()
    ws_dir.mkdir(parents=True, exist_ok=True)
    docs_path = ws_dir / "docs.json"
    faiss_path = ws_dir / "index.faiss"

    conflict_docs = sorted(ws_dir.glob("docs.json*sync-conflict*"))
    conflict_faiss = sorted(ws_dir.glob("index.faiss*sync-conflict*"))
    other_conflicts = [
        p
        for p in ws_dir.glob("*sync-conflict*")
        if p not in conflict_docs and p not in conflict_faiss
    ]

    if not docs_path.exists():
        _atomic_write_json(docs_path, [])

    base_docs, base_vecs = load_docs_and_vectors(docs_path, faiss_path, embedder.dim)
    collections: list[tuple[list[dict[str, Any]], dict[str, np.ndarray]]] = [
        (base_docs, base_vecs)
    ]
    no_faiss = ws_dir / ".none.faiss"
    for i, cdoc in enumerate(conflict_docs):
        cf_path = conflict_faiss[i] if i < len(conflict_faiss) else no_faiss
        c_docs, c_vecs = load_docs_and_vectors(cdoc, cf_path, embedder.dim)
        collections.append((c_docs, c_vecs))

    needs_write = bool(conflict_docs or conflict_faiss)
    if not needs_write:
        if not faiss_path.exists():
            needs_write = True
        else:
            try:
                idx = faiss.read_index(str(faiss_path))
                if int(idx.ntotal) != len(base_docs) or int(idx.d) != embedder.dim:
                    needs_write = True
            except Exception:
                needs_write = True

    if needs_write:
        final_docs, final_index = merge_doc_collections(
            collections,
            embedder=embedder,
            near_dup_threshold=active_policy.near_duplicate_threshold,
        )
        _atomic_write_json(docs_path, final_docs)
        _atomic_write_faiss(final_index, faiss_path)
        base_docs = final_docs

    removed = 0
    for p in [*conflict_docs, *conflict_faiss, *other_conflicts]:
        try:
            p.unlink(missing_ok=True)
            removed += 1
        except OSError:
            pass

    return {
        "workspace": ws_dir.name,
        "docs": len(base_docs),
        "rebuilt_or_merged": needs_write,
        "removed_conflicts": removed,
    }


class ConcurrencyGuard:
    """Controls read-only fallback to `_pending/<ws>/*.md` and conflict-free FAISS compaction."""

    def __init__(
        self,
        base_dir: Path,
        read_only: bool = False,
        detector: Callable[[], tuple[bool, str]] | None = None,
        policy: SyncPolicy | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.read_only = read_only
        self.detector = detector
        self.policy = policy or SyncPolicy()
        self._cache_ts: float = 0.0
        self._cache_blocked: bool = False
        self._cache_detail: str = ""

    def is_blocked(self, workspace: str | None = None) -> tuple[bool, str]:
        """Return `(blocked, reason)` for routing writes to `_pending`."""
        if (
            os.getenv("MEMORY_GUARD_DISABLE", "0") == "1"
            or os.getenv("MEMORY_GUARD_DISABLE_CONCURRENCY", "0") == "1"
        ):
            return False, ""

        if self.read_only:
            return True, "explicit read_only mode active"

        conflict = has_sync_conflict(self.base_dir, workspace=workspace)
        if conflict:
            return True, f"unresolved sync-conflict: {Path(conflict).name}"

        if self.detector is not None:
            now = time.time()
            if now - self._cache_ts < self.policy.ttl_guard_cache_sec:
                return self._cache_blocked, self._cache_detail
            blocked, detail = self.detector()
            self._cache_ts = now
            self._cache_blocked = blocked
            self._cache_detail = detail
            return blocked, detail

        return False, ""

    def invalidate_cache(self) -> None:
        self._cache_ts = 0.0

    def enqueue_pending(
        self,
        workspace: str,
        category: str,
        text: str,
        source: str = "manual",
    ) -> Path:
        """Write an observation to `_pending/<workspace>/*.md`."""
        return write_pending_markdown(
            self.base_dir,
            workspace=workspace,
            category=category,
            text=text,
            source=source,
        )

    def merge_pending_to_faiss(
        self,
        embedder: Embedder,
        workspace: str | None = None,
        force: bool = False,
        mtime_delay: float | None = None,
    ) -> dict[str, Any]:
        """Drain `_pending/<ws>/*.md` into workspace FAISS indices when locks are free."""
        pending_root = self.base_dir / "_pending"
        if not pending_root.exists():
            return {"ok": True, "merged": 0, "deduped": 0, "remaining": 0}

        if not force:
            blocked, detail = self.is_blocked(workspace=workspace)
            if blocked:
                return {
                    "ok": False,
                    "reason": f"guard active ({detail})",
                    "merged": 0,
                    "deduped": 0,
                    "remaining": pending_count(self.base_dir, workspace=workspace),
                }

        acquired_global, global_fh = _try_global_flock_nonblock(self.base_dir)
        if not acquired_global:
            return {
                "ok": False,
                "reason": "global lock held by another merger",
                "merged": 0,
                "deduped": 0,
                "remaining": pending_count(self.base_dir, workspace=workspace),
            }

        delay = self.policy.mtime_stability_sec if mtime_delay is None else mtime_delay
        try:
            if workspace:
                ws_target = sanitize(workspace)
                mds = sorted(
                    p
                    for p in (pending_root / ws_target).glob("*.md")
                    if not p.is_symlink()
                )
            else:
                mds = sorted(p for p in pending_root.rglob("*.md") if not p.is_symlink())

            if not mds:
                return {"ok": True, "merged": 0, "deduped": 0, "remaining": 0}

            by_ws: dict[str, list[tuple[str, str, str, Path]]] = {}
            now = time.time()
            for p in mds:
                if delay > 0 and (now - p.stat().st_mtime) < delay:
                    continue
                cat, body, src, ws = parse_pending_markdown(p)
                if not body.strip():
                    p.unlink(missing_ok=True)
                    continue
                by_ws.setdefault(ws, []).append((cat, body, src, p))

            total_merged = 0
            total_deduped = 0
            for ws, items in by_ws.items():
                ws_acquired, ws_fh = _try_flock_nonblock(self.base_dir, ws)
                if not ws_acquired:
                    continue
                try:
                    ws_index = WorkspaceIndex(self.base_dir / ws, embedder=embedder)
                    existing_hashes = {text_hash(str(d.get("text", ""))) for d in ws_index.docs}
                    seen = set(existing_hashes)
                    to_add: list[tuple[str, str, str]] = []
                    expected_hashes: list[str] = []
                    for c, t, s, _ in items:
                        h = text_hash(t)
                        if h not in seen:
                            seen.add(h)
                            to_add.append((c, t, s))
                            expected_hashes.append(h)

                    added = ws_index.add_many(to_add)
                    added_set = set(expected_hashes[:added])
                    for _, t, _, p in items:
                        h = text_hash(t)
                        if h in added_set or h in existing_hashes:
                            p.unlink(missing_ok=True)
                            if h in existing_hashes and h not in added_set:
                                total_deduped += 1
                    total_merged += added
                finally:
                    _release_flock(ws_fh)

            return {
                "ok": True,
                "merged": total_merged,
                "deduped": total_deduped,
                "remaining": pending_count(self.base_dir, workspace=workspace),
            }
        finally:
            _release_flock(global_fh)
