"""`ScopedMemoryStore`: high-level facade for hierarchical, zero-pollution agent memory."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from scoped_memory.embed import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    Embedder,
    MockEmbedder,
    chunk_text,
)
from scoped_memory.guard import (
    ConcurrencyGuard,
    PeerNode,
    SyncPolicy,
    has_sync_conflict,
    heal_workspace_conflicts,
    pending_count,
    pending_health,
)
from scoped_memory.hierarchy import (
    EmptyContext,
    HierarchyEngine,
    SearchHit,
    SearchResult,
    requires_confirmation,
)
from scoped_memory.index import ALLOWED_CATEGORIES, WorkspaceIndex
from scoped_memory.locks import (
    _lock_path_for_workspace,
    _release_flock,
    _try_flock_nonblock,
)
from scoped_memory.sanitize import normalize_text, sanitize

IGNORED_DIRS = {"_pending", "backups", "_archived", "__pycache__"}
_List = list


class ScopedMemoryStore:
    """Hierarchical Scoped Memory Engine (Tree-RAG) with zero-pollution and HITL guarantees.

    Features:
    - Per-workspace isolated FAISS `IndexFlatIP` + `docs.json` namespaces.
    - Deterministic `EmptyContext` on empty or sub-threshold local searches (zero pollution).
    - Human-in-the-Loop (`requires_confirmation()`) gate before global cross-workspace escalation.
    - SHA-256 content deduplication and semantic near-duplicate filtering.
    - Crash-safe atomic writes (`tmp + fsync + rename`) and per-workspace POSIX `flock`.
    - Read-only fallback to `_pending/<ws>/*.md` queue with conflict-free compaction.
    """

    def __init__(
        self,
        base_dir: str | Path | None = None,
        default_workspace: str | None = None,
        embedder: Embedder | None = None,
        min_score: float | None = None,
        read_only: bool = False,
        require_global_confirmation: bool = False,
        confirmation_callback: Callable[[str, str], bool] | None = None,
        concurrency_detector: Callable[[], tuple[bool, str]] | None = None,
        sync_policy: SyncPolicy | None = None,
        peers: Sequence[PeerNode] | None = None,
    ) -> None:
        resolved_base = (
            Path(base_dir)
            if base_dir is not None
            else Path(os.environ.get("TREE_RAG_BASE_DIR", "./.treerag_data"))
        )
        self.base_dir = resolved_base.resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "_pending").mkdir(parents=True, exist_ok=True)

        raw_ws = (
            default_workspace
            or os.environ.get("TREE_RAG_DEFAULT_WORKSPACE")
            or os.environ.get("MEMORY_WORKSPACE")
            or "default"
        )
        self.default_workspace = sanitize(raw_ws)
        self.embedder: Embedder = embedder if embedder is not None else MockEmbedder()
        self.min_score = (
            float(min_score)
            if min_score is not None
            else float(os.environ.get("TREE_RAG_MIN_SCORE", "0.25"))
        )
        self.sync_policy = sync_policy or SyncPolicy()
        self.peers: list[PeerNode] = list(peers) if peers else []
        self.guard = ConcurrencyGuard(
            base_dir=self.base_dir,
            read_only=read_only,
            detector=concurrency_detector,
            policy=self.sync_policy,
        )
        self.hierarchy = HierarchyEngine(
            min_score=self.min_score,
            require_global_confirmation=require_global_confirmation,
            confirmation_callback=confirmation_callback,
        )
        self._cache: dict[str, WorkspaceIndex] = {}

    @property
    def read_only(self) -> bool:
        return self.guard.read_only

    @read_only.setter
    def read_only(self, value: bool) -> None:
        self.guard.read_only = bool(value)

    def _resolve_workspace(self, workspace: str | None = None) -> str:
        return sanitize(workspace or self.default_workspace)

    def get_workspace_index(self, workspace: str | None = None) -> WorkspaceIndex:
        """Return or initialize the `WorkspaceIndex` for a given workspace branch."""
        ws = self._resolve_workspace(workspace)
        if ws not in self._cache:
            self._cache[ws] = WorkspaceIndex(self.base_dir / ws, embedder=self.embedder)
        else:
            self._cache[ws]._check_stale_and_reload()
        return self._cache[ws]

    def all_workspace_indices(self) -> dict[str, WorkspaceIndex]:
        """Discover all initialized workspace branches under `base_dir`."""
        stores: dict[str, WorkspaceIndex] = {}
        seen_inodes: set[int] = set()
        if not self.base_dir.exists():
            return stores

        for d in sorted(self.base_dir.iterdir()):
            if d.name in IGNORED_DIRS or d.name.startswith("."):
                continue
            if d.is_symlink() or not d.is_dir():
                continue
            if not (d / "docs.json").exists():
                continue
            try:
                ino = d.stat().st_ino
            except OSError:
                ino = None
            if ino is not None:
                if ino in seen_inodes:
                    continue
                seen_inodes.add(ino)
            stores[d.name] = self.get_workspace_index(d.name)
        return stores

    def save(
        self,
        category: str,
        text: str,
        source: str = "",
        workspace: str | None = None,
        tags: Sequence[str] | str | None = None,
    ) -> dict[str, Any]:
        """Save an observation to a workspace branch with secret scrubbing, dedup, and atomic write.

        If the concurrency guard is active or the workspace lock is held by another writer,
        persists the observation to `_pending/<workspace>/*.md` (read-only mode) without data loss.
        """
        ws = self._resolve_workspace(workspace)
        client_id = os.getenv("MEMORY_CLIENT_ID", "").strip()
        resolved_source = source or "manual"
        if client_id and f"[{client_id}]" not in resolved_source and resolved_source != client_id:
            resolved_source = f"{resolved_source} [{client_id}]" if source else client_id

        cat = (category or "note").lower()
        if cat not in ALLOWED_CATEGORIES:
            cat = "note"

        norm_text = normalize_text(text)
        if not norm_text:
            return {
                "ok": False,
                "status": "empty",
                "workspace": ws,
                "message": "memory_save: empty text — nothing saved.",
            }

        if len(norm_text) > 8000:
            chunks = chunk_text(norm_text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            chunk_results = [
                self.save(category=cat, text=ch, source=resolved_source, workspace=ws, tags=tags)
                for ch in chunks
            ]
            saved_count = sum(
                1 for r in chunk_results if r.get("status") in {"saved", "duplicate"}
            )
            first = chunk_results[0] if chunk_results else {}
            first_msg = first.get("message", "")
            return {
                **first,
                "chunks_total": len(chunks),
                "chunks_saved": saved_count,
                "message": f"{first_msg} (auto-split: {saved_count}/{len(chunks)} chunks)",
            }

        ws_index = self.get_workspace_index(ws)
        dup = ws_index.find_duplicate_by_hash(norm_text)
        if dup is not None:
            return {
                "ok": True,
                "status": "duplicate",
                "id": dup["id"],
                "workspace": ws,
                "category": dup.get("category", cat),
                "doc": dup,
                "message": f"Duplicate — entry already exists in [{ws}] (id={dup['id']}).",
            }

        blocked, detail = self.guard.is_blocked(workspace=ws)
        if blocked:
            pending_path = self.guard.enqueue_pending(
                workspace=ws,
                category=cat,
                text=norm_text,
                source=resolved_source,
            )
            rel = (
                pending_path.relative_to(self.base_dir)
                if pending_path.is_relative_to(self.base_dir)
                else pending_path
            )
            return {
                "ok": True,
                "status": "pending",
                "workspace": ws,
                "category": cat,
                "pending_path": str(pending_path),
                "reason": detail,
                "message": f"Read-only mode ({detail}) — queued to {rel}",
            }

        acquired, fh = _try_flock_nonblock(self.base_dir, ws)
        if not acquired:
            pending_path = self.guard.enqueue_pending(
                workspace=ws,
                category=cat,
                text=norm_text,
                source=resolved_source,
            )
            rel = (
                pending_path.relative_to(self.base_dir)
                if pending_path.is_relative_to(self.base_dir)
                else pending_path
            )
            return {
                "ok": True,
                "status": "pending",
                "workspace": ws,
                "category": cat,
                "pending_path": str(pending_path),
                "reason": f"lock busy for {ws}",
                "message": f"Read-only mode (lock busy for {ws}) — queued to {rel}",
            }

        try:
            ws_index._check_stale_and_reload()
            dup = ws_index.find_duplicate_by_hash(norm_text)
            if dup is not None:
                return {
                    "ok": True,
                    "status": "duplicate",
                    "id": dup["id"],
                    "workspace": ws,
                    "category": dup.get("category", cat),
                    "doc": dup,
                    "message": f"Duplicate — entry already exists in [{ws}] (id={dup['id']}).",
                }

            try:
                vec = self.embedder.embed_passages([norm_text])[0]
            except Exception as exc:
                pending_path = self.guard.enqueue_pending(
                    workspace=ws,
                    category=cat,
                    text=norm_text,
                    source=resolved_source,
                )
                return {
                    "ok": True,
                    "status": "pending",
                    "workspace": ws,
                    "category": cat,
                    "pending_path": str(pending_path),
                    "reason": f"embedder error: {exc}",
                    "message": f"Embedder unavailable ({exc}) — queued to {pending_path.name}",
                }

            status, doc = ws_index.append_document(
                category=cat,
                normalized_text=norm_text,
                vec=vec,
                source=resolved_source,
                tags=tags,
            )
            if status == "near_duplicate":
                return {
                    "ok": True,
                    "status": "near_duplicate",
                    "id": doc["id"],
                    "workspace": ws,
                    "category": doc.get("category", cat),
                    "doc": doc,
                    "message": (
                        f"Near-duplicate (cosine={doc.get('similarity')}) — "
                        f"already in [{ws}] (id={doc['id']})."
                    ),
                }

            return {
                "ok": True,
                "status": "saved",
                "id": doc["id"],
                "workspace": ws,
                "category": doc["category"],
                "layer": doc["layer"],
                "doc": doc,
                "message": (
                    f"Saved to [{ws}] (id={doc['id']}, "
                    f"category={doc['category']}, layer={doc['layer']})."
                ),
            }
        finally:
            _release_flock(fh)

    def add_many(
        self,
        items: Sequence[tuple[str, str, str]],
        workspace: str | None = None,
    ) -> int:
        """Batch-insert `(category, text, source)` tuples into a workspace branch."""
        ws = self._resolve_workspace(workspace)
        acquired, fh = _try_flock_nonblock(self.base_dir, ws)
        if not acquired:
            for cat, txt, src in items:
                if txt and normalize_text(txt):
                    self.guard.enqueue_pending(ws, cat, txt, src)
            return 0
        try:
            ws_index = self.get_workspace_index(ws)
            return ws_index.add_many(items)
        finally:
            _release_flock(fh)

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: str = "",
        scope: str = "workspace",
        workspace: str | None = None,
        min_score: float | None = None,
        confirmed: bool | None = None,
    ) -> SearchResult | EmptyContext:
        """Perform hierarchical semantic search (`scope="workspace"` or `scope="global"`).

        - `scope="workspace"` (default): Queries only the isolated workspace branch. If the
          branch is empty or has no hits >= `min_score`, returns a deterministic `EmptyContext`
          (zero context pollution) whose `.requires_confirmation()` is `True`.
        - `scope="global"`: Queries all workspace branches and labels each hit with
          `[workspace=<name>]`. Enforces Human-in-the-Loop confirmation when configured or
          when `confirmed=False` is passed.
        """
        ws = self._resolve_workspace(workspace)
        if scope == "global":
            stores = self.all_workspace_indices()
            return self.hierarchy.search_global(
                stores=stores,
                query=query,
                top_k=top_k,
                category=category,
                min_score=min_score,
                origin_workspace=ws,
                confirmed=confirmed,
            )

        ws_index = self.get_workspace_index(ws)
        return self.hierarchy.search_workspace(
            ws_index=ws_index,
            query=query,
            top_k=top_k,
            category=category,
            min_score=min_score,
        )

    def search_formatted(
        self,
        query: str,
        top_k: int = 5,
        category: str = "",
        scope: str = "workspace",
        workspace: str | None = None,
        min_score: float | None = None,
        confirmed: bool | None = None,
    ) -> str:
        """Run `search()` and format the result as a human/LLM-readable markdown report."""
        res = self.search(
            query=query,
            top_k=top_k,
            category=category,
            scope=scope,
            workspace=workspace,
            min_score=min_score,
            confirmed=confirmed,
        )
        return res.format()

    def requires_confirmation(
        self,
        query_or_result: str | SearchResult | EmptyContext | list[SearchHit],
        workspace: str | None = None,
        min_score: float | None = None,
        category: str = "",
    ) -> bool:
        """Check whether a local workspace query requires HITL confirmation to escalate globally."""
        threshold = self.min_score if min_score is None else min_score
        if isinstance(query_or_result, str):
            local_res = self.search(
                query=query_or_result,
                top_k=1,
                category=category,
                scope="workspace",
                workspace=workspace,
                min_score=threshold,
            )
            return local_res.requires_confirmation()
        return requires_confirmation(query_or_result, min_score=threshold)

    def list(
        self,
        category: str = "",
        limit: int = 50,
        scope: str = "workspace",
        workspace: str | None = None,
    ) -> list[dict[str, Any]]:
        """List recent documents in a workspace branch or across all global workspaces."""
        cat_filter = category.strip().lower()
        if scope == "global":
            stores = self.all_workspace_indices()
            out: list[dict[str, Any]] = []
            for ws_name, ws_index in sorted(stores.items()):
                ws_index._check_stale_and_reload()
                items = ws_index.docs
                if cat_filter:
                    items = [d for d in items if d.get("category", "").lower() == cat_filter]
                for d in items[-limit:]:
                    entry = dict(d)
                    entry["workspace"] = ws_name
                    entry["workspace_label"] = f"[workspace={ws_name}]"
                    out.append(entry)
            return out

        ws = self._resolve_workspace(workspace)
        ws_index = self.get_workspace_index(ws)
        ws_index._check_stale_and_reload()
        items = ws_index.docs
        if cat_filter:
            items = [d for d in items if d.get("category", "").lower() == cat_filter]
        return [dict(d, workspace=ws) for d in items[-limit:]]

    def list_formatted(
        self,
        category: str = "",
        limit: int = 50,
        scope: str = "workspace",
        workspace: str | None = None,
    ) -> str:
        """Format `list()` output as a concise markdown summary."""
        items = self.list(category=category, limit=limit, scope=scope, workspace=workspace)
        if not items:
            return "memory_list: 0 entries."
        lines = [f"Last {len(items)} entries ({scope}):"]
        for d in items:
            txt = str(d.get("text", ""))
            if len(txt) > 160:
                txt = txt[:160] + " [...]"
            ws_prefix = f"[workspace={d['workspace']}] " if scope == "global" else ""
            lines.append(
                f"- {ws_prefix}[{d.get('category', 'note')}] "
                f"{d.get('source', 'manual')} ({d.get('time', '')}): {txt}"
            )
        return chr(10).join(lines)

    def delete_doc(self, doc_id: str, workspace: str | None = None) -> dict[str, Any]:
        """Delete a document by ID and rebuild its workspace FAISS index without re-embedding."""
        ws = self._resolve_workspace(workspace)
        acquired, fh = _try_flock_nonblock(self.base_dir, ws)
        if not acquired:
            return {"ok": False, "error": f"Lock busy for workspace {ws}"}
        try:
            ws_index = self.get_workspace_index(ws)
            return ws_index.delete_doc(doc_id)
        finally:
            _release_flock(fh)

    def update_doc(
        self,
        doc_id: str,
        text: str | None = None,
        category: str | None = None,
        source: str | None = None,
        tags: Sequence[str] | str | None = None,
        workspace: str | None = None,
    ) -> dict[str, Any]:
        """Update an existing document and refresh its FAISS vector if the text changed."""
        ws = self._resolve_workspace(workspace)
        acquired, fh = _try_flock_nonblock(self.base_dir, ws)
        if not acquired:
            return {"ok": False, "error": f"Lock busy for workspace {ws}"}
        try:
            ws_index = self.get_workspace_index(ws)
            return ws_index.update_doc(
                doc_id=doc_id,
                text=text,
                category=category,
                source=source,
                tags=tags,
            )
        finally:
            _release_flock(fh)

    def nearest_neighbors(
        self,
        doc_id: str,
        top_k: int = 6,
        workspace: str | None = None,
    ) -> _List[tuple[float, dict[str, Any]]]:
        """Find top-K nearest neighbors to `doc_id` within `workspace` via reconstructed vectors."""
        ws = self._resolve_workspace(workspace)
        ws_index = self.get_workspace_index(ws)
        return ws_index.nearest_neighbors(doc_id=doc_id, top_k=top_k)

    def merge_pending(
        self,
        workspace: str | None = None,
        force: bool = False,
        mtime_delay: float | None = None,
    ) -> dict[str, Any]:
        """Merge queued `_pending/<ws>/*.md` files into workspace FAISS indices."""
        res = self.guard.merge_pending_to_faiss(
            embedder=self.embedder,
            workspace=workspace,
            force=force,
            mtime_delay=mtime_delay,
        )
        for ws_idx in self._cache.values():
            ws_idx._check_stale_and_reload()
        return res

    def heal_workspace(self, workspace: str | None = None) -> dict[str, Any]:
        """Heal any `*.sync-conflict-*` files in a workspace branch via lossless union merge."""
        ws = self._resolve_workspace(workspace)
        acquired, fh = _try_flock_nonblock(self.base_dir, ws)
        if not acquired:
            return {"ok": False, "workspace": ws, "error": f"Lock busy for {ws}"}
        try:
            res = heal_workspace_conflicts(
                self.base_dir / ws,
                embedder=self.embedder,
                policy=self.sync_policy,
            )
            if ws in self._cache:
                self._cache[ws]._load()
            return {"ok": True, **res}
        finally:
            _release_flock(fh)

    def health(self) -> dict[str, Any]:
        """Return diagnostic health snapshot of workspaces, locks, pending queue, and peers."""
        stores = self.all_workspace_indices()
        blocked, detail = self.guard.is_blocked()
        locks_status: dict[str, str] = {}
        for ws_name in sorted(stores.keys()):
            acquired, fh = _try_flock_nonblock(self.base_dir, ws_name)
            if acquired:
                _release_flock(fh)
                locks_status[ws_name] = "free"
            else:
                locks_status[ws_name] = "busy"

        return {
            "base_dir": str(self.base_dir),
            "default_workspace": self.default_workspace,
            "workspaces_count": len(stores),
            "workspaces": {
                ws_name: {
                    "docs": len(st.docs),
                    "faiss_ntotal": int(st.index.ntotal) if st.index is not None else 0,
                    "lock_path": str(_lock_path_for_workspace(self.base_dir, ws_name)),
                    "lock": locks_status.get(ws_name, "unknown"),
                }
                for ws_name, st in sorted(stores.items())
            },
            "pending_total": pending_count(self.base_dir),
            "pending_by_workspace": pending_health(self.base_dir),
            "guard_blocked": blocked,
            "guard_detail": detail or "ok",
            "sync_conflict": has_sync_conflict(self.base_dir),
            "peers": [
                {
                    "node_id": p.node_id,
                    "endpoint_url": p.endpoint_url,
                    "enabled": p.enabled,
                }
                for p in self.peers
            ],
        }
