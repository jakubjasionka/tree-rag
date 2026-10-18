"""Universal standalone delta-based RAG synchronization (`sync.py`).

Provides transport-agnostic delta sync (`DeltaTransport`, `LocalFsTransport`, `HttpTransport`)
and configurable conflict policies (`FirstWriteWins`, `LastWriteWins`, `UnionMerge`)
without any external dependency on Syncthing.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from scoped_memory.atomic import _atomic_write_json
from scoped_memory.guard import DEFAULT_PEER_URI, PeerNode
from scoped_memory.index import (
    ALLOWED_CATEGORIES,
    CATEGORY_TO_LAYER,
    CATEGORY_TO_PILLAR,
    WorkspaceIndex,
)
from scoped_memory.sanitize import normalize_text, sanitize, scrub_secrets, text_hash
from scoped_memory.store import ScopedMemoryStore

NEAR_DUP_DEFAULT_THRESHOLD = 0.955


class DimensionMismatchError(ValueError):
    """Raised when a synchronized document vector or peer index fails dimension validation."""


class ConflictPolicy(StrEnum):
    """Conflict resolution policy for multi-node RAG synchronization."""

    UNION_MERGE = "union_merge"
    FIRST_WRITE_WINS = "first_write_wins"
    LAST_WRITE_WINS = "last_write_wins"

    # Alias descriptors matching specification naming
    UnionMerge = "union_merge"
    FirstWriteWins = "first_write_wins"
    LastWriteWins = "last_write_wins"

    def __call__(self) -> ConflictPolicy:
        return self


FirstWriteWins = ConflictPolicy.FIRST_WRITE_WINS
LastWriteWins = ConflictPolicy.LAST_WRITE_WINS
UnionMerge = ConflictPolicy.UNION_MERGE


def _resolve_conflict_policy(policy: ConflictPolicy | str | None) -> ConflictPolicy:
    if policy is None:
        env_pol = os.environ.get("TREE_RAG_SYNC_CONFLICT_POLICY", "union").strip().lower()
        return _resolve_conflict_policy(env_pol)
    if isinstance(policy, ConflictPolicy):
        return policy
    key = str(policy).strip().lower().replace("-", "_")
    mapping = {
        "union": ConflictPolicy.UNION_MERGE,
        "union_merge": ConflictPolicy.UNION_MERGE,
        "unionmerge": ConflictPolicy.UNION_MERGE,
        "first": ConflictPolicy.FIRST_WRITE_WINS,
        "first_write": ConflictPolicy.FIRST_WRITE_WINS,
        "first_write_wins": ConflictPolicy.FIRST_WRITE_WINS,
        "firstwritewins": ConflictPolicy.FIRST_WRITE_WINS,
        "last": ConflictPolicy.LAST_WRITE_WINS,
        "last_write": ConflictPolicy.LAST_WRITE_WINS,
        "last_write_wins": ConflictPolicy.LAST_WRITE_WINS,
        "lastwritewins": ConflictPolicy.LAST_WRITE_WINS,
    }
    if key in mapping:
        return mapping[key]
    raise ValueError(f"Unsupported conflict policy: {policy!r}")


class Doc(dict[str, Any]):
    """Dictionary representation of a synchronized RAG document and optional vector."""


@dataclass
class PushResult:
    """Result returned by `DeltaTransport.push_delta()`."""

    accepted: int
    rejected: int = 0
    conflicts_resolved: int = 0
    head_hash: str = ""
    ok: bool = True

    @property
    def pushed(self) -> int:
        return self.accepted


@dataclass
class SyncReport:
    """Structured telemetry report returned by `RAGSyncEngine.sync_workspace()`."""

    pushed: int
    pulled: int
    conflicts_resolved: int
    conflict_policy_used: str
    duration_ms: float
    rejected: int = 0
    workspace: str = ""
    last_sync_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace": self.workspace,
            "pushed": self.pushed,
            "pulled": self.pulled,
            "conflicts_resolved": self.conflicts_resolved,
            "conflict_policy_used": self.conflict_policy_used,
            "duration_ms": self.duration_ms,
            "rejected": self.rejected,
            "last_sync_hash": self.last_sync_hash,
        }


def _doc_fingerprint(doc: Mapping_or_Dict) -> str:
    """Compute a deterministic version fingerprint `(id, text_hash, updated_at, category)`."""
    doc_id = str(doc.get("id", ""))
    txt = scrub_secrets(str(doc.get("text", "")))
    h = text_hash(txt)
    upd = str(doc.get("updated_at") or doc.get("time") or "")
    cat = str(doc.get("category", "note"))
    return hashlib.sha256(f"{doc_id}|{h}|{upd}|{cat}".encode()).hexdigest()[:24]


Mapping_or_Dict = dict[str, Any]


def _compute_workspace_hash(docs: Sequence[dict[str, Any]]) -> str:
    """Compute a deterministic state digest over a workspace's document list."""
    if not docs:
        return ""
    fps = sorted(_doc_fingerprint(d) for d in docs)
    return hashlib.sha256("|".join(fps).encode("utf-8")).hexdigest()[:24]


def _sanitize_doc(raw_doc: dict[str, Any]) -> Doc:
    """Scrub secrets from all text/source fields of a document before push or pull."""
    cleaned = Doc(raw_doc)
    raw_text = str(cleaned.get("text", ""))
    scrubbed_text = normalize_text(scrub_secrets(raw_text))
    cleaned["text"] = scrubbed_text
    cleaned["source"] = scrub_secrets(str(cleaned.get("source", "sync") or "sync"))
    cat = str(cleaned.get("category", "note") or "note").lower()
    if cat not in ALLOWED_CATEGORIES:
        cat = "note"
    cleaned["category"] = cat
    cleaned["layer"] = str(cleaned.get("layer") or CATEGORY_TO_LAYER.get(cat, "L0"))
    cleaned["pillar"] = str(cleaned.get("pillar") or CATEGORY_TO_PILLAR.get(cat, "chat"))
    if not cleaned.get("id"):
        cleaned["id"] = "mem_" + uuid.uuid4().hex[:12]
    if not cleaned.get("time"):
        cleaned["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if "tags" not in cleaned or not isinstance(cleaned["tags"], list):
        cleaned["tags"] = WorkspaceIndex.extract_tags(scrubbed_text, cleaned["source"])
    else:
        cleaned["tags"] = [scrub_secrets(str(t)) for t in cleaned["tags"]]
    return cleaned


def _validate_doc_vector(
    doc: dict[str, Any],
    expected_dim: int,
    strict: bool = True,
) -> tuple[bool, np.ndarray | None]:
    """Validate optional vector/dim carried by `doc` against `expected_dim` before persistence."""
    declared_dim = doc.get("dim")
    if declared_dim is not None and int(declared_dim) != expected_dim:
        if strict:
            raise DimensionMismatchError(
                f"Vector dimension mismatch: doc declared dim={declared_dim}, "
                f"expected dim={expected_dim}"
            )
        return False, None

    raw_vec = doc.get("vector") if "vector" in doc else doc.get("embedding")
    if raw_vec is not None:
        arr = np.asarray(raw_vec, dtype="float32").reshape(-1)
        if int(arr.shape[0]) != expected_dim:
            if strict:
                raise DimensionMismatchError(
                    f"Vector dimension mismatch: vector has dim={arr.shape[0]}, "
                    f"expected dim={expected_dim}"
                )
            return False, None
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = (arr / norm).astype("float32")
        return True, arr

    return True, None


def _apply_delta_to_store(
    target_store: ScopedMemoryStore,
    workspace: str,
    delta: Sequence[dict[str, Any]],
    conflict_policy: ConflictPolicy,
    near_dup_threshold: float = NEAR_DUP_DEFAULT_THRESHOLD,
    strict_dim: bool = True,
) -> tuple[int, int, int]:
    """Apply incoming delta documents to `target_store` under `conflict_policy`.

    Returns `(applied_count, conflicts_resolved, rejected_count)`.
    """
    ws = sanitize(workspace)
    ws_index = target_store.get_workspace_index(ws)
    ws_index._check_stale_and_reload()
    expected_dim = int(target_store.embedder.dim)

    if ws_index.index is not None and int(ws_index.index.d) != expected_dim:
        if strict_dim:
            raise DimensionMismatchError(
                f"Target workspace '{ws}' index has dim={ws_index.index.d}, "
                f"expected dim={expected_dim}"
            )
        return 0, 0, len(delta)

    docs: list[dict[str, Any]] = [dict(d) for d in ws_index.docs]
    vecs: list[np.ndarray] = []
    if ws_index.index is not None and int(ws_index.index.ntotal) == len(docs) and len(docs) > 0:
        for i in range(len(docs)):
            vecs.append(ws_index.index.reconstruct(i).astype("float32"))
    elif docs:
        computed = target_store.embedder.embed_passages([str(d.get("text", "")) for d in docs])
        vecs = [row.astype("float32") for row in computed]

    applied = 0
    conflicts_resolved = 0
    rejected = 0

    for raw_incoming in delta:
        valid_dim, incoming_vec = _validate_doc_vector(
            raw_incoming, expected_dim=expected_dim, strict=strict_dim
        )
        if not valid_dim:
            rejected += 1
            continue

        clean_doc = _sanitize_doc(raw_incoming)
        # Strip transient transport keys before persisting to docs.json
        persisted_doc = {
            k: v for k, v in clean_doc.items() if k not in {"vector", "embedding", "dim"}
        }
        inc_text = str(persisted_doc.get("text", ""))
        if not inc_text:
            continue

        if incoming_vec is None:
            incoming_vec = target_store.embedder.embed_passages([inc_text])[0].astype("float32")
            if int(incoming_vec.shape[0]) != expected_dim:
                if strict_dim:
                    raise DimensionMismatchError(
                        f"Embedder produced dim={incoming_vec.shape[0]}, expected {expected_dim}"
                    )
                rejected += 1
                continue

        inc_id = str(persisted_doc.get("id", ""))
        inc_hash = text_hash(inc_text)
        inc_ts = str(persisted_doc.get("updated_at") or persisted_doc.get("time") or "")

        # 1. Check for ID match (same document edited on two nodes)
        id_match_idx = next(
            (i for i, existing in enumerate(docs) if str(existing.get("id", "")) == inc_id),
            -1,
        )
        if id_match_idx >= 0:
            existing_doc = docs[id_match_idx]
            existing_hash = text_hash(str(existing_doc.get("text", "")))
            existing_ts = str(existing_doc.get("updated_at") or existing_doc.get("time") or "")

            if existing_hash == inc_hash and existing_doc.get("category") == persisted_doc.get(
                "category"
            ):
                # Identical document already present
                if conflict_policy == ConflictPolicy.UNION_MERGE:
                    conflicts_resolved += 1
                continue

            conflicts_resolved += 1
            if conflict_policy == ConflictPolicy.FIRST_WRITE_WINS:
                # Keep the earlier write; if incoming is strictly older, take it,
                # otherwise local first write wins and is preserved.
                if inc_ts and existing_ts and inc_ts < existing_ts:
                    docs[id_match_idx] = persisted_doc
                    vecs[id_match_idx] = incoming_vec
                    applied += 1
                continue

            if conflict_policy == ConflictPolicy.LAST_WRITE_WINS:
                if not existing_ts or not inc_ts or inc_ts >= existing_ts:
                    docs[id_match_idx] = persisted_doc
                    vecs[id_match_idx] = incoming_vec
                    applied += 1
                continue

            # UNION_MERGE with same ID but different text:
            # Check if near-duplicate; if distinct, keep both by assigning a merged ID
            sim = float(np.dot(vecs[id_match_idx], incoming_vec))
            if sim >= near_dup_threshold:
                if inc_ts > existing_ts:
                    docs[id_match_idx] = persisted_doc
                    vecs[id_match_idx] = incoming_vec
                continue
            disambiguated = dict(persisted_doc)
            disambiguated["id"] = f"{inc_id}_{inc_hash[:6]}"
            docs.append(disambiguated)
            vecs.append(incoming_vec)
            applied += 1
            continue

        # 2. Check for exact content hash match (same text saved independently on two nodes)
        hash_match_idx = next(
            (
                i
                for i, existing in enumerate(docs)
                if text_hash(str(existing.get("text", ""))) == inc_hash
            ),
            -1,
        )
        if hash_match_idx >= 0:
            existing_doc = docs[hash_match_idx]
            existing_ts = str(existing_doc.get("updated_at") or existing_doc.get("time") or "")
            conflicts_resolved += 1
            if conflict_policy == ConflictPolicy.LAST_WRITE_WINS and (
                not existing_ts or inc_ts >= existing_ts
            ):
                docs[hash_match_idx] = persisted_doc
                vecs[hash_match_idx] = incoming_vec
            elif conflict_policy == ConflictPolicy.FIRST_WRITE_WINS and (
                inc_ts and existing_ts and inc_ts < existing_ts
            ):
                docs[hash_match_idx] = persisted_doc
                vecs[hash_match_idx] = incoming_vec
            # Under UNION_MERGE (and default dedup), 1 copy is retained
            continue

        # 3. Check for cosine near-duplicate (>= near_dup_threshold, default 0.955)
        if vecs and conflict_policy == ConflictPolicy.UNION_MERGE:
            mat = np.vstack(vecs).astype("float32")
            sims = np.dot(mat, incoming_vec)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim >= near_dup_threshold:
                existing_tokens = set(str(docs[best_idx].get("text", "")).lower().split())
                inc_tokens = set(inc_text.lower().split())
                union_len = len(existing_tokens | inc_tokens)
                jaccard = len(existing_tokens & inc_tokens) / union_len if union_len else 1.0
                if expected_dim >= 64 or jaccard >= 0.85:
                    conflicts_resolved += 1
                    continue

        docs.append(persisted_doc)
        vecs.append(incoming_vec)
        applied += 1

    # Persist updated docs and FAISS index atomically
    new_index = faiss.IndexFlatIP(expected_dim)
    if vecs:
        new_index.add(np.vstack(vecs).astype("float32"))

    ws_index.docs = docs
    ws_index.index = new_index
    ws_index._persist()

    return applied, conflicts_resolved, rejected


class DeltaTransport(ABC):
    """Abstract transport interface for pushing and pulling workspace document deltas."""

    @abstractmethod
    def push_delta(self, workspace: str, delta: list[Doc]) -> PushResult:
        """Push a list of delta documents to the remote peer for `workspace`."""

    @abstractmethod
    def pull_delta(self, workspace: str, since_hash: str) -> list[Doc]:
        """Pull documents added or modified in `workspace` on the peer since `since_hash`."""


class LocalFsTransport(DeltaTransport):
    """Filesystem-backed `DeltaTransport` (`path <-> path` or `store <-> store`) for local sync."""

    def __init__(
        self,
        peer: ScopedMemoryStore | Path | str | None = None,
        conflict_policy: ConflictPolicy | str = ConflictPolicy.UNION_MERGE,
        near_dup_threshold: float = NEAR_DUP_DEFAULT_THRESHOLD,
        strict_dim: bool = True,
    ) -> None:
        self._peer_store: ScopedMemoryStore | None = None
        self._peer_dir: Path | None = None
        self.conflict_policy = _resolve_conflict_policy(conflict_policy)
        self.near_dup_threshold = near_dup_threshold
        self.strict_dim = strict_dim
        self._snapshots: dict[str, dict[str, set[str]]] = {}
        if peer is not None:
            self.bind_peer(peer)

    def bind_peer(
        self,
        peer: ScopedMemoryStore | Path | str,
        fallback_embedder: Any = None,
    ) -> None:
        """Bind a target `ScopedMemoryStore` or filesystem directory path as the peer."""
        if isinstance(peer, ScopedMemoryStore):
            self._peer_store = peer
            self._peer_dir = peer.base_dir
        elif isinstance(peer, (Path, str)):
            self._peer_dir = Path(peer).resolve()
            self._peer_dir.mkdir(parents=True, exist_ok=True)
            if self._peer_store is None and fallback_embedder is not None:
                self._peer_store = ScopedMemoryStore(
                    base_dir=self._peer_dir,
                    embedder=fallback_embedder,
                )

    @property
    def peer_store(self) -> ScopedMemoryStore:
        if self._peer_store is None:
            if self._peer_dir is None:
                raise RuntimeError("LocalFsTransport has no bound peer store or directory")
            self._peer_store = ScopedMemoryStore(base_dir=self._peer_dir)
        return self._peer_store

    def _state_file(self, ws: str) -> Path:
        return self.peer_store.base_dir / f".sync_snapshots_{sanitize(ws)}.json"

    def _load_snapshots(self, ws: str) -> dict[str, set[str]]:
        ws_slug = sanitize(ws)
        mem = self._snapshots.setdefault(ws_slug, {})
        sf = self._state_file(ws_slug)
        if sf.exists():
            try:
                raw = json.loads(sf.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    for k, v in raw.items():
                        if isinstance(v, list):
                            mem[k] = set(str(x) for x in v)
            except Exception:
                pass
        return mem

    def record_snapshot(self, workspace: str, state_hash: str | None = None) -> str:
        """Record current document fingerprints in peer's `workspace` under `state_hash`."""
        ws = sanitize(workspace)
        ws_idx = self.peer_store.get_workspace_index(ws)
        ws_idx._check_stale_and_reload()
        fps = {_doc_fingerprint(d) for d in ws_idx.docs}
        digest = state_hash or _compute_workspace_hash(ws_idx.docs)
        if digest:
            snaps = self._load_snapshots(ws)
            snaps[digest] = fps
            _atomic_write_json(
                self._state_file(ws),
                {k: sorted(v) for k, v in snaps.items()},
            )
        return digest

    def list_peer_workspaces(self) -> list[str]:
        """List all workspace names present on the peer store."""
        return sorted(self.peer_store.all_workspace_indices().keys())

    def push_delta(self, workspace: str, delta: list[Doc]) -> PushResult:
        """Scrub secrets, validate vector dimensions, and merge `delta` into peer's workspace."""
        ws = sanitize(workspace)
        scrubbed_delta = [_sanitize_doc(d) for d in delta]
        # Preserve vector & dim fields for dimension validation in _apply_delta_to_store
        for orig, clean in zip(delta, scrubbed_delta, strict=False):
            if "vector" in orig:
                clean["vector"] = orig["vector"]
            if "embedding" in orig:
                clean["embedding"] = orig["embedding"]
            if "dim" in orig:
                clean["dim"] = orig["dim"]

        applied, conflicts, rejected = _apply_delta_to_store(
            target_store=self.peer_store,
            workspace=ws,
            delta=scrubbed_delta,
            conflict_policy=self.conflict_policy,
            near_dup_threshold=self.near_dup_threshold,
            strict_dim=self.strict_dim,
        )
        head_hash = self.record_snapshot(ws)
        return PushResult(
            accepted=applied,
            rejected=rejected,
            conflicts_resolved=conflicts,
            head_hash=head_hash,
            ok=rejected == 0,
        )

    def pull_delta(self, workspace: str, since_hash: str) -> list[Doc]:
        """Pull documents added or updated on the peer since `since_hash` with scrubbed text."""
        ws = sanitize(workspace)
        ws_idx = self.peer_store.get_workspace_index(ws)
        ws_idx._check_stale_and_reload()

        snaps = self._load_snapshots(ws)
        known_fps: set[str] = snaps.get(since_hash, set()) if since_hash else set()

        peer_dim = (
            int(ws_idx.index.d)
            if ws_idx.index is not None and int(ws_idx.index.ntotal) > 0
            else int(self.peer_store.embedder.dim)
        )

        out: list[Doc] = []
        for i, raw_doc in enumerate(ws_idx.docs):
            fp = _doc_fingerprint(raw_doc)
            if since_hash and fp in known_fps:
                continue
            clean = _sanitize_doc(raw_doc)
            clean["dim"] = peer_dim
            if ws_idx.index is not None and i < int(ws_idx.index.ntotal):
                clean["vector"] = ws_idx.index.reconstruct(i).astype("float32").tolist()
            out.append(clean)
        return out


class HttpTransport(DeltaTransport):
    """HTTP `DeltaTransport` client (placeholder/reference implementation for remote peer nodes)."""

    def __init__(
        self,
        endpoint_url: str = DEFAULT_PEER_URI,
        timeout_sec: float = 10.0,
        handler: Callable[[str, str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.endpoint_url = (endpoint_url or DEFAULT_PEER_URI).rstrip("/")
        self.timeout_sec = timeout_sec
        self._handler = handler

    def push_delta(self, workspace: str, delta: list[Doc]) -> PushResult:
        """Push scrubbed delta documents to `{endpoint_url}/sync/{workspace}/push`."""
        ws = sanitize(workspace)
        payload_docs: list[dict[str, Any]] = [dict(_sanitize_doc(d)) for d in delta]
        body: dict[str, Any] = {"workspace": ws, "delta": payload_docs}

        if self._handler is not None:
            resp = self._handler("POST", f"{self.endpoint_url}/sync/{ws}/push", body)
            return PushResult(
                accepted=int(resp.get("accepted", len(payload_docs))),
                rejected=int(resp.get("rejected", 0)),
                conflicts_resolved=int(resp.get("conflicts_resolved", 0)),
                head_hash=str(resp.get("head_hash", "")),
                ok=bool(resp.get("ok", True)),
            )

        if ".example" in self.endpoint_url:
            # Placeholder offline mode for RFC 2606 example endpoints
            digest = _compute_workspace_hash(payload_docs)
            return PushResult(accepted=len(payload_docs), head_hash=digest, ok=True)

        import requests

        r = requests.post(
            f"{self.endpoint_url}/sync/{ws}/push",
            json=body,
            timeout=self.timeout_sec,
        )
        r.raise_for_status()
        data = r.json()
        return PushResult(
            accepted=int(data.get("accepted", 0)),
            rejected=int(data.get("rejected", 0)),
            conflicts_resolved=int(data.get("conflicts_resolved", 0)),
            head_hash=str(data.get("head_hash", "")),
            ok=bool(data.get("ok", True)),
        )

    def pull_delta(self, workspace: str, since_hash: str) -> list[Doc]:
        """Pull delta documents from `{endpoint_url}/sync/{workspace}/pull?since_hash=...`."""
        ws = sanitize(workspace)
        if self._handler is not None:
            resp = self._handler(
                "GET",
                f"{self.endpoint_url}/sync/{ws}/pull",
                {"workspace": ws, "since_hash": since_hash},
            )
            raw_list = resp.get("delta", [])
            return [_sanitize_doc(d) for d in raw_list if isinstance(d, dict)]

        if ".example" in self.endpoint_url:
            return []

        import requests

        r = requests.get(
            f"{self.endpoint_url}/sync/{ws}/pull",
            params={"since_hash": since_hash},
            timeout=self.timeout_sec,
        )
        r.raise_for_status()
        data = r.json()
        raw_list = data.get("delta", [])
        return [_sanitize_doc(d) for d in raw_list if isinstance(d, dict)]


class RAGSyncEngine:
    """Standalone delta-based multi-node synchronization engine for Tree-RAG stores."""

    def __init__(
        self,
        local_store: ScopedMemoryStore,
        peer: ScopedMemoryStore | PeerNode | Path | str | None = None,
        transport: DeltaTransport | None = None,
        conflict_policy: ConflictPolicy | str | None = ConflictPolicy.UNION_MERGE,
        near_dup_threshold: float = NEAR_DUP_DEFAULT_THRESHOLD,
        strict_dim: bool = True,
    ) -> None:
        self.local_store = local_store
        self.peer = peer
        self.conflict_policy = _resolve_conflict_policy(conflict_policy)
        self.near_dup_threshold = near_dup_threshold
        self.strict_dim = strict_dim

        if transport is None:
            if isinstance(peer, PeerNode):
                transport = HttpTransport(endpoint_url=peer.endpoint_url)
            elif isinstance(peer, str) and peer.startswith(("http://", "https://")):
                transport = HttpTransport(endpoint_url=peer)
            elif peer is not None:
                transport = LocalFsTransport(
                    peer=peer,
                    conflict_policy=self.conflict_policy,
                    near_dup_threshold=self.near_dup_threshold,
                    strict_dim=self.strict_dim,
                )
            else:
                raise ValueError("Either `peer` or `transport` must be provided to RAGSyncEngine")
        elif isinstance(transport, LocalFsTransport):
            transport.conflict_policy = self.conflict_policy
            transport.near_dup_threshold = self.near_dup_threshold
            transport.strict_dim = self.strict_dim
            if transport._peer_store is None and transport._peer_dir is None and peer is not None:
                if isinstance(peer, (ScopedMemoryStore, Path, str)):
                    transport.bind_peer(peer, fallback_embedder=local_store.embedder)

        self.transport: DeltaTransport = transport
        self._state_path = self.local_store.base_dir / ".rag_sync_state.json"
        self.last_sync_hash: dict[str, str] = {}
        self._synced_local_fps: dict[str, set[str]] = {}
        self._load_sync_state()

    def _load_sync_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                hashes = raw.get("last_sync_hash", {})
                if isinstance(hashes, dict):
                    self.last_sync_hash = {str(k): str(v) for k, v in hashes.items()}
                fps = raw.get("synced_local_fps", {})
                if isinstance(fps, dict):
                    self._synced_local_fps = {
                        str(k): set(str(x) for x in v)
                        for k, v in fps.items()
                        if isinstance(v, list)
                    }
        except Exception:
            pass

    def _save_sync_state(self) -> None:
        payload = {
            "last_sync_hash": self.last_sync_hash,
            "synced_local_fps": {k: sorted(v) for k, v in self._synced_local_fps.items()},
        }
        _atomic_write_json(self._state_path, payload)

    def _collect_local_delta(self, workspace: str, since_hash: str) -> list[Doc]:
        """Collect local documents created or updated in `workspace` since `since_hash`."""
        ws = sanitize(workspace)
        ws_idx = self.local_store.get_workspace_index(ws)
        ws_idx._check_stale_and_reload()

        known_fps = self._synced_local_fps.get(ws, set()) if since_hash else set()
        local_dim = (
            int(ws_idx.index.d)
            if ws_idx.index is not None and int(ws_idx.index.ntotal) > 0
            else int(self.local_store.embedder.dim)
        )

        delta: list[Doc] = []
        for i, raw_doc in enumerate(ws_idx.docs):
            fp = _doc_fingerprint(raw_doc)
            if since_hash and fp in known_fps:
                continue
            clean = _sanitize_doc(raw_doc)
            # Persist scrubbed text back if raw_doc had un-scrubbed secrets
            if raw_doc.get("text") != clean["text"]:
                raw_doc["text"] = clean["text"]
            clean["dim"] = local_dim
            if ws_idx.index is not None and i < int(ws_idx.index.ntotal):
                clean["vector"] = ws_idx.index.reconstruct(i).astype("float32").tolist()
            delta.append(clean)
        return delta

    def _validate_peer_dimension(self, workspace: str) -> None:
        """Validate that local and peer embedder/index dimensions match before syncing."""
        if not self.strict_dim:
            return
        local_dim = int(self.local_store.embedder.dim)
        if isinstance(self.peer, ScopedMemoryStore):
            peer_dim = int(self.peer.embedder.dim)
            if peer_dim != local_dim:
                raise DimensionMismatchError(
                    f"Dimension mismatch between local store (dim={local_dim}) "
                    f"and peer store (dim={peer_dim}) for workspace '{workspace}'"
                )
        if isinstance(self.transport, LocalFsTransport) and self.transport._peer_store is not None:
            peer_dim = int(self.transport.peer_store.embedder.dim)
            if peer_dim != local_dim:
                raise DimensionMismatchError(
                    f"Dimension mismatch between local store (dim={local_dim}) "
                    f"and transport peer store (dim={peer_dim}) for workspace '{workspace}'"
                )

    def sync_workspace(
        self,
        workspace: str,
        direction: str = "bidirectional",
    ) -> SyncReport:
        """Synchronize `workspace` delta since `last_sync_hash` using `self.conflict_policy`.

        Supported `direction` values:
        - `"bidirectional"` (default): pull delta from peer and push local delta to peer.
        - `"push"` / `"one_way"` / `"local_to_peer"`: push local delta to peer only.
        - `"pull"` / `"peer_to_local"`: pull remote delta from peer into local store only.
        """
        t0 = time.perf_counter()
        ws = sanitize(workspace)
        dir_norm = (direction or "bidirectional").strip().lower()
        if dir_norm not in {
            "bidirectional",
            "push",
            "pull",
            "one_way",
            "local_to_peer",
            "peer_to_local",
        }:
            raise ValueError(f"Unsupported sync direction: {direction!r}")

        self._validate_peer_dimension(ws)

        do_pull = dir_norm in {"bidirectional", "pull", "peer_to_local"}
        do_push = dir_norm in {"bidirectional", "push", "one_way", "local_to_peer"}

        since_hash = self.last_sync_hash.get(ws, "")

        # 1. Compute local and remote deltas relative to `since_hash` before mutating either side
        local_delta = self._collect_local_delta(ws, since_hash) if do_push else []
        remote_delta = self.transport.pull_delta(ws, since_hash) if do_pull else []

        pulled_count = 0
        pushed_count = 0
        conflicts_total = 0
        rejected_total = 0

        # 2. Apply remote delta to local store
        if do_pull and remote_delta:
            applied_local, conflicts_local, rejected_local = _apply_delta_to_store(
                target_store=self.local_store,
                workspace=ws,
                delta=remote_delta,
                conflict_policy=self.conflict_policy,
                near_dup_threshold=self.near_dup_threshold,
                strict_dim=self.strict_dim,
            )
            pulled_count += applied_local
            conflicts_total += conflicts_local
            rejected_total += rejected_local

        # 3. Push local delta to peer
        if do_push and local_delta:
            push_res = self.transport.push_delta(ws, local_delta)
            pushed_count += push_res.accepted
            conflicts_total += push_res.conflicts_resolved
            rejected_total += push_res.rejected

        # 4. Record converged snapshot hash on both local and peer
        ws_idx = self.local_store.get_workspace_index(ws)
        ws_idx._check_stale_and_reload()
        current_local_fps = {_doc_fingerprint(d) for d in ws_idx.docs}
        new_sync_hash = _compute_workspace_hash(ws_idx.docs) or hashlib.sha256(
            f"{ws}:empty".encode()
        ).hexdigest()[:24]

        if isinstance(self.transport, LocalFsTransport):
            self.transport.record_snapshot(ws, state_hash=new_sync_hash)

        self.last_sync_hash[ws] = new_sync_hash
        self._synced_local_fps[ws] = current_local_fps
        self._save_sync_state()

        duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return SyncReport(
            pushed=pushed_count,
            pulled=pulled_count,
            conflicts_resolved=conflicts_total,
            conflict_policy_used=self.conflict_policy.value,
            duration_ms=duration_ms,
            rejected=rejected_total,
            workspace=ws,
            last_sync_hash=new_sync_hash,
        )

    def sync_all(
        self,
        workspaces: Sequence[str] | None = None,
        direction: str = "bidirectional",
    ) -> dict[str, SyncReport]:
        """Synchronize multiple workspaces and return `{workspace: SyncReport}`."""
        if workspaces is None:
            discovered = set(self.local_store.all_workspace_indices().keys())
            if isinstance(self.transport, LocalFsTransport):
                discovered.update(self.transport.list_peer_workspaces())
            elif isinstance(self.peer, ScopedMemoryStore):
                discovered.update(self.peer.all_workspace_indices().keys())
            target_workspaces = sorted(discovered)
        else:
            target_workspaces = [sanitize(w) for w in workspaces]

        reports: dict[str, SyncReport] = {}
        for ws in target_workspaces:
            reports[ws] = self.sync_workspace(ws, direction=direction)
        return reports
